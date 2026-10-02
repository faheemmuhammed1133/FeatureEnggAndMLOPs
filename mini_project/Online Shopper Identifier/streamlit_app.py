"""Streamlit client for single-session and CSV batch shopper-intention scoring."""

from __future__ import annotations

import math
import os

import pandas as pd
import requests
import streamlit as st

from src.schema import EXCLUDED_COLUMNS, FEATURE_COLUMNS, FLOAT_COLUMNS, INTEGER_COLUMNS

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
BATCH_SIZE = 500
MAX_UPLOAD_ROWS = 20_000
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

SAMPLE_SESSION = {
    "Administrative": 2,
    "Administrative_Duration": 25.0,
    "Informational": 0,
    "Informational_Duration": 0.0,
    "ProductRelated": 18,
    "ProductRelated_Duration": 620.5,
    "BounceRates": 0.01,
    "ExitRates": 0.03,
    "SpecialDay": 0.0,
    "Month": "May",
    "OperatingSystems": 2,
    "Browser": 2,
    "Region": 1,
    "TrafficType": 2,
    "VisitorType": "Returning_Visitor",
    "Weekend": False,
}


def _canonicalize_headers(frame: pd.DataFrame) -> pd.DataFrame:
    """Trim headers and match feature names without regard to case."""
    expected_names = {name.casefold(): name for name in FEATURE_COLUMNS}
    expected_names.update({name.casefold(): name for name in EXCLUDED_COLUMNS})
    rename = {}
    for column in frame.columns:
        clean_name = str(column).strip()
        rename[column] = expected_names.get(clean_name.casefold(), clean_name)
    return frame.rename(columns=rename)


def _missing(value: object) -> bool:
    result = pd.isna(value)
    return bool(result) if not hasattr(result, "__len__") else False


def _convert_weekend(value: object, row_number: int) -> bool | None:
    if _missing(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"true", "yes", "y", "1"}:
        return True
    if normalized in {"false", "no", "n", "0"}:
        return False
    raise ValueError(
        f"Row {row_number}: Weekend must be true/false, yes/no, or 1/0; received {value!r}."
    )


def _session_payload(row: pd.Series, row_number: int) -> dict[str, object]:
    """Convert pandas/numpy values into JSON-safe values for the API."""
    payload: dict[str, object] = {}
    for column in FEATURE_COLUMNS:
        value = row[column]
        if column == "Weekend":
            payload[column] = _convert_weekend(value, row_number)
        elif _missing(value):
            payload[column] = None
        elif column in INTEGER_COLUMNS:
            numeric_value = float(value)
            if not math.isfinite(numeric_value) or not numeric_value.is_integer():
                raise ValueError(
                    f"Row {row_number}: {column} must contain a whole-number code or count."
                )
            payload[column] = int(numeric_value)
        elif column in FLOAT_COLUMNS:
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(f"Row {row_number}: {column} must be a finite number.")
            payload[column] = numeric_value
        else:
            text_value = str(value).strip()
            payload[column] = text_value or None
    return payload


def _single_session_tab() -> None:
    with st.form("shopper_session"):
        st.subheader("Enter one session")
        left, middle, right = st.columns(3)
        with left:
            administrative = st.number_input("Administrative pages", 0, 100, 2)
            administrative_duration = st.number_input(
                "Administrative duration", 0.0, 10000.0, 25.0
            )
            informational = st.number_input("Informational pages", 0, 100, 0)
            informational_duration = st.number_input(
                "Informational duration", 0.0, 10000.0, 0.0
            )
        with middle:
            product_related = st.number_input("Product related pages", 0, 1000, 18)
            product_related_duration = st.number_input(
                "Product related duration", 0.0, 100000.0, 620.5
            )
            bounce_rates = st.number_input(
                "Bounce rate", 0.0, 1.0, 0.01, step=0.001, format="%.3f"
            )
            exit_rates = st.number_input(
                "Exit rate", 0.0, 1.0, 0.03, step=0.001, format="%.3f"
            )
        with right:
            special_day = st.number_input(
                "Special day proximity", 0.0, 1.0, 0.0, step=0.1
            )
            month = st.selectbox(
                "Month",
                ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                index=3,
            )
            visitor_type = st.selectbox(
                "Visitor type", ["Returning_Visitor", "New_Visitor", "Other"]
            )
            weekend = st.checkbox("Weekend session")

        st.subheader("Session context")
        context_left, context_right = st.columns(2)
        with context_left:
            operating_systems = st.number_input("Operating system code", 1, 20, 2)
            browser = st.number_input("Browser code", 1, 30, 2)
        with context_right:
            region = st.number_input("Region code", 1, 30, 1)
            traffic_type = st.number_input("Traffic type code", 1, 50, 2)

        submitted = st.form_submit_button(
            "Estimate purchase probability", type="primary"
        )

    if not submitted:
        return

    payload = {
        "Administrative": int(administrative),
        "Administrative_Duration": float(administrative_duration),
        "Informational": int(informational),
        "Informational_Duration": float(informational_duration),
        "ProductRelated": int(product_related),
        "ProductRelated_Duration": float(product_related_duration),
        "BounceRates": float(bounce_rates),
        "ExitRates": float(exit_rates),
        "SpecialDay": float(special_day),
        "Month": month,
        "OperatingSystems": int(operating_systems),
        "Browser": int(browser),
        "Region": int(region),
        "TrafficType": int(traffic_type),
        "VisitorType": visitor_type,
        "Weekend": bool(weekend),
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=20)
        response.raise_for_status()
        result = response.json()
        st.metric(
            "Estimated purchase probability",
            f"{100 * result['purchase_probability']:.1f}%",
        )
        st.write(
            f"At the validation-selected threshold ({result['decision_threshold']:.3f}), "
            f"the model classifies this session as **"
            f"{'purchase likely' if result['predicted_purchase'] else 'below threshold'}**."
        )
        st.caption(
            f"Model version: {result['model_version']}. A probability is a ranking signal, "
            "not a guarantee that an intervention will cause a purchase."
        )
    except requests.RequestException as error:
        st.error(f"Prediction request failed: {error}")


def _batch_csv_tab(default_model_threshold: float) -> None:
    st.subheader("Score many sessions from a CSV")
    st.write(
        "Upload one row per session. The original `Revenue` and `PageValues` columns, if "
        "present, are ignored for scoring. Blank feature cells are handled by the model's "
        "training-fitted imputers."
    )
    st.info(
        "This model scores session summaries, not unique customers. Include an external "
        "session/customer ID column if you need to join results back to your own records; "
        "that ID is kept out of model inputs."
    )

    template = pd.DataFrame([SAMPLE_SESSION], columns=FEATURE_COLUMNS).to_csv(index=False)
    st.download_button(
        "Download CSV template",
        data=template.encode("utf-8"),
        file_name="shopper_sessions_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file", type=["csv"], key="shopper_batch_csv"
    )
    if uploaded_file is None:
        return
    if uploaded_file.size > MAX_UPLOAD_BYTES:
        st.error("The CSV exceeds the 15 MB demo upload limit.")
        return

    try:
        source_frame = _canonicalize_headers(pd.read_csv(uploaded_file))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        st.error(f"Could not read this CSV: {error}")
        return

    if source_frame.empty:
        st.error("The CSV has no data rows.")
        return
    if len(source_frame) > MAX_UPLOAD_ROWS:
        st.error(f"This demo accepts at most {MAX_UPLOAD_ROWS:,} session rows per upload.")
        return

    missing_columns = [name for name in FEATURE_COLUMNS if name not in source_frame.columns]
    if missing_columns:
        st.error(
            "The CSV is missing required model columns: " + ", ".join(missing_columns)
        )
        st.caption("Use the downloadable template to see the expected feature headers.")
        return

    for column in FLOAT_COLUMNS + INTEGER_COLUMNS:
        original = source_frame[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & converted.isna()
        if invalid.any():
            bad_rows = (invalid[invalid].index[:5] + 2).tolist()
            st.error(
                f"Column {column} contains non-numeric values near CSV row(s) "
                f"{bad_rows}. Fix those cells and upload again."
            )
            return
        source_frame[column] = converted

    id_columns = [
        name
        for name in source_frame.columns
        if name not in FEATURE_COLUMNS and name not in EXCLUDED_COLUMNS
    ]
    id_choice = st.selectbox(
        "Identifier column (optional)",
        options=["Use row number"] + id_columns,
        help="This column is shown with results but is never sent to the model.",
    )
    nudge_floor = st.slider(
        "Minimum likelihood for ‘Needs a nudge’",
        min_value=0.0,
        max_value=0.95,
        value=float(min(max(default_model_threshold, 0.0), 0.95)),
        step=0.01,
        help=(
            "Defaults to the model's validation-selected decision threshold. This is a "
            "review band, not a proven marketing policy."
        ),
    )
    high_minimum = min(nudge_floor + 0.01, 1.0)
    high_default = min(max(0.65, high_minimum), 1.0)
    high_cutoff = st.slider(
        "High purchase-likelihood cutoff",
        min_value=float(high_minimum),
        max_value=1.0,
        value=float(high_default),
        step=0.01,
    )

    st.dataframe(source_frame.head(5), hide_index=True, use_container_width=True)
    if not st.button("Score uploaded sessions", type="primary", key="score_batch"):
        return

    feature_frame = source_frame[FEATURE_COLUMNS]
    try:
        payloads = [
            _session_payload(row, index + 2)
            for index, (_, row) in enumerate(feature_frame.iterrows())
        ]
    except (TypeError, ValueError, OverflowError) as error:
        st.error(str(error))
        return

    predictions: list[dict[str, object]] = []
    progress = st.progress(0, text="Preparing batch prediction…")
    status = st.status(f"Scoring {len(payloads):,} sessions", expanded=True)
    try:
        for start in range(0, len(payloads), BATCH_SIZE):
            chunk = payloads[start : start + BATCH_SIZE]
            response = requests.post(
                f"{API_URL}/predict/batch",
                json={"sessions": chunk},
                timeout=90,
            )
            response.raise_for_status()
            chunk_predictions = response.json()["predictions"]
            if len(chunk_predictions) != len(chunk):
                raise RuntimeError("The API returned a different number of rows than requested.")
            predictions.extend(chunk_predictions)
            completed = len(predictions)
            progress.progress(
                completed / len(payloads),
                text=f"Scored {completed:,} of {len(payloads):,} sessions…",
            )
            status.write(f"Scored {completed:,} / {len(payloads):,}")
    except requests.RequestException as error:
        status.update(label="Batch scoring failed", state="error")
        response_detail = (
            getattr(error.response, "text", "")
            if error.response is not None
            else ""
        )
        st.error(f"Batch prediction failed: {error}. {response_detail}")
        return
    except (KeyError, RuntimeError, ValueError) as error:
        status.update(label="Batch scoring failed", state="error")
        st.error(f"Could not interpret the API response: {error}")
        return

    progress.progress(1.0, text=f"Finished scoring {len(predictions):,} sessions.")
    status.update(label=f"Scored {len(predictions):,} sessions", state="complete")

    scored = source_frame.drop(columns=EXCLUDED_COLUMNS, errors="ignore").copy()
    row_column = "Session row"
    while row_column in scored.columns:
        row_column = f"{row_column} (model row)"
    scored.insert(0, row_column, range(1, len(scored) + 1))
    probabilities = [float(item["purchase_probability"]) for item in predictions]
    scored["purchase_probability"] = probabilities
    scored["Purchase likelihood (%)"] = [round(probability * 100, 1) for probability in probabilities]
    scored["Model decision"] = [
        "Purchase likely" if item["predicted_purchase"] else "Below model threshold"
        for item in predictions
    ]
    scored["Priority group"] = [
        "High purchase likelihood"
        if probability >= high_cutoff
        else "Needs a nudge"
        if probability >= nudge_floor
        else "Low purchase likelihood"
        for probability in probabilities
    ]
    scored["decision_threshold"] = [float(item["decision_threshold"]) for item in predictions]
    scored["model_version"] = [str(item["model_version"]) for item in predictions]

    group_names = [
        "High purchase likelihood",
        "Needs a nudge",
        "Low purchase likelihood",
    ]
    counts = [int((scored["Priority group"] == name).sum()) for name in group_names]
    metrics = st.columns(4)
    metrics[0].metric("Sessions scored", f"{len(scored):,}")
    for column, label, count in zip(metrics[1:], group_names, counts):
        column.metric(label, f"{count:,}")

    display_columns = [row_column]
    if id_choice != "Use row number":
        display_columns.append(id_choice)
    display_columns.extend(
        name
        for name in [
            "Month",
            "VisitorType",
            "ProductRelated",
            "ProductRelated_Duration",
            "ExitRates",
            "BounceRates",
        ]
        if name in scored.columns and name not in display_columns
    )
    display_columns.extend(
        ["Purchase likelihood (%)", "Model decision", "Priority group"]
    )

    high_tab, nudge_tab, cold_tab, all_tab = st.tabs(
        [
            f"High likelihood ({counts[0]:,})",
            f"Needs a nudge ({counts[1]:,})",
            f"Low likelihood ({counts[2]:,})",
            "All scored rows",
        ]
    )
    for tab, group in zip((high_tab, nudge_tab, cold_tab), group_names):
        with tab:
            group_frame = scored[scored["Priority group"] == group]
            group_frame = group_frame.sort_values(
                "purchase_probability", ascending=(group == group_names[2])
            )
            if group_frame.empty:
                st.caption("No sessions fell into this group at the selected cutoffs.")
            else:
                st.dataframe(
                    group_frame[display_columns],
                    hide_index=True,
                    use_container_width=True,
                )
    with all_tab:
        ordered = scored.sort_values("purchase_probability", ascending=False)
        st.dataframe(ordered[display_columns], hide_index=True, use_container_width=True)

    st.download_button(
        "Download all scored sessions as CSV",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name="shopper_session_predictions.csv",
        mime="text/csv",
        key="download_scored_sessions",
    )
    st.caption(
        "Priority groups are probability bands for triage. They are not causal claims, "
        "and a high score does not prove that a promotion will change the outcome."
    )


st.set_page_config(
    page_title="Online Shopper Intention",
    page_icon="🛍️",
    layout="wide",
)
st.title("🛍️ Online Shopper Classifier")
st.caption(
    "Analyze real online shopping sessions and predict purchase likelihood using a trained machine learning model. Enter a session manually or upload a CSV to classify shoppers into High Purchase Likelihood, Potential Customer, or Low Purchase Likelihood based on their observed browsing behavior."
)

model_threshold = 0.319
with st.sidebar:
    st.subheader("API status")
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=3)
        health_response.raise_for_status()
        health = health_response.json()
        if health.get("model_loaded"):
            st.success(f"Model ready · version {health.get('model_version', 'unknown')}")
            info_response = requests.get(f"{API_URL}/model-info", timeout=3)
            info_response.raise_for_status()
            model_threshold = float(info_response.json()["decision_threshold"])
            st.caption(f"Validation-selected cutoff: {model_threshold:.3f}")
        else:
            st.warning("API is running, but the model is not ready.")
    except requests.RequestException as error:
        st.error(f"Cannot reach the API at {API_URL}")
        st.caption(str(error))

single_tab, batch_tab = st.tabs(["Single session", "Batch CSV upload"])
with single_tab:
    _single_session_tab()
with batch_tab:
    _batch_csv_tab(model_threshold)

st.divider()
st.caption(
    "PageValues is intentionally excluded from model inputs because of target-proxy risk. "
    "See the README and notebook for the modeling scope and limitations."
)
