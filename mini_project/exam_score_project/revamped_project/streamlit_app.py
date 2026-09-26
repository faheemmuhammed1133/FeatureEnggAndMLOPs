"""Streamlit client for the FastAPI shopper-intention service."""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="Online Shopper Intention",
    page_icon="🛍️",
    layout="wide",
)
st.title("🛍️ Online Shopper Purchase Intention")
st.caption(
    "Educational session-level classifier. It estimates the Revenue label from supplied "
    "session summaries. The original dataset is not timestamped event data, so this UI "
    "does not claim to predict intent early during a live session."
)

with st.sidebar:
    st.subheader("API status")
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=3)
        health_response.raise_for_status()
        health = health_response.json()
        if health.get("model_loaded"):
            st.success(f"Model ready · version {health.get('model_version', 'unknown')}")
        else:
            st.warning("API is running, but the model is not ready.")
    except requests.RequestException as error:
        st.error(f"Cannot reach the API at {API_URL}")
        st.caption(str(error))

with st.form("shopper_session"):
    st.subheader("Session activity")
    left, middle, right = st.columns(3)
    with left:
        administrative = st.number_input("Administrative pages", 0, 100, 2)
        administrative_duration = st.number_input("Administrative duration", 0.0, 10000.0, 25.0)
        informational = st.number_input("Informational pages", 0, 100, 0)
        informational_duration = st.number_input("Informational duration", 0.0, 10000.0, 0.0)
    with middle:
        product_related = st.number_input("Product related pages", 0, 1000, 18)
        product_related_duration = st.number_input("Product related duration", 0.0, 100000.0, 620.5)
        bounce_rates = st.number_input("Bounce rate", 0.0, 1.0, 0.01, step=0.001, format="%.3f")
        exit_rates = st.number_input("Exit rate", 0.0, 1.0, 0.03, step=0.001, format="%.3f")
    with right:
        special_day = st.number_input("Special day proximity", 0.0, 1.0, 0.0, step=0.1)
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

    submitted = st.form_submit_button("Estimate purchase probability", type="primary")

if submitted:
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
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        st.metric("Estimated purchase probability", f"{100 * result['purchase_probability']:.1f}%")
        st.write(
            f"Decision at the validation-selected threshold "
            f"({result['decision_threshold']:.3f}): "
            f"**{'Purchase' if result['predicted_purchase'] else 'No purchase'}**"
        )
        st.caption(
            f"Model version: {result['model_version']}. The threshold was selected to "
            "maximize validation F1 for this educational project; a real business should "
            "choose a threshold from its intervention costs and capacity."
        )
    except requests.RequestException as error:
        st.error(f"Prediction request failed: {error}")

st.divider()
st.caption(
    "PageValues is intentionally absent from the form and prediction API. See the README "
    "and notebook for the reason and the limits of this session-level dataset."
)
