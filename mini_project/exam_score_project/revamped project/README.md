# Online Shoppers Purchasing Intention

A documented binary-classification project using the attached online shopping sessions
dataset. It includes a learning notebook, a reproducible training command, a saved
scikit-learn pipeline, a FastAPI prediction service, a Streamlit client, and one Dockerfile
that runs both services.

For a dataset-independent explanation of the ML/DS fundamentals and how to adapt them, see
the [general project learning script](../ML_DS_PROJECT_LEARNING_SCRIPT.md).

## What the model predicts

Each row is one summarized shopping session. The target is `Revenue`: whether that session
ended with a purchase. This is **binary classification**, not regression. The supplied CSV
contains 12,330 sessions; 1,908 (about 15.5%) have `Revenue=True`, so the target is
imbalanced. A model that always says “no purchase” would be about 84.5% accurate while
missing every purchase, which is why this project reports precision, recall, F1,
average precision, and ROC AUC instead of accuracy alone.

### Scope and target-proxy caution

The CSV contains session-level summaries, not timestamped events. It supports an educational
estimate of the final session outcome; by itself, it does not support a claim that the model
can intervene early during a live session. The `PageValues` column is intentionally excluded
from the model and prediction API. It is a Google Analytics page-value measure associated
with pages visited before a transaction and could carry information too close to the outcome
for a genuine early-intent prediction. The notebook discusses this modeling risk. A real
deployment would need event-time snapshots and a precise prediction cutoff before claiming
real-time prediction.

The integer values in `OperatingSystems`, `Browser`, `Region`, and `TrafficType` are treated
as category codes, not ordered numeric measurements. `Month`, `VisitorType`, and `Weekend`
are also categorical. Numeric session counts, durations, rates, and `SpecialDay` are
imputed and scaled for logistic regression. The random forest comparison does not need
scaling. Preprocessing is inside each scikit-learn pipeline, so learned imputation,
scaling, and category encodings are fitted on training data only.

## Project layout

```text
revamped project/
├── data/raw/online_shoppers_intention.csv  # supplied dataset copy
├── notebooks/online_shoppers_intention.ipynb
├── src/shopper_pipeline.py                 # feature contract and candidate pipelines
├── src/training.py                         # split, selection, threshold, persistence
├── train.py                                # reproducible command-line training entry point
├── models/shopper_purchase_pipeline.joblib # created by training
├── artifacts/model_metadata.json           # validation/test metrics and threshold
├── app/main.py                             # FastAPI service
├── streamlit_app.py                        # API client UI
├── docker-entrypoint.sh
├── Dockerfile                              # single combined image
└── requirements.txt
```

## Use your existing Python environment

The project is under the current workspace and the existing environment is at
`/Users/muhammedfaheem/Desktop/AI_DS/.venv`. From a terminal:

```bash
cd "/Users/muhammedfaheem/Desktop/AI_DS/Feature Engineering/repo/mini_project/exam_score_project/revamped project"
source "/Users/muhammedfaheem/Desktop/AI_DS/.venv/bin/activate"
```

The current parent environment has the data science and API packages, but Streamlit is not
installed there. The project requirements include Streamlit. To install the project
dependencies into this environment, run:

```bash
python -m pip install -r requirements.txt
```

If you prefer not to modify the shared parent environment, create a project-local virtual
environment and install the same requirements there.

## Train the model

Run this once before starting the API or building the Docker image:

```bash
python train.py
```

Training performs a stratified 60/20/20 train/validation/test split. It compares a
class-weighted logistic regression with a class-weighted random forest using validation
average precision, selects the probability threshold that maximizes validation F1, refits
the selected pipeline on train plus validation, and evaluates once on the held-out test
partition. The output model and metrics are written to `models/` and `artifacts/`.

The original data contains 125 exact duplicate rows. The training script keeps them because
the file is session-level and does not provide a session identifier that would let us
determine whether identical summaries are duplicate records or distinct sessions. This is
recorded in model metadata as a limitation to revisit when better identifiers are available.

## Open the notebook in your running Jupyter server

The notebook is saved under `notebooks/online_shoppers_intention.ipynb` and reads the CSV
through a path relative to the project, so it does not depend on a machine-specific absolute
path. From Jupyter's root, navigate to `Feature Engineering/repo/mini_project/exam_score_project/revamped project/notebooks/` and open
`online_shoppers_intention.ipynb`. Use the Python kernel associated with the parent `.venv`.

You can also start Jupyter from the project directory:

```bash
jupyter notebook
```

Notebook sections explain the business question, target imbalance, data-quality checks,
leakage-aware split, categorical and numerical preprocessing, baseline, candidate models,
threshold choice, held-out evaluation, and model persistence. The source helpers are in
`src/` so training from the notebook and `python train.py` use the same feature contract.

## Run locally

Start the API in terminal 1:

```bash
uvicorn app.main:app --reload --port 8000
```

Start the UI in terminal 2:

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` for the form or `http://localhost:8000/docs` for the API
contract. The Streamlit app sends the same raw fields to FastAPI; it does not load or run
the model itself.

The API contract intentionally omits both `Revenue` and `PageValues`. Example request:

```json
{
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
  "Weekend": false
}
```

The response includes `purchase_probability`, `predicted_purchase`, the validation-selected
decision threshold, and model version. The threshold is a teaching default selected to
maximize validation F1. A real business should select it based on the cost of false alarms,
missed purchases, and the number of sessions it can act on.

## Build and run the one Docker image

After training has created the model and metadata:

```bash
docker build -t online-shopper-intention:1.0 .
docker run --rm -p 8000:8000 -p 8501:8501 online-shopper-intention:1.0
```

Open `http://localhost:8501` for the UI and `http://localhost:8000/docs` for FastAPI.
The container entrypoint starts the API, waits for its model-backed health endpoint, then
starts Streamlit. Docker runs as a non-root user.

## Data source and attribution

The supplied CSV is the **Online Shoppers Purchasing Intention Dataset** by Sakar and
Kastro, made available through the UCI Machine Learning Repository. UCI reports 12,330
sessions, with 84.5% negative and 15.5% positive labels, and describes the session measures
and Google Analytics rate/value fields. Dataset DOI: [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q).
The dataset is licensed CC BY 4.0; retain attribution if redistributing it.

## Limitations

- This is a learning project using historical, aggregated sessions, not a production
  purchase intervention system.
- `PageValues` is excluded because of target-proxy risk; the remaining finalized session
  summaries may still be unavailable at an earlier intervention time.
- The reported test metrics are estimates for this dataset and split, not a guarantee of
  performance on a different store, time period, or user population.
- The F1-selected threshold encodes no business cost model and should not be treated as a
  policy decision.
- The model is not causal: a high predicted probability does not show that changing a
  feature would cause a purchase.
