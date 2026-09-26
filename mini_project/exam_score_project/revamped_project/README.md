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
revamped_project/
├── data/raw/online_shoppers_intention.csv  # supplied dataset copy
├── notebooks/online_shoppers_intention.ipynb
├── src/schema.py                         # shared, lightweight feature contract
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
cd "/Users/muhammedfaheem/Desktop/AI_DS/Feature Engineering/repo/mini_project/exam_score_project/revamped_project"
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
path. From Jupyter's root, navigate to `Feature Engineering/repo/mini_project/exam_score_project/revamped_project/notebooks/` and open
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

## Score a CSV batch

In the Streamlit app, open **Batch CSV upload**, download the template, add one session per
row, and upload it. The file needs the 16 feature columns used by the model; header matching
ignores capitalization and surrounding spaces. Blank feature cells are imputed by the
training-fitted preprocessing pipeline. Other columns can be included for identifiers, but
they are not sent to the model. `Revenue` and `PageValues` are ignored and removed from the
scored export.

The upload screen shows progress while it sends bounded groups of rows to the
`POST /predict/batch` endpoint, then presents separate tables for **High purchase
likelihood**, **Needs a nudge**, and **Low purchase likelihood**. It also provides a table of
all scored rows and a downloadable CSV. The nudge floor defaults to the model's
validation-selected classification threshold; the high-likelihood cutoff defaults to 65%.
Both are editable triage settings, not proven marketing policies. The API accepts up to 500
sessions per request, and the UI splits larger uploads into those chunks.

Each row represents a summarized session. The supplied dataset does not identify unique
customers; include your own session or customer ID column to link scored rows to your
records. The ID is shown in results but is excluded from model inputs. Avoid uploading
personally identifying or otherwise sensitive data to a public demo service.

## Build and run the one Docker image

After training has created the model and metadata:

```bash
docker build -t online-shopper-intention:1.0 .
docker run --rm -p 8000:8000 -p 8501:8501 online-shopper-intention:1.0
```

Open `http://localhost:8501` for the UI and `http://localhost:8000/docs` for FastAPI.
The container entrypoint starts the API, waits for its model-backed health endpoint, then
starts Streamlit. Locally Streamlit uses port 8501 by default; the entrypoint uses Render's
`PORT` environment variable when one is provided. Docker runs as a non-root user.

## Deploy on Render's free web service

The project can deploy from this GitHub monorepo using its Dockerfile:

1. In Render, choose **New → Web Service**, connect GitHub, and select
   `faheemmuhammed1133/FeatureEnggAndMLOPs` on branch `main`.
2. Set **Root Directory** to
   `mini_project/exam_score_project/revamped_project` and **Runtime** to `Docker`.
   Use `./Dockerfile` as the Dockerfile path if Render asks for it. The Docker build context
   should be this project directory.
3. Choose the **Free** instance type. Leave the Docker command/start command and `API_URL`
   unset. The entrypoint starts both services; Streamlit calls FastAPI at
   `http://localhost:8000` inside the same container.
4. Deploy and open the `onrender.com` URL shown in the service dashboard.

The public Render port comes from Render's `PORT` variable; the entrypoint honors it for
Streamlit while keeping FastAPI on its internal port 8000. The UI is public, while the
FastAPI `/docs` endpoint is not separately exposed in this single-service layout. To make
the API independently public, deploy it as a separate web service and configure the UI to
use that service's URL.

Render's free web services have 512 MB RAM, spin down after 15 minutes without traffic, and
can take about a minute to start again. They have ephemeral filesystems; this project reads
the model bundled in the image and does not need to persist uploads. The combined app may
be memory-constrained on the free plan, so check the service logs if it restarts. See
[Render's free-instance limits](https://render.com/docs/free) and
[Docker deployment settings](https://render.com/docs/docker).

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
