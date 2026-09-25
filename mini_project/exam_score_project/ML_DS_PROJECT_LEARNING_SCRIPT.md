# ML/DS Project Learning Script

## A reusable walkthrough for building a prediction project on a new dataset

This guide explains the ideas demonstrated by the exam score project and gives you a
repeatable way to apply them to another dataset. It is written as both a study guide and a
talk track: you can follow the sections while building a notebook, or use the quoted
paragraphs to explain your choices to someone else.

The most important principle is: **choose methods because the data and the question need
them.** A good project does not use every technique in this guide. PCA, Lasso selection,
date features, interaction features, and even scaling are conditional choices, not boxes to
check.

---

## 1. Start with the question, not the algorithm

Before opening a modeling library, write down what a prediction would help someone decide.
The question determines the target, the unit of analysis, what information is allowed, and
how success should be measured.

### Questions to answer

1. **Who or what is one row?** One student, one transaction, one house, one day, or one
   customer account?
2. **What should the model predict?** This is the target (`y`). It must be a value that is
   available in the training data and meaningful to the user.
3. **At what moment will the prediction be made?** List what is known at that moment. Any
   column only known afterward may leak the answer.
4. **What decision follows the prediction?** A useful model should improve a decision, not
   only produce a score.
5. **What kind of target is it?** A number suggests regression; a category suggests
   classification; a future sequence suggests forecasting; an unlabelled grouping task may
   call for clustering.

### Talk track

> “I am predicting **[target]** for **[one row represents]** using information available
> **[when the prediction is made]**. The prediction will support **[decision]**. I will
> measure success with **[metric]**, because it reflects the cost of being wrong for this
> use case.”

### Why the prediction time matters: leakage

**Target leakage** happens when a training feature contains information that would not be
available at prediction time, or directly encodes the target. For example, predicting
whether a loan defaults using a field populated only after collection activity begins makes
the test score look good while teaching the model an impossible shortcut.

Make a small table before modeling:

| Column | Meaning | Available at prediction time? | Keep, transform, or exclude? |
|---|---|---:|---|
| `...` | ... | Yes / No | ... |

An identifier can be useful for joining or auditing records without being a useful feature.
Do not feed an ID into a model just because it is numeric. A row number can let a model
memorize the dataset, and often has no stable meaning for a new example.

---

## 2. Load the data and check what is actually there

The first pass is data understanding, not feature engineering. Confirm that the file loaded
as expected, that each column has the expected type, and that the target and rows mean what
you think they mean.

```python
import pandas as pd

df = pd.read_csv("data/raw/my_data.csv")
print(df.shape)                 # row count and column count
display(df.head())              # inspect real examples
df.info()                       # types and non-null counts
display(df.isna().sum())        # missing values per column
display(df.describe(include="all").T)
```

Then ask:

- Are there duplicate rows or duplicate entities?
- Are dates parsed as dates, numbers as numbers, and categories as categories?
- Are missing values represented consistently (`NaN`, blank strings, `"Unknown"`, etc.)?
- Are values outside their expected range? Are there impossible dates or negative amounts?
- Is the target missing? If so, those rows usually cannot be used as labeled examples.
- Is there a column that was created after the target event?
- Does one row represent one independent example, or do rows share a person, household,
  device, hospital, or time series?

### Missing data is a property to understand

Missingness can be random, or it can carry information. For example, an unrecorded income
may have a different meaning from an income of zero. Count missing values and investigate
patterns before choosing a treatment. Common options include dropping a small number of
unusable rows, median imputation for numeric values, most-frequent imputation for categories,
an explicit “Unknown” category, or adding a missingness indicator when the fact that a value
is missing may itself matter.

Imputation values must be **learned from training data only**. If the full dataset is used
to calculate a median before the split, information from the test set has influenced the
model. A scikit-learn pipeline helps prevent that mistake.

---

## 3. Explore the target and predictors

Exploratory data analysis (EDA) is the process of using summaries and visualizations to
understand distributions, relationships, data quality, and possible modeling choices. It
helps generate questions; it does not prove that a relationship is causal.

For a numeric target, inspect its distribution, range, and relationship to numeric features.
For a categorical target, inspect class counts and whether one class is rare. Look at
categories and the number of distinct values. Plot only what helps answer a question; a chart
is useful when you can say what decision it informs.

Examples of useful questions:

- Is the target heavily skewed or capped at a maximum?
- Are a few outliers data errors, rare but valid cases, or influential examples?
- Do multiple columns measure nearly the same thing?
- Are categories rare enough that encoding them individually could be brittle?
- Does the target vary by a group, date, or region in a way the split must respect?

Correlation measures a particular kind of association, mainly linear association for
Pearson correlation. A high correlation does not show causation, and a low correlation does
not rule out a nonlinear or interaction relationship.

---

## 4. Establish a baseline, then split the data correctly

A baseline is a simple reference point. For regression, a `DummyRegressor` can predict the
training mean or median. For classification, a `DummyClassifier` can predict the most common
class. A complex model should beat a reasonable baseline on data it did not train on.

### Pick a split that reflects future use

- **Independent, similarly distributed rows:** a random train/test split can be suitable.
- **Classification:** consider stratifying the split so class proportions are represented
  in both sets, especially when the target is imbalanced.
- **Repeated people, sites, or devices:** split by group so records from the same group do
  not appear in both train and test.
- **Future prediction from past data:** train on earlier dates and test on later dates.
  Randomly mixing future and past can overstate performance.

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["target", "record_id"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

The test set is a final check of expected performance on unseen examples. Do not repeatedly
make decisions based on it. Use cross-validation on the training portion for model and
hyperparameter comparisons, then evaluate the selected approach on the held-out test set.
For time-ordered problems, use time-aware validation instead of ordinary random folds.

**The central rule:** split first; fit imputers, scalers, encoders, PCA, feature selectors,
and the model using training data only.

---

## 5. Create features when they express useful domain information

Feature engineering means turning available raw columns into representations that make
useful patterns easier for a model to learn. It should have a reason grounded in the
problem.

### Date-derived features

A timestamp may become elapsed time, day of week, month, or time since an event. Choose the
reference date carefully. If every training row uses a fixed snapshot date, the prediction
service needs a consistent prediction-time date too. A hard-coded old date can make a model
stale. Do not derive a feature using a date that would be unknown when predictions are made.

### Interaction features

An interaction lets the effect of one feature depend on another. For example, the value of
study hours may depend on attendance. A basic linear model using `study_hours` and
`attendance_pct` separately assumes their effects add independently. Adding
`study_hours * attendance_pct` gives it a way to represent a joint effect.

Create an interaction when there is a plausible domain reason or evidence in training data.
It can increase complexity and overfit, particularly with small datasets. Evaluate whether
it improves validation performance; a larger correlation with the target alone is not proof
that it will generalize.

### Keep transformations reproducible

Any custom transformation used during training must also run during inference with the same
logic. Put reusable transformations in importable Python code and include them in the saved
pipeline. Avoid notebook-only definitions and machine-specific file paths.

---

## 6. Preprocess by column type

Models generally need numerical arrays, so raw columns need appropriate treatment. Different
columns need different transformations; `ColumnTransformer` applies the right pipeline to
each selected group.

### Numeric columns

- Impute missing values, often with the median because it is less affected by extreme values
  than the mean.
- Scale when the model or method is sensitive to feature magnitude. Standard scaling
  subtracts the training mean and divides by the training standard deviation.

Scaling is often helpful for linear models with regularization, distance-based methods,
gradient-based models, and PCA. It is usually unnecessary for tree-based models such as
random forests. Scaling does not make the underlying units disappear; it changes what a
coefficient or distance means.

### Categorical columns

- **Nominal categories** have no natural order, such as city or product type. One-hot
  encoding makes indicator columns. `handle_unknown="ignore"` lets inference accept a
  category not seen during training, though the model has not learned a specific effect for
  that category.
- **Ordinal categories** have a meaningful order, such as low, medium, high. Ordinal
  encoding should explicitly state that order. Do not assign arbitrary numbers to nominal
  categories: the model could treat those numbers as meaningful distances or rankings.

### Reusable preprocessing skeleton

Replace the example column names with those from your data. Add or remove branches based on
what your dataset contains.

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_columns),
    ("categorical", categorical_pipe, categorical_columns),
])
```

If you have ordinal columns, create a separate branch with `OrdinalEncoder(categories=[...])`
and supply each category order explicitly. If your data has text, images, or complex
timestamps, those need suitable additional transformations; this simple tabular template
does not solve those cases automatically.

---

## 7. PCA: compress correlated numeric measurements when it helps

Principal Component Analysis (PCA) rotates numeric features into new axes called principal
components. The first component captures the direction of greatest variance; later
components capture remaining variance while being orthogonal to earlier ones. Keeping fewer
components compresses the inputs.

In the exam score example, three mock test columns measure similar underlying performance.
PCA reduced them to one component that captured much of their shared variation.

PCA can help reduce redundancy, noise, or a very wide numeric feature set. It is not a
default requirement. It can make results harder to explain because components combine the
original columns. It is sensitive to scale, so numeric columns generally need scaling
before PCA. Fit PCA on training data only, inside the pipeline.

Use PCA when validation results or a real dimensionality constraint justify it. Choose the
number of components using training-only validation, explained variance as a diagnostic, and
the needs of the use case. Do not apply ordinary PCA to unencoded categories or mix all
columns indiscriminately.

---

## 8. Feature selection: keep useful inputs without peeking at the test set

Feature selection chooses which transformed inputs a model uses. It can make a model smaller,
reduce noise, or improve interpretability. It can also discard useful signal if done
carelessly.

The example uses **Lasso**, a linear regression method with an L1 penalty. The penalty can
shrink some coefficients to zero. `SelectFromModel` uses those coefficients to select
features. Because Lasso penalizes coefficient size, inputs should generally be on comparable
scales first. The penalty strength (`alpha`) controls how aggressively coefficients shrink
and should be selected using cross-validation on training data, not copied as a magic
constant.

Other methods include regularized models, permutation importance, and domain-informed
selection. Correlated features can make selection unstable: a method may keep one of several
similar columns and drop the others without proving that the discarded variables are
useless. Feature selection does not establish causality.

Keep the selector inside the full pipeline so it is fitted independently within each
training fold. Never run selection on the entire dataset before making the test split.

---

## 9. Put learned steps and the estimator in one pipeline

A pipeline is a sequence of transformations followed by a model. It provides one interface
for fitting and predicting and reduces the chance that training and serving preprocess data
differently. `Pipeline` and `ColumnTransformer` also allow cross-validation to refit learned
preprocessing on each fold.

```python
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

model_pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", LinearRegression()),
])

model_pipeline.fit(X_train, y_train)
predictions = model_pipeline.predict(X_test)
```

Add feature creation before preprocessing if it is part of your design. Add PCA or a feature
selector only if the data and validation results support doing so. Start with a simple
baseline model, then compare sensible alternatives. The “best” model depends on predictive
quality, stability, interpretability, latency, and the cost of errors.

Linear regression is useful as an interpretable baseline for a numeric target. Its
coefficients are conditional associations under the model, not automatically causal effects.
If inputs were standardized, a coefficient is per standard deviation of that transformed
feature, not per original unit. Check assumptions and residuals; predictions can exceed the
target’s natural range unless the modeling approach accounts for bounds.

---

## 10. Evaluate with metrics that match the task

Use a held-out test set to estimate how the selected approach may perform on unseen data.
Always compare to a baseline and inspect errors, not only one headline score.

### Regression metrics

- **MAE (mean absolute error):** average absolute distance between prediction and truth. It
  is in the target’s units and is easy to explain.
- **RMSE (root mean squared error):** also in target units, but penalizes large errors more
  heavily.
- **R²:** compares error to predicting the target mean. It can be negative on a test set and
  does not directly express error in business units.

For classification, accuracy can be misleading when classes are imbalanced. Depending on
the goal, examine precision, recall, F1, confusion matrices, and threshold behavior. For
forecasting, use time-aware splits and metrics suited to the target scale and forecast
horizon.

Inspect residuals (`actual - predicted`), errors by useful slices, and failures near the
range limits. Ask whether the model is systematically wrong for particular dates, groups,
regions, or target values. A strong average score can hide poor performance for an important
subset.

### Validation checklist

- Does it beat the baseline by a meaningful amount?
- Are train and validation scores far apart, suggesting overfitting?
- Does performance hold across folds or time periods?
- Are metrics reported in understandable units?
- Are the largest errors acceptable for the decision being made?
- Does the test split resemble the examples that will arrive later?

---

## 11. Save and reload the complete fitted pipeline

Saving the full fitted pipeline preserves learned preprocessing as well as the estimator:
imputation values, category vocabulary, scaling statistics, PCA directions, selected
features, and model parameters. Serving code can then pass raw columns through the same
transformations used during training.

```python
import joblib

joblib.dump(model_pipeline, "models/model_pipeline.joblib")
loaded_pipeline = joblib.load("models/model_pipeline.joblib")
```

Custom transformer classes need to live in importable `.py` modules that exist in the
inference environment. A notebook-defined class can be difficult to unpickle from an API
process. Record the training data version, feature schema, library versions, metric results,
and model version alongside the artifact. Joblib/pickle artifacts can execute code while
loading; only load artifacts from a trusted source.

Test a freshly loaded pipeline on a raw example. Compare that result with prediction from the
still-live fitted pipeline. Also check that missing values and unseen categories behave as
intended.

---

## 12. Separate the model from the application interface

An API can validate raw requests, convert them into a one-row DataFrame, call
`pipeline.predict`, and return a structured response. A UI can collect values and call the
API. Keeping the prediction logic in one service gives different clients the same behavior.

Keep these responsibilities clear:

- **Model pipeline:** feature transformations and prediction.
- **API:** input contract, validation, model loading, prediction response, errors, health.
- **UI:** collect user inputs, send requests, display results.
- **Tests:** model behavior, API contract, and important edge cases.
- **Deployment:** package and run the services in a reproducible environment.

Validation should happen at the API boundary too: required fields, types, valid ranges,
allowed categories, and date formats. Distinguish invalid user input from an internal model
failure. A liveness check and a readiness check answer different questions: “is the process
up?” versus “is the model loaded and ready?”

Docker, Streamlit, and cloud deployment are useful application and MLOps topics, but they
are not substitutes for defining a valid target, preventing leakage, or evaluating a model.
For real personal data, consider access control, encryption in transit, data retention, and
who can view predictions before exposing an endpoint publicly.

---

## 13. What the exam score project demonstrates, and what to adapt

| Topic | How it appears in the example | How to decide for a new dataset |
|---|---|---|
| Regression target | Predicts a numeric final score | Choose regression only for a numeric outcome; classification and forecasting need different evaluation and splits |
| Train/test split | Random 80/20 split before preprocessing | Use group- or time-aware splits when rows are related or time ordered |
| Imputation | Median for numeric study hours | Inspect missingness and choose a suitable strategy per column |
| Feature creation | Enrollment duration and study/attendance interaction | Derive features from domain logic available at prediction time |
| Encoding | One-hot city and ordered income bracket | Separate nominal and genuinely ordered categories |
| Scaling | StandardScaler for numeric inputs and PCA inputs | Use for scale-sensitive methods; usually skip for tree models |
| PCA | Compresses three correlated test scores | Use only when compression or redundancy reduction helps; trade off interpretability |
| Lasso selection | Removes weak inputs after preprocessing | Tune regularization with training-only validation; watch correlated variables |
| Linear regression | Simple interpretable estimator | Keep as a baseline; compare alternatives and inspect assumptions |
| Pipeline persistence | Saves preprocessing and model together | Save schema/version information and test a fresh reload |
| API and UI | FastAPI predicts; Streamlit calls it | Keep user interface separate from authoritative inference logic |

The example notebook reports one held-out split. For a stronger new project, add
cross-validation on the training set, compare against a baseline, inspect residuals, and
write down the limitations of the data. Treat a synthetic or small educational dataset as a
learning exercise, not proof that the model is ready for high-impact decisions.

---

## 14. A practical build order for your next dataset

Use this sequence as your project checklist. Write down why each step is included.

1. State the prediction question, user, prediction time, target, and success metric.
2. Describe one row and audit the feature columns for leakage and identifiers.
3. Load the data; inspect shape, types, missingness, duplicates, ranges, and target values.
4. Explore distributions and relationships that affect modeling decisions.
5. Choose a split that matches how future examples will arrive.
6. Create a simple baseline before trying advanced methods.
7. Build numeric and categorical preprocessing inside a `ColumnTransformer`.
8. Add domain-based features only when they are available at prediction time.
9. Add PCA, feature selection, or other complexity only when justified and validated.
10. Fit with cross-validation on training data; select metrics that express useful error.
11. Evaluate once on the held-out test data; inspect errors and important data slices.
12. Save the complete fitted pipeline; reload it and check raw-input predictions.
13. Expose the pipeline through an API only if another application needs it.
14. Document data limitations, reproducibility details, and how the model should be used.

### Final explanation template

> “This project predicts **[target]** from **[available inputs]** for **[intended user or
> decision]**. I used **[split strategy]** because **[reason]**. Numeric and categorical
> features are treated with **[preprocessing choices]** inside a single pipeline, so those
> transformations are learned on training data and repeated consistently at prediction
> time. I chose **[model]** as **[baseline or validated choice]** and report **[metrics]** on
> **[test/validation design]**. The main limitations are **[data, validation, or deployment
> limitations]**.”

If you can fill in that explanation with evidence from your own dataset, you understand the
project decisions more deeply than if you merely reproduce the same list of algorithms.
