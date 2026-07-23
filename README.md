# medstats

An embeddable Python analytics component for medical statistics workflows in liver disease research.

**Not a REST service.** Backend code imports it directly, calls a function with a `pandas.DataFrame`, and gets back a plain JSON-safe dict to store or return to the frontend.

## Install

```bash
pip install git+https://github.com/bxjxis/medstats.git
```

## Functions

| Area | Function | What it does |
|---|---|---|
| Survival | `cox_regression` | Cox PH model — HR, 95% CI, p-value, C-index |
| Survival | `kaplan_meier` | KM curve coordinates, median survival, log-rank p |
| Regression | `logistic_regression` | Logistic regression — OR, 95% CI, p-value, predicted probabilities |
| Evaluation | `roc_analysis` | ROC curve, AUC with DeLong CI, optional two-model comparison |
| Evaluation | `calibration_analysis` | Calibration curve, Brier score, Hosmer-Lemeshow test |
| Evaluation | `decision_curve_analysis` | Net benefit curves vs probability threshold |
| Preprocessing | `multiple_imputation` | Iterative (MICE-style) imputation for numeric columns |
| Causal | `propensity_score_matching` | 1:1 nearest-neighbour PSM with SMD balance table |
| Causal | `iptw_weights` | Stabilized IPTW weights with weighted SMD balance table |

## Quick start

```python
import pandas as pd
from medstats import cox_regression, kaplan_meier, roc_analysis

df = pd.read_csv("your_data.csv")

result = cox_regression(
    df,
    duration_col="time",
    event_col="event",
    covariates=["age", "albumin", "stage"],
    categorical_cols=["stage"],   # string columns → dummy-encoded automatically
)

import json
json.dumps(result)   # always safe — no numpy types, NaN → null
```

## Output envelope

Every function returns the same wrapper:

```json
{
  "method": "cox_regression",
  "status": "ok",
  "params": {"duration_col": "time", "event_col": "event", "covariates": ["age", "albumin"]},
  "n_input": 300,
  "n_used": 285,
  "dropped_rows": 15,
  "warnings": [],
  "result": { ... }
}
```

`status` is `"warning"` when `warnings` is non-empty. Backend can store the full envelope; frontend reads from `result` and surfaces `warnings`.

## Row identity

Functions that return per-row outputs (`logistic_regression`, `multiple_imputation`, `iptw_weights`, `propensity_score_matching`) include an `id` field on every row so the backend can JOIN results back to database records.

Pass a primary-key column with `id_col=`:

```python
result = logistic_regression(df, "outcome", ["age", "albumin"], id_col="patient_id")
# result["result"]["predicted_probabilities"]
# → [{"id": "P001", "predicted_probability": 0.23}, ...]
```

If `id_col` is omitted, `df.index` values are used.

## Categorical covariates

String or category columns must be declared in `categorical_cols`. They are dummy-encoded with `pd.get_dummies(drop_first=True)` before modelling. Passing a string column without declaring it raises `ValueError`.

```python
logistic_regression(df, "outcome", ["age", "stage"], categorical_cols=["stage"])
# term names include "stage_early", "stage_advanced", etc.
```

## Function reference

### `cox_regression`

```python
cox_regression(df, duration_col, event_col, covariates, categorical_cols=None)
```

```json
"result": {
  "c_index": 0.74,
  "n_events": 88,
  "terms": [
    {"name": "age", "coef": 0.031, "hr": 1.031, "hr_ci_low": 1.01, "hr_ci_high": 1.05, "p": 0.002}
  ]
}
```

---

### `kaplan_meier`

```python
kaplan_meier(df, duration_col, event_col, group_col=None)
```

```json
"result": {
  "groups": [
    {"name": "early", "times": [0, 3, 6], "survival": [1.0, 0.97, 0.94], "median_survival": 42.0}
  ],
  "overall_logrank_p": 0.003
}
```

`times` / `survival` → frontend chart x/y. Two groups: two-sample log-rank. Three+: multivariate log-rank overall p.

---

### `logistic_regression`

```python
logistic_regression(df, outcome_col, covariates, add_intercept=True,
                    categorical_cols=None, id_col=None)
```

```json
"result": {
  "converged": true,
  "n_events": 72,
  "pseudo_r2": 0.18,
  "terms": [
    {"name": "age", "coef": 0.04, "or": 1.04, "or_ci_low": 1.01, "or_ci_high": 1.07, "p": 0.008}
  ],
  "predicted_probabilities": [
    {"id": "P001", "predicted_probability": 0.23}
  ]
}
```

---

### `roc_analysis`

```python
roc_analysis(df, label_col, score_col, compare_score_col=None)
```

`score_col` must be in `[0, 1]`. DeLong variance uses ddof=1 (Sun & Xu 2014).

```json
"result": {
  "auc": 0.81, "auc_ci_low": 0.76, "auc_ci_high": 0.86,
  "fpr": [...], "tpr": [...], "thresholds": [...],
  "compare": {
    "score_col": "model_b", "auc": 0.75,
    "fpr": [...], "tpr": [...],
    "delong_z": 1.82, "delong_p": 0.069
  }
}
```

`fpr` / `tpr` → frontend ROC chart x/y.

---

### `calibration_analysis`

```python
calibration_analysis(df, label_col, score_col, n_bins=10, hosmer_lemeshow=True)
```

```json
"result": {
  "mean_predicted": [0.05, 0.15, ...],
  "fraction_observed": [0.04, 0.18, ...],
  "brier_score": 0.19,
  "hosmer_lemeshow": {"statistic": 8.4, "p": 0.39, "df": 8}
}
```

`mean_predicted` / `fraction_observed` → calibration plot x/y. Perfect calibration = diagonal.

---

### `decision_curve_analysis`

```python
decision_curve_analysis(df, label_col, score_cols, threshold_min=0.01,
                        threshold_max=0.99, n_thresholds=99)
```

```json
"result": {
  "thresholds": [0.01, 0.02, ...],
  "treat_all": [0.24, 0.23, ...],
  "treat_none": [0.0, 0.0, ...],
  "models": [{"name": "score_a", "net_benefit": [0.22, ...]}],
  "prevalence": 0.24
}
```

`thresholds` → x-axis; net benefit values → y-axis per curve.

---

### `multiple_imputation`

```python
multiple_imputation(df, target_cols, m=1, random_state=42, id_col=None)
```

Iterative imputation (sklearn `IterativeImputer`). Returns `m` datasets. Rubin pooling is **not** included — do that in the calling layer if needed.

```json
"result": {
  "m": 1,
  "missing_rates": {"albumin": 0.05},
  "imputed_cols": ["albumin"],
  "datasets": [
    [{"id": 0, "albumin": 34.2}, {"id": 1, "albumin": 38.7}]
  ]
}
```

---

### `propensity_score_matching`

```python
propensity_score_matching(df, treatment_col, covariates, caliper=0.2,
                          replace=False, random_state=42, categorical_cols=None)
```

`caliper` = fraction of PS standard deviation. `treated_id` / `control_id` are original `df.index` values.

```json
"result": {
  "n_pairs": 142,
  "n_unmatched_treated": 8,
  "pairs": [{"treated_id": 5, "control_id": 23, "ps_diff": 0.003}],
  "balance_before": [{"covariate": "age", "mean_treated": 56.1, "mean_control": 54.2, "smd": 0.19}],
  "balance_after":  [{"covariate": "age", "mean_treated": 55.8, "mean_control": 55.4, "smd": 0.04}],
  "matched_data": {"id": [5, 23, ...], "treatment": [1, 0, ...]}
}
```

---

### `iptw_weights`

```python
iptw_weights(df, treatment_col, covariates, clip_quantiles=(0.01, 0.99),
             random_state=42, categorical_cols=None, id_col=None)
```

```json
"result": {
  "rows": [
    {"id": 0, "propensity_score": 0.43, "weight": 1.12}
  ],
  "balance_before":         [{"covariate": "age", "smd": 0.19}],
  "balance_after_weighted": [{"covariate": "age", "smd_weighted": 0.02}]
}
```

---

## Integration notes

**Backend**
- Validate user-selected columns before calling a function.
- Store the full returned envelope for audit/reproducibility.
- Surface `warnings` to logs or downstream consumers.
- Use `id_col` or rely on `df.index` to JOIN row-level results back to DB records.

**Frontend**
- Do not expect images — curves are coordinate arrays, draw them yourself.
- Show `warnings` near the result panel.
- Use `status == "warning"` for non-blocking data-quality indicators.

## Tests

```bash
git clone https://github.com/bxjxis/medstats.git
cd medstats
pip install -e ".[dev]"
pytest -q
```

Expected: **38 passed**
