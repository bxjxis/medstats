# medstats

Medical statistics module for liver disease research. A backend analytics component — import it directly into your Python backend. Not a standalone service.

## Install

```bash
pip install -r requirements.txt
```

## Quick start

```python
import pandas as pd
from medstats import cox_regression, kaplan_meier, logistic_regression
from medstats import roc_analysis, calibration_analysis, decision_curve_analysis
from medstats import multiple_imputation, propensity_score_matching, iptw_weights

df = pd.read_csv("sample_data.csv")
result = cox_regression(df, duration_col="time", event_col="event", covariates=["age", "albumin"])
import json
print(json.dumps(result, indent=2))  # always JSON-safe
```

## Common output envelope

Every function returns:

```json
{
  "method": "cox_regression",
  "status": "ok",
  "params": {"duration_col": "time", "event_col": "event", "covariates": ["age"]},
  "n_input": 300,
  "n_used": 285,
  "dropped_rows": 15,
  "warnings": [],
  "result": { ... }
}
```

`status` is `"warning"` when `warnings` is non-empty. All values are JSON-native (no numpy types). `NaN`/`Inf` become `null`.

---

## API reference

### `cox_regression`

```python
cox_regression(
    df, duration_col, event_col, covariates,
    categorical_cols=None   # list of covariates to dummy-encode
)
```

`result`:
```json
{
  "c_index": 0.74,
  "n_events": 88,
  "terms": [
    {"name": "age", "coef": 0.031, "hr": 1.031,
     "hr_ci_low": 1.01, "hr_ci_high": 1.05, "p": 0.002}
  ]
}
```

---

### `kaplan_meier`

```python
kaplan_meier(df, duration_col, event_col, group_col=None)
```

`result`:
```json
{
  "groups": [
    {"name": "early", "times": [0, 3, 6], "survival": [1.0, 0.97, 0.94],
     "median_survival": 42.0}
  ],
  "overall_logrank_p": 0.003
}
```

`overall_logrank_p` is absent when `group_col=None`. With 2 groups: two-sample log-rank. With 3+ groups: multivariate log-rank overall p only.

Chart: `times` (x-axis) vs `survival` (y-axis) per group.

---

### `logistic_regression`

```python
logistic_regression(
    df, outcome_col, covariates,
    add_intercept=True,
    categorical_cols=None,
    id_col=None    # column to use as row identity; df.index used if None
)
```

`result`:
```json
{
  "converged": true,
  "n_events": 72,
  "pseudo_r2": 0.18,
  "terms": [
    {"name": "age", "coef": 0.04, "or": 1.04,
     "or_ci_low": 1.01, "or_ci_high": 1.07, "p": 0.008}
  ],
  "predicted_probabilities": [
    {"id": 0, "predicted_probability": 0.23},
    {"id": 1, "predicted_probability": 0.67}
  ]
}
```

`id` is the value from `id_col` (or `df.index`) — use it to JOIN predictions back to database rows.

---

### `roc_analysis`

```python
roc_analysis(
    df, label_col, score_col,
    compare_score_col=None   # optional second model for DeLong comparison
)
```

`score_col` must be in `[0, 1]` (probabilities). Validates this.

`result`:
```json
{
  "auc": 0.81,
  "auc_ci_low": 0.76,
  "auc_ci_high": 0.86,
  "fpr": [0.0, 0.05, ...],
  "tpr": [0.0, 0.12, ...],
  "thresholds": [1.0, 0.93, ...],
  "compare": {
    "score_col": "model_b",
    "auc": 0.75, "auc_ci_low": 0.69, "auc_ci_high": 0.81,
    "fpr": [...], "tpr": [...], "thresholds": [...],
    "delong_z": 1.82, "delong_p": 0.069
  }
}
```

Chart: `fpr` (x) vs `tpr` (y). DeLong variance uses `ddof=1` (Sun & Xu 2014).

---

### `calibration_analysis`

```python
calibration_analysis(
    df, label_col, score_col,
    n_bins=10,
    hosmer_lemeshow=True
)
```

`score_col` must be in `[0, 1]`.

`result`:
```json
{
  "mean_predicted": [0.05, 0.15, 0.28, ...],
  "fraction_observed": [0.04, 0.18, 0.31, ...],
  "brier_score": 0.19,
  "n_bins": 10,
  "hosmer_lemeshow": {"statistic": 8.4, "p": 0.39, "df": 8}
}
```

Chart: `mean_predicted` (x) vs `fraction_observed` (y). Perfect calibration = diagonal line.

---

### `decision_curve_analysis`

```python
decision_curve_analysis(
    df, label_col, score_cols,     # list of score columns
    threshold_min=0.01, threshold_max=0.99, n_thresholds=99
)
```

All `score_cols` must be in `[0, 1]`.

`result`:
```json
{
  "thresholds": [0.01, 0.02, ...],
  "treat_all": [0.24, 0.23, ...],
  "treat_none": [0.0, 0.0, ...],
  "models": [
    {"name": "score_a", "net_benefit": [0.22, 0.21, ...]},
    {"name": "score_b", "net_benefit": [0.19, 0.18, ...]}
  ],
  "prevalence": 0.24
}
```

Chart: `thresholds` (x) vs net benefit (y) for each model + baselines.

---

### `multiple_imputation`

```python
multiple_imputation(
    df, target_cols,
    m=1,              # number of imputed datasets (Rubin pooling NOT included)
    random_state=42,
    id_col=None       # row identity column; df.index used if None
)
```

**Note:** Uses sklearn `IterativeImputer` (MICE-style). Returns `m` independently-seeded datasets. Rubin pooling for inference must be done by the caller.

`result`:
```json
{
  "m": 1,
  "missing_rates": {"albumin": 0.05, "bilirubin": 0.05},
  "imputed_cols": ["albumin", "bilirubin"],
  "unchanged_cols": [],
  "datasets": [
    [
      {"id": 0, "albumin": 34.2, "bilirubin": 1.1},
      {"id": 1, "albumin": 38.7, "bilirubin": 0.9}
    ]
  ]
}
```

`id` enables JOIN to database rows.

---

### `propensity_score_matching`

```python
propensity_score_matching(
    df, treatment_col, covariates,
    caliper=0.2,      # fraction of PS std dev; None = no caliper
    replace=False,
    random_state=42,
    categorical_cols=None
)
```

`result`:
```json
{
  "n_pairs": 142,
  "n_unmatched_treated": 8,
  "pairs": [
    {"treated_id": 5, "control_id": 23, "ps_diff": 0.003}
  ],
  "balance_before": [
    {"covariate": "age", "mean_treated": 56.1, "mean_control": 54.2, "smd": 0.19}
  ],
  "balance_after": [
    {"covariate": "age", "mean_treated": 55.8, "mean_control": 55.4, "smd": 0.04}
  ],
  "matched_data": {"id": [5, 23, ...], "treatment": [1, 0, ...], "age": [...]}
}
```

`treated_id` / `control_id` are original `df.index` values for DB JOIN.

---

### `iptw_weights`

```python
iptw_weights(
    df, treatment_col, covariates,
    clip_quantiles=(0.01, 0.99),   # None = no clipping
    random_state=42,
    categorical_cols=None,
    id_col=None
)
```

`result`:
```json
{
  "rows": [
    {"id": 0, "propensity_score": 0.43, "weight": 1.12},
    {"id": 1, "propensity_score": 0.61, "weight": 0.82}
  ],
  "balance_before": [
    {"covariate": "age", "mean_treated": 56.1, "mean_control": 54.2, "smd": 0.19}
  ],
  "balance_after_weighted": [
    {"covariate": "age", "smd_weighted": 0.02}
  ]
}
```

Stabilized IPTW weights. `id` enables JOIN to database rows.

---

## Categorical covariates

String/category columns will raise `ValueError` unless listed in `categorical_cols`. They are then dummy-encoded with `pd.get_dummies(drop_first=True)`. Generated dummy column names appear in model term outputs.

```python
cox_regression(df, "time", "event", ["age", "stage"], categorical_cols=["stage"])
# → terms include {"name": "stage_early", ...}
```

## Testing

```bash
pytest tests/ -q
```

38 tests. All functions verified `json.dumps`-safe.
