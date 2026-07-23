"""Logistic regression with OR / 95% CI / p-value output."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ._typing import JsonDict
from ._utils import (
    build_envelope, drop_na_rows, encode_categoricals, require_binary, require_columns, to_json_safe,
)


def logistic_regression(
    df: pd.DataFrame,
    outcome_col: str,
    covariates: list[str],
    add_intercept: bool = True,
    categorical_cols: list[str] | None = None,
    id_col: str | None = None,
) -> JsonDict:
    """Fit a binary logistic regression model via statsmodels.

    Returns OR, 95% CI, p-value per term, plus per-row predicted probabilities
    with their original row identity.

    Args:
        categorical_cols: subset of covariates to dummy-encode (drop_first=True).
        id_col: column to use as row identifier in predicted_probabilities output.
            If None, df.index values are used.
    """
    require_columns(df, outcome_col, *covariates)
    if id_col:
        require_columns(df, id_col)
    require_binary(df, outcome_col)

    df, covariates = encode_categoricals(df, covariates, categorical_cols)

    cols = [outcome_col, *covariates] + ([id_col] if id_col else [])
    df_clean, dropped = drop_na_rows(df, cols)
    n_input = len(df)
    n_used = len(df_clean)
    warnings: list[str] = []

    # EPV: events per variable = min(events, non-events) / n_covariates
    n_events = int(df_clean[outcome_col].sum())
    n_nonevents = n_used - n_events
    epv = min(n_events, n_nonevents) / max(len(covariates), 1)
    if epv < 10:
        warnings.append(
            f"EPV={epv:.1f} (min(events,non-events)/covariates={min(n_events,n_nonevents)}/{len(covariates)}); "
            "estimates may be unstable. Rule of thumb: EPV ≥ 10."
        )

    X = df_clean[covariates].astype(float)
    if add_intercept:
        X = sm.add_constant(X, has_constant="add")
    y = df_clean[outcome_col].astype(float)

    try:
        model = sm.Logit(y, X).fit(disp=False, maxiter=200)
        converged = bool(model.mle_retvals.get("converged", True))
    except Exception as exc:
        raise ValueError(f"Logistic regression failed to fit: {exc}") from exc

    if not converged:
        warnings.append("Model did not converge; estimates may be unreliable.")

    conf = model.conf_int()
    terms = []
    for name in model.params.index:
        coef = float(model.params[name])
        terms.append({
            "name": str(name),
            "coef": coef,
            "or": float(np.exp(coef)),
            "or_ci_low": float(np.exp(conf.loc[name, 0])),
            "or_ci_high": float(np.exp(conf.loc[name, 1])),
            "p": float(model.pvalues[name]),
        })

    # Row-level predictions: include original identifier so backend can JOIN to DB
    probs = model.predict(X).tolist()
    ids = df_clean[id_col].tolist() if id_col else df_clean.index.tolist()
    predicted = [{"id": i, "predicted_probability": p} for i, p in zip(ids, probs)]

    result = {
        "converged": converged,
        "n_events": n_events,
        "pseudo_r2": float(model.prsquared),
        "terms": terms,
        "predicted_probabilities": predicted,
    }

    return to_json_safe(build_envelope(
        method="logistic_regression",
        params={"outcome_col": outcome_col, "covariates": covariates,
                "add_intercept": add_intercept, "categorical_cols": categorical_cols,
                "id_col": id_col},
        n_input=n_input, n_used=n_used, dropped=dropped,
        warnings=warnings, result=result,
    ))
