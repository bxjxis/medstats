"""Survival analysis: Cox PH model and Kaplan-Meier curves."""
from __future__ import annotations

import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

from ._typing import JsonDict
from ._utils import (
    build_envelope, drop_na_rows, encode_categoricals, require_binary, require_columns, to_json_safe,
)


def cox_regression(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariates: list[str],
    categorical_cols: list[str] | None = None,
) -> JsonDict:
    """Fit a Cox proportional-hazards model.

    Returns per-term HR / 95% CI / p-value and overall C-index.

    Args:
        categorical_cols: subset of covariates to dummy-encode (drop_first=True).
            String/category columns not listed here will raise ValueError.
    """
    require_columns(df, duration_col, event_col, *covariates)
    require_binary(df, event_col)

    df, covariates = encode_categoricals(df, covariates, categorical_cols)

    cols = [duration_col, event_col, *covariates]
    df_clean, dropped = drop_na_rows(df, cols)
    n_input = len(df)
    n_used = len(df_clean)
    warnings: list[str] = []

    if n_used < 20:
        warnings.append(f"Only {n_used} complete rows; results may be unstable.")
    if df_clean[event_col].sum() < 5:
        warnings.append("Fewer than 5 events; Cox model may not converge reliably.")

    cph = CoxPHFitter()
    cph.fit(df_clean[cols], duration_col=duration_col, event_col=event_col)

    summary = cph.summary
    terms = []
    for name, row in summary.iterrows():
        terms.append({
            "name": str(name),
            "coef": row["coef"],
            "hr": row["exp(coef)"],
            "hr_ci_low": row["exp(coef) lower 95%"],
            "hr_ci_high": row["exp(coef) upper 95%"],
            "p": row["p"],
        })

    result = {
        "c_index": cph.concordance_index_,
        "n_events": int(df_clean[event_col].sum()),
        "terms": terms,
    }

    return to_json_safe(build_envelope(
        method="cox_regression",
        params={"duration_col": duration_col, "event_col": event_col,
                "covariates": covariates, "categorical_cols": categorical_cols},
        n_input=n_input, n_used=n_used, dropped=dropped,
        warnings=warnings, result=result,
    ))


def kaplan_meier(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    group_col: str | None = None,
) -> JsonDict:
    """Kaplan-Meier survival curves with optional group comparison.

    Single group: returns one curve.
    Two groups: log-rank test.
    Three+ groups: multivariate log-rank overall p only.
    """
    cols = [duration_col, event_col] + ([group_col] if group_col else [])
    require_columns(df, *cols)
    require_binary(df, event_col)

    df_clean, dropped = drop_na_rows(df, cols)
    n_input = len(df)
    n_used = len(df_clean)
    warnings: list[str] = []

    def _km_group(sub: pd.DataFrame, name: str) -> dict:
        kmf = KaplanMeierFitter()
        kmf.fit(sub[duration_col], event_observed=sub[event_col])
        sf = kmf.survival_function_
        return {
            "name": str(name),
            "times": sf.index.tolist(),
            "survival": sf["KM_estimate"].tolist(),
            "median_survival": kmf.median_survival_time_,
        }

    if group_col is None:
        groups_data = [_km_group(df_clean, "overall")]
        logrank_p = None
    else:
        group_vals = sorted(df_clean[group_col].unique())
        groups_data = [_km_group(df_clean[df_clean[group_col] == g], g) for g in group_vals]

        if len(group_vals) < 2:
            warnings.append("Only one group found; log-rank test skipped.")
            logrank_p = None
        elif len(group_vals) == 2:
            a, b = group_vals
            res = logrank_test(
                df_clean.loc[df_clean[group_col] == a, duration_col],
                df_clean.loc[df_clean[group_col] == b, duration_col],
                event_observed_A=df_clean.loc[df_clean[group_col] == a, event_col],
                event_observed_B=df_clean.loc[df_clean[group_col] == b, event_col],
            )
            logrank_p = res.p_value
        else:
            res = multivariate_logrank_test(
                df_clean[duration_col], df_clean[group_col], df_clean[event_col]
            )
            logrank_p = res.p_value

    result: dict = {"groups": groups_data}
    if logrank_p is not None:
        result["overall_logrank_p"] = logrank_p

    return to_json_safe(build_envelope(
        method="kaplan_meier",
        params={"duration_col": duration_col, "event_col": event_col, "group_col": group_col},
        n_input=n_input, n_used=n_used, dropped=dropped,
        warnings=warnings, result=result,
    ))
