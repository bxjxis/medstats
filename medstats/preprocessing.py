"""Data preprocessing: MICE imputation, propensity score matching, IPTW."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ._typing import JsonDict
from ._utils import (
    build_envelope, encode_categoricals, require_binary, require_columns, to_json_safe,
)


def _smd(col_a: pd.Series, col_b: pd.Series) -> float:
    """Standardized mean difference between two groups for a continuous column."""
    mean_diff = col_a.mean() - col_b.mean()
    pooled_std = np.sqrt((col_a.std() ** 2 + col_b.std() ** 2) / 2)
    if pooled_std == 0:
        return 0.0
    return float(mean_diff / pooled_std)


def _balance_table(df: pd.DataFrame, covariates: list[str], treatment_col: str) -> list[dict]:
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    return [
        {
            "covariate": cov,
            "mean_treated": float(treated[cov].mean()),
            "mean_control": float(control[cov].mean()),
            "smd": _smd(treated[cov], control[cov]),
        }
        for cov in covariates
    ]


def multiple_imputation(
    df: pd.DataFrame,
    target_cols: list[str],
    m: int = 1,
    random_state: int = 42,
    id_col: str | None = None,
) -> JsonDict:
    """Iterative (MICE-style) imputation for specified columns.

    Note: this performs iterative imputation (sklearn IterativeImputer).
    When m=1, returns one completed dataset. Rubin pooling is NOT performed.
    Set m>1 to get m independently-seeded imputed datasets for downstream pooling.

    Args:
        id_col: column to include as row identifier in each dataset row.
            If None, df.index values are used so rows can be JOINed back to the DB.
    """
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    require_columns(df, *target_cols)
    if id_col:
        require_columns(df, id_col)

    n_input = len(df)
    warnings: list[str] = []

    missing_rates = {col: float(df[col].isna().mean()) for col in target_cols}
    cols_with_missing = [c for c, r in missing_rates.items() if r > 0]
    cols_no_missing = [c for c, r in missing_rates.items() if r == 0]

    if not cols_with_missing:
        warnings.append("No missing values found in target_cols; returning original data.")

    numeric_df = df.select_dtypes(include=[np.number])
    non_numeric = [c for c in target_cols if c not in numeric_df.columns]
    if non_numeric:
        raise ValueError(f"Non-numeric columns cannot be imputed: {non_numeric}")

    ids = df[id_col].tolist() if id_col else df.index.tolist()

    datasets = []
    for i in range(m):
        imp = IterativeImputer(random_state=random_state + i, max_iter=10)
        imputed = imp.fit_transform(numeric_df)
        imputed_df = pd.DataFrame(imputed, columns=numeric_df.columns, index=df.index)
        # Include row identity so backend can JOIN results to database rows
        rows = [
            {"id": row_id, **{col: imputed_df.at[idx, col] for col in target_cols}}
            for row_id, idx in zip(ids, df.index)
        ]
        datasets.append(rows)

    result: dict = {
        "m": m,
        "missing_rates": missing_rates,
        "imputed_cols": cols_with_missing,
        "unchanged_cols": cols_no_missing,
        "datasets": datasets,
    }

    return to_json_safe(build_envelope(
        method="multiple_imputation",
        params={"target_cols": target_cols, "m": m, "random_state": random_state, "id_col": id_col},
        n_input=n_input, n_used=n_input, dropped=0,
        warnings=warnings, result=result,
    ))


def propensity_score_matching(
    df: pd.DataFrame,
    treatment_col: str,
    covariates: list[str],
    caliper: float | None = 0.2,
    replace: bool = False,
    random_state: int = 42,
    categorical_cols: list[str] | None = None,
) -> JsonDict:
    """1:1 nearest-neighbour propensity score matching.

    Returns matched pair indices (with original row identity), matched dataset,
    and SMD before/after.
    caliper is expressed as a fraction of the PS standard deviation (default 0.2).

    Args:
        categorical_cols: covariates to dummy-encode before PS estimation.
    """
    require_columns(df, treatment_col, *covariates)
    require_binary(df, treatment_col)

    df, covariates = encode_categoricals(df, covariates, categorical_cols)

    df_clean = df[[treatment_col, *covariates]].dropna().copy()
    dropped = len(df) - len(df_clean)
    n_input = len(df)
    warnings: list[str] = []

    # Check group presence BEFORE fitting the model
    treated_idx = df_clean.index[df_clean[treatment_col] == 1].tolist()
    control_idx = df_clean.index[df_clean[treatment_col] == 0].tolist()
    if not treated_idx or not control_idx:
        raise ValueError(
            f"Treatment column '{treatment_col}' has only one group after dropping NAs; matching impossible."
        )

    X = df_clean[covariates].astype(float).values
    t = df_clean[treatment_col].astype(int).values

    lr = LogisticRegression(max_iter=500, random_state=random_state)
    lr.fit(X, t)
    ps = lr.predict_proba(X)[:, 1]
    df_clean["_ps"] = ps

    caliper_val = caliper * float(np.std(ps)) if caliper is not None else None

    rng = np.random.default_rng(random_state)
    rng.shuffle(treated_idx)  # type: ignore[arg-type]

    pairs = []
    used_controls: set = set()

    for ti in treated_idx:
        ps_t = df_clean.at[ti, "_ps"]
        available = [ci for ci in control_idx if (replace or ci not in used_controls)]
        if not available:
            break
        ps_c = np.array([df_clean.at[ci, "_ps"] for ci in available])
        diffs = np.abs(ps_c - ps_t)
        best_i = int(np.argmin(diffs))
        if caliper_val is not None and diffs[best_i] > caliper_val:
            continue
        ci = available[best_i]
        # Store original index values (not positional) for DB JOIN
        pairs.append({"treated_id": ti, "control_id": ci, "ps_diff": float(diffs[best_i])})
        used_controls.add(ci)

    if not pairs:
        raise ValueError("No matches found; try removing the caliper or checking covariate overlap.")

    matched_idx = [p["treated_id"] for p in pairs] + [p["control_id"] for p in pairs]
    df_matched = df_clean.loc[matched_idx].drop(columns=["_ps"])

    balance_before = _balance_table(df_clean.drop(columns=["_ps"]), covariates, treatment_col)
    balance_after = _balance_table(df_matched, covariates, treatment_col)

    n_unmatched = len(treated_idx) - len(pairs)
    if n_unmatched > 0:
        warnings.append(
            f"{n_unmatched} treated units could not be matched (caliper too tight or no control overlap)."
        )

    result = {
        "n_pairs": len(pairs),
        "n_unmatched_treated": n_unmatched,
        "pairs": pairs,
        "balance_before": balance_before,
        "balance_after": balance_after,
        "matched_data": df_matched.reset_index(names="id").to_dict(orient="list"),
    }

    return to_json_safe(build_envelope(
        method="propensity_score_matching",
        params={"treatment_col": treatment_col, "covariates": covariates,
                "caliper": caliper, "replace": replace, "categorical_cols": categorical_cols},
        n_input=n_input, n_used=len(df_clean), dropped=dropped,
        warnings=warnings, result=result,
    ))


def iptw_weights(
    df: pd.DataFrame,
    treatment_col: str,
    covariates: list[str],
    clip_quantiles: tuple[float, float] | None = (0.01, 0.99),
    random_state: int = 42,
    categorical_cols: list[str] | None = None,
    id_col: str | None = None,
) -> JsonDict:
    """Inverse probability of treatment weighting (stabilized IPTW).

    Returns per-row propensity score and stabilized weight with original row identity,
    plus SMD before/after weighting.

    Args:
        clip_quantiles: trim extreme weights at these quantile bounds (None = no clipping).
        categorical_cols: covariates to dummy-encode before PS estimation.
        id_col: column to use as row identifier. If None, df.index values are used.
    """
    require_columns(df, treatment_col, *covariates)
    if id_col:
        require_columns(df, id_col)
    require_binary(df, treatment_col)

    df, covariates = encode_categoricals(df, covariates, categorical_cols)

    keep_cols = [treatment_col, *covariates] + ([id_col] if id_col else [])
    df_clean = df[keep_cols].dropna().copy()
    dropped = len(df) - len(df_clean)
    n_input = len(df)
    warnings: list[str] = []

    X = df_clean[covariates].astype(float).values
    t = df_clean[treatment_col].astype(int).values

    lr = LogisticRegression(max_iter=500, random_state=random_state)
    lr.fit(X, t)
    ps = lr.predict_proba(X)[:, 1]

    p_treat = t.mean()
    p_control = 1 - p_treat
    weights = np.where(t == 1, p_treat / ps, p_control / (1 - ps))

    if clip_quantiles is not None:
        lo = float(np.quantile(weights, clip_quantiles[0]))
        hi = float(np.quantile(weights, clip_quantiles[1]))
        n_clipped = int(((weights < lo) | (weights > hi)).sum())
        weights = np.clip(weights, lo, hi)
        if n_clipped > 0:
            warnings.append(f"{n_clipped} weights clipped to [{lo:.3f}, {hi:.3f}].")

    df_clean = df_clean.copy()
    df_clean["_weight"] = weights
    df_clean["_ps"] = ps

    def _weighted_smd(cov: str) -> float:
        w1 = df_clean.loc[df_clean[treatment_col] == 1, "_weight"]
        w0 = df_clean.loc[df_clean[treatment_col] == 0, "_weight"]
        x1 = df_clean.loc[df_clean[treatment_col] == 1, cov]
        x0 = df_clean.loc[df_clean[treatment_col] == 0, cov]
        wm1 = float(np.average(x1, weights=w1))
        wm0 = float(np.average(x0, weights=w0))
        wv1 = float(np.average((x1 - wm1) ** 2, weights=w1))
        wv0 = float(np.average((x0 - wm0) ** 2, weights=w0))
        pooled = np.sqrt((wv1 + wv0) / 2)
        return float((wm1 - wm0) / pooled) if pooled > 0 else 0.0

    balance_before = _balance_table(df_clean.drop(columns=["_weight", "_ps"]), covariates, treatment_col)
    balance_after_weighted = [
        {"covariate": cov, "smd_weighted": _weighted_smd(cov)} for cov in covariates
    ]

    # Row-level output with identity for DB JOIN
    ids = df_clean[id_col].tolist() if id_col else df_clean.index.tolist()
    row_outputs = [
        {"id": rid, "propensity_score": float(p), "weight": float(w)}
        for rid, p, w in zip(ids, ps, weights)
    ]

    result = {
        "rows": row_outputs,
        "balance_before": balance_before,
        "balance_after_weighted": balance_after_weighted,
    }

    return to_json_safe(build_envelope(
        method="iptw_weights",
        params={"treatment_col": treatment_col, "covariates": covariates,
                "clip_quantiles": list(clip_quantiles) if clip_quantiles else None,
                "categorical_cols": categorical_cols, "id_col": id_col},
        n_input=n_input, n_used=len(df_clean), dropped=dropped,
        warnings=warnings, result=result,
    ))
