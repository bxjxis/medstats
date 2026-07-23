"""Internal utilities: JSON-safe conversion, input validation, column helpers."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to JSON-serialisable Python natives.

    NaN and Inf become None so downstream json.dumps never raises.
    """
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return [to_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (float,)):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, pd.Series):
        return [to_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def require_columns(df: pd.DataFrame, *cols: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def require_binary(df: pd.DataFrame, col: str) -> None:
    vals = df[col].dropna().unique()
    if not set(vals).issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"Column '{col}' must be binary (0/1), got values: {list(vals)[:10]}")


def require_probability(df: pd.DataFrame, col: str) -> None:
    """Raise if column contains values outside [0, 1] (ignoring NaN)."""
    s = df[col].dropna()
    if (s < 0).any() or (s > 1).any():
        raise ValueError(
            f"Column '{col}' must contain probabilities in [0, 1]; "
            f"got min={s.min():.4g}, max={s.max():.4g}."
        )


def encode_categoricals(
    df: pd.DataFrame,
    covariates: list[str],
    categorical_cols: list[str] | None,
) -> tuple[pd.DataFrame, list[str]]:
    """Return (df_encoded, final_covariate_list).

    If categorical_cols is provided, dummy-encode those columns (drop_first=True)
    and replace them in the covariate list. Non-categorical covariates are cast to float.
    Raises ValueError for string columns not listed in categorical_cols.
    """
    if not categorical_cols:
        # No encoding requested — check for accidental string columns
        for c in covariates:
            if df[c].dtype == object or str(df[c].dtype) == "category":
                raise ValueError(
                    f"Column '{c}' is string/categorical. Pass it via categorical_cols= for dummy encoding."
                )
        return df, covariates

    cat_set = set(categorical_cols)
    non_cat = [c for c in covariates if c not in cat_set]
    cat_cols = [c for c in covariates if c in cat_set]

    dummies = pd.get_dummies(df[cat_cols], drop_first=True, dtype=float)
    df_out = df.copy()
    df_out = df_out.drop(columns=cat_cols)
    for col in dummies.columns:
        df_out[col] = dummies[col].values

    new_covariates = non_cat + list(dummies.columns)
    return df_out, new_covariates


def drop_na_rows(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.dropna(subset=cols).copy()
    return df, before - len(df)


def build_envelope(method: str, params: dict, n_input: int, n_used: int,
                   dropped: int, warnings: list[str], result: dict) -> dict:
    return {
        "method": method,
        "status": "warning" if warnings else "ok",
        "params": params,
        "n_input": n_input,
        "n_used": n_used,
        "dropped_rows": dropped,
        "warnings": warnings,
        "result": result,
    }
