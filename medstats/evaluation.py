"""Model evaluation: ROC/AUC+DeLong, calibration curve, decision curve analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from ._typing import JsonDict
from ._utils import (
    build_envelope, drop_na_rows, require_binary, require_columns, require_probability, to_json_safe,
)


# ---------------------------------------------------------------------------
# DeLong AUC confidence interval and two-AUC comparison
# Reference: Sun & Xu (2014), "Fast Implementation of DeLong's Algorithm for
#            Comparing the Areas Under Correlated Receiver Operating
#            Characteristic Curves", IEEE Signal Processing Letters 21(11).
# ---------------------------------------------------------------------------

def _delong_auc_ci(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 0.05):
    """Return (auc, ci_low, ci_high) using the DeLong variance estimator."""
    from scipy import stats

    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n1, n0 = len(pos), len(neg)

    if n1 == 0 or n0 == 0:
        raise ValueError("Both positive and negative samples are required for DeLong CI.")

    # Placement values (psi matrices)
    # V10[i] = (1/n0) * sum_j 1{neg_j < pos_i} + 0.5 * 1{neg_j == pos_i}
    # V01[j] = (1/n1) * sum_i 1{pos_i > neg_j} + 0.5 * 1{pos_i == neg_j}
    V10 = np.mean(
        (pos[:, None] > neg[None, :]).astype(float)
        + 0.5 * (pos[:, None] == neg[None, :]).astype(float),
        axis=1,
    )
    V01 = np.mean(
        (pos[None, :] > neg[:, None]).astype(float)
        + 0.5 * (pos[None, :] == neg[:, None]).astype(float),
        axis=1,
    )

    auc = float(np.mean(V10))
    # Sun & Xu (2014): use sample variance (ddof=1) for the structural components
    var = (np.var(V10, ddof=1) / n1) + (np.var(V01, ddof=1) / n0)
    se = float(np.sqrt(var))

    z = stats.norm.ppf(1 - alpha / 2)
    return auc, max(0.0, auc - z * se), min(1.0, auc + z * se)


def _delong_compare(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray):
    """Two-sample DeLong test for H0: AUC_A == AUC_B. Returns z and p (two-sided)."""
    from scipy import stats

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    pos_a, neg_a = score_a[pos_mask], score_a[neg_mask]
    pos_b, neg_b = score_b[pos_mask], score_b[neg_mask]
    n1, n0 = pos_mask.sum(), neg_mask.sum()

    def _psi(pos, neg):
        return (
            (pos[:, None] > neg[None, :]).astype(float)
            + 0.5 * (pos[:, None] == neg[None, :]).astype(float)
        )

    Pa = _psi(pos_a, neg_a)  # shape (n1, n0)
    Pb = _psi(pos_b, neg_b)

    # Structural components
    Va10 = Pa.mean(axis=1)  # (n1,)
    Va01 = Pa.mean(axis=0)  # (n0,)
    Vb10 = Pb.mean(axis=1)
    Vb01 = Pb.mean(axis=0)

    auc_a = float(Va10.mean())
    auc_b = float(Vb10.mean())

    # Covariance matrix S (2x2) via joint structural components; ddof=1 throughout
    S11 = (np.var(Va10, ddof=1) / n1) + (np.var(Va01, ddof=1) / n0)
    S22 = (np.var(Vb10, ddof=1) / n1) + (np.var(Vb01, ddof=1) / n0)
    S12 = (np.cov(Va10, Vb10)[0, 1] / n1) + (np.cov(Va01, Vb01)[0, 1] / n0)

    var_diff = S11 + S22 - 2 * S12
    if var_diff <= 0:
        return 0.0, 1.0  # identical scores

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return float(z), p


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def roc_analysis(
    df: pd.DataFrame,
    label_col: str,
    score_col: str,
    compare_score_col: str | None = None,
) -> JsonDict:
    """ROC curve, AUC with DeLong CI, optional two-model comparison.

    Args:
        df: input DataFrame.
        label_col: binary (0/1) ground-truth column.
        score_col: predicted probability / score for primary model.
        compare_score_col: optional second model score for DeLong comparison.
    """
    cols = [label_col, score_col] + ([compare_score_col] if compare_score_col else [])
    require_columns(df, *cols)
    require_binary(df, label_col)
    require_probability(df, score_col)
    if compare_score_col:
        require_probability(df, compare_score_col)

    df_clean, dropped = drop_na_rows(df, cols)
    n_input, n_used = len(df), len(df_clean)
    warnings: list[str] = []

    y = df_clean[label_col].astype(int).values
    s = df_clean[score_col].astype(float).values

    fpr, tpr, thresholds = roc_curve(y, s)
    auc, ci_low, ci_high = _delong_auc_ci(y, s)

    result: dict = {
        "auc": auc,
        "auc_ci_low": ci_low,
        "auc_ci_high": ci_high,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }

    if compare_score_col:
        s2 = df_clean[compare_score_col].astype(float).values
        fpr2, tpr2, thr2 = roc_curve(y, s2)
        auc2, ci_low2, ci_high2 = _delong_auc_ci(y, s2)
        z, p = _delong_compare(y, s, s2)
        result["compare"] = {
            "score_col": compare_score_col,
            "auc": auc2,
            "auc_ci_low": ci_low2,
            "auc_ci_high": ci_high2,
            "fpr": fpr2.tolist(),
            "tpr": tpr2.tolist(),
            "thresholds": thr2.tolist(),
            "delong_z": z,
            "delong_p": p,
        }

    return to_json_safe(build_envelope(
        method="roc_analysis",
        params={"label_col": label_col, "score_col": score_col, "compare_score_col": compare_score_col},
        n_input=n_input, n_used=n_used, dropped=dropped,
        warnings=warnings, result=result,
    ))


def calibration_analysis(
    df: pd.DataFrame,
    label_col: str,
    score_col: str,
    n_bins: int = 10,
    hosmer_lemeshow: bool = True,
) -> JsonDict:
    """Calibration curve (reliability diagram) + Brier score.

    Optionally computes Hosmer-Lemeshow goodness-of-fit test.
    """
    require_columns(df, label_col, score_col)
    require_binary(df, label_col)
    require_probability(df, score_col)

    df_clean, dropped = drop_na_rows(df, [label_col, score_col])
    n_input, n_used = len(df), len(df_clean)
    warnings: list[str] = []

    y = df_clean[label_col].astype(int).values
    s = df_clean[score_col].astype(float).values

    fraction_pos, mean_pred = calibration_curve(y, s, n_bins=n_bins, strategy="uniform")
    brier = brier_score_loss(y, s)

    result: dict = {
        "mean_predicted": mean_pred.tolist(),
        "fraction_observed": fraction_pos.tolist(),
        "brier_score": float(brier),
        "n_bins": n_bins,
    }

    if hosmer_lemeshow:
        # Hosmer-Lemeshow test: chi2 across quantile-based deciles
        from scipy import stats as sp_stats

        n_hl_bins = min(10, n_used // 5)
        if n_hl_bins < 3:
            warnings.append("Too few samples for Hosmer-Lemeshow test (need ≥15 per bin); skipped.")
        else:
            order = np.argsort(s)
            s_sorted, y_sorted = s[order], y[order]
            bins = np.array_split(np.arange(n_used), n_hl_bins)
            hl_stat = 0.0
            for b in bins:
                obs = y_sorted[b].sum()
                exp = s_sorted[b].sum()
                n_b = len(b)
                if exp > 0 and (n_b - exp) > 0:
                    hl_stat += (obs - exp) ** 2 / (exp * (1 - exp / n_b))
            hl_p = float(sp_stats.chi2.sf(hl_stat, df=n_hl_bins - 2))
            result["hosmer_lemeshow"] = {"statistic": float(hl_stat), "p": hl_p, "df": n_hl_bins - 2}

    return to_json_safe(build_envelope(
        method="calibration_analysis",
        params={"label_col": label_col, "score_col": score_col, "n_bins": n_bins},
        n_input=n_input, n_used=n_used, dropped=dropped,
        warnings=warnings, result=result,
    ))


def decision_curve_analysis(
    df: pd.DataFrame,
    label_col: str,
    score_cols: list[str],
    threshold_min: float = 0.01,
    threshold_max: float = 0.99,
    n_thresholds: int = 99,
) -> JsonDict:
    """Decision curve analysis: net benefit vs probability threshold.

    Returns net benefit curves for each model plus treat-all and treat-none baselines.
    Formula: NB(pt) = TP/n - FP/n * pt/(1-pt)
    """
    require_columns(df, label_col, *score_cols)
    require_binary(df, label_col)
    for sc in score_cols:
        require_probability(df, sc)

    all_cols = [label_col, *score_cols]
    df_clean, dropped = drop_na_rows(df, all_cols)
    n_input, n_used = len(df), len(df_clean)
    warnings: list[str] = []

    y = df_clean[label_col].astype(int).values
    n = len(y)
    prevalence = y.mean()

    thresholds = np.linspace(threshold_min, threshold_max, n_thresholds).tolist()

    def _net_benefit(score: np.ndarray, pts: list[float]) -> list[float | None]:
        nb = []
        for pt in pts:
            pred_pos = score >= pt
            tp = float((pred_pos & (y == 1)).sum())
            fp = float((pred_pos & (y == 0)).sum())
            nb.append(tp / n - fp / n * pt / (1 - pt))
        return nb

    # treat-all: NB = prevalence - (1-prevalence)*pt/(1-pt)
    treat_all = [prevalence - (1 - prevalence) * pt / (1 - pt) for pt in thresholds]
    # treat-none: NB = 0 always
    treat_none = [0.0] * len(thresholds)

    models = []
    for sc in score_cols:
        s = df_clean[sc].astype(float).values
        models.append({"name": sc, "net_benefit": _net_benefit(s, thresholds)})

    result = {
        "thresholds": thresholds,
        "treat_all": treat_all,
        "treat_none": treat_none,
        "models": models,
        "prevalence": float(prevalence),
    }

    return to_json_safe(build_envelope(
        method="decision_curve_analysis",
        params={"label_col": label_col, "score_cols": score_cols,
                "threshold_min": threshold_min, "threshold_max": threshold_max},
        n_input=n_input, n_used=n_used, dropped=dropped,
        warnings=warnings, result=result,
    ))
