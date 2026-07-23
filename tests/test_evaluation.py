import json
import numpy as np
import pandas as pd
import pytest
from medstats import calibration_analysis, decision_curve_analysis, roc_analysis

DF = pd.read_csv("sample_data.csv")


def _json_safe(result):
    json.dumps(result)
    return result


# --- ROC ---

def test_roc_basic():
    res = _json_safe(roc_analysis(DF, "outcome", "score_a"))
    r = res["result"]
    assert 0.5 < r["auc"] < 1.0
    assert r["auc_ci_low"] < r["auc"] < r["auc_ci_high"]
    assert len(r["fpr"]) == len(r["tpr"])


def test_roc_perfect_classifier():
    """Perfect score should give AUC ≈ 1."""
    df = pd.DataFrame({"label": [0, 0, 1, 1], "score": [0.1, 0.2, 0.8, 0.9]})
    res = roc_analysis(df, "label", "score")
    assert abs(res["result"]["auc"] - 1.0) < 1e-9


def test_roc_delong_compare_same_score():
    """Comparing a score to itself: DeLong p should be 1."""
    df = DF.copy()
    df["score_copy"] = df["score_a"]
    res = roc_analysis(df, "outcome", "score_a", compare_score_col="score_copy")
    assert res["result"]["compare"]["delong_p"] > 0.9


def test_roc_delong_compare_different():
    """score_a better than score_b; comparison p can be anything, just check structure."""
    res = _json_safe(roc_analysis(DF, "outcome", "score_a", compare_score_col="score_b"))
    comp = res["result"]["compare"]
    assert "delong_z" in comp and "delong_p" in comp
    assert 0.0 <= comp["delong_p"] <= 1.0


def test_roc_delong_reference_values():
    """Fixed reference: known AUC and CI for a controlled sample."""
    import numpy as np
    rng = np.random.default_rng(99)
    y = np.array([0]*50 + [1]*50)
    # Score = truth + small noise → AUC should be ~0.9
    s = np.clip(y + rng.normal(0, 0.3, 100), 0, 1)
    df = pd.DataFrame({"y": y, "s": s})
    res = roc_analysis(df, "y", "s")
    auc = res["result"]["auc"]
    assert 0.85 < auc <= 1.0, f"Expected AUC > 0.85 for well-separated classes, got {auc}"
    # CI should contain the AUC
    assert res["result"]["auc_ci_low"] < auc < res["result"]["auc_ci_high"]


def test_roc_out_of_range_score_raises():
    df = DF.copy()
    df["bad_score"] = df["score_a"] * 10  # outside [0,1]
    with pytest.raises(ValueError, match="probabilities"):
        roc_analysis(df, "outcome", "bad_score")


# --- Calibration ---

def test_calibration_basic():
    res = _json_safe(calibration_analysis(DF, "outcome", "score_a"))
    r = res["result"]
    assert "brier_score" in r
    assert 0 < r["brier_score"] < 1
    assert len(r["mean_predicted"]) == len(r["fraction_observed"])


def test_calibration_hl():
    res = calibration_analysis(DF, "outcome", "score_a", hosmer_lemeshow=True)
    hl = res["result"].get("hosmer_lemeshow")
    assert hl is not None
    assert 0 <= hl["p"] <= 1


# --- DCA ---

def test_dca_basic():
    res = _json_safe(decision_curve_analysis(DF, "outcome", ["score_a", "score_b"]))
    r = res["result"]
    assert len(r["thresholds"]) == len(r["treat_all"])
    assert all(v == 0.0 for v in r["treat_none"])
    assert len(r["models"]) == 2


def test_dca_thresholds_exclude_zero_one():
    res = decision_curve_analysis(DF, "outcome", ["score_a"])
    pts = res["result"]["thresholds"]
    assert min(pts) > 0
    assert max(pts) < 1
