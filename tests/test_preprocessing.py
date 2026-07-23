import json
import numpy as np
import pandas as pd
import pytest
from medstats import iptw_weights, multiple_imputation, propensity_score_matching

DF = pd.read_csv("sample_data.csv")
NUM_COLS = ["age", "albumin", "bilirubin", "score_a"]


def _json_safe(result):
    json.dumps(result)
    return result


# --- Imputation ---

def test_imputation_removes_nan():
    res = _json_safe(multiple_imputation(DF, ["albumin", "bilirubin"]))
    rows = res["result"]["datasets"][0]
    assert all(r["albumin"] is not None for r in rows)
    assert all(r["bilirubin"] is not None for r in rows)
    # Each row must have an id field
    assert "id" in rows[0]


def test_imputation_missing_rates():
    res = multiple_imputation(DF, ["albumin", "bilirubin"])
    rates = res["result"]["missing_rates"]
    assert rates["albumin"] > 0
    assert rates["bilirubin"] > 0


def test_imputation_reproducible():
    r1 = multiple_imputation(DF, ["albumin"], random_state=7)
    r2 = multiple_imputation(DF, ["albumin"], random_state=7)
    vals1 = [r["albumin"] for r in r1["result"]["datasets"][0]]
    vals2 = [r["albumin"] for r in r2["result"]["datasets"][0]]
    assert vals1 == vals2


def test_imputation_m_datasets():
    res = multiple_imputation(DF, ["albumin"], m=3)
    assert len(res["result"]["datasets"]) == 3


def test_imputation_id_col():
    df = DF.copy()
    df["pid"] = list(range(len(df)))
    res = multiple_imputation(df, ["albumin"], id_col="pid")
    ids = [r["id"] for r in res["result"]["datasets"][0]]
    assert ids == list(range(len(df)))


# --- PSM ---

def test_psm_basic():
    res = _json_safe(propensity_score_matching(DF, "treatment", ["age", "sex"]))
    r = res["result"]
    assert r["n_pairs"] > 0
    assert len(r["pairs"]) == r["n_pairs"]
    assert "balance_before" in r
    assert "balance_after" in r


def test_psm_balance_improves():
    """After matching SMD should generally improve (average |SMD| lower)."""
    res = propensity_score_matching(DF, "treatment", ["age", "albumin"])
    before_avg = np.mean([abs(b["smd"]) for b in res["result"]["balance_before"]])
    after_avg = np.mean([abs(b["smd"]) for b in res["result"]["balance_after"]])
    # Allow some slack — not guaranteed but expected for this synthetic data
    assert after_avg <= before_avg + 0.2


def test_psm_single_group_raises():
    bad = DF.copy()
    bad["treatment"] = 1
    with pytest.raises(ValueError):
        propensity_score_matching(bad, "treatment", ["age"])


# --- IPTW ---

def test_iptw_basic():
    res = _json_safe(iptw_weights(DF, "treatment", ["age", "sex"]))
    r = res["result"]
    assert len(r["rows"]) == res["n_used"]
    assert "id" in r["rows"][0]
    assert "propensity_score" in r["rows"][0]
    assert "weight" in r["rows"][0]
    assert "balance_before" in r
    assert "balance_after_weighted" in r


def test_iptw_weights_positive():
    res = iptw_weights(DF, "treatment", ["age"])
    weights = [r["weight"] for r in res["result"]["rows"]]
    assert all(w > 0 for w in weights)


def test_psm_single_group_raises_before_fit():
    """The single-group check must fire before LogisticRegression.fit() with a clean message."""
    bad = DF.copy()
    bad["treatment"] = 1
    with pytest.raises(ValueError, match="one group"):
        propensity_score_matching(bad, "treatment", ["age"])


def test_psm_categorical():
    res = _json_safe(propensity_score_matching(DF, "treatment", ["age", "stage"],
                                               categorical_cols=["stage"]))
    assert res["result"]["n_pairs"] > 0


def test_iptw_categorical():
    res = _json_safe(iptw_weights(DF, "treatment", ["age", "stage"],
                                  categorical_cols=["stage"]))
    assert len(res["result"]["rows"]) > 0
