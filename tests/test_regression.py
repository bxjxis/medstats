import json
import pandas as pd
import pytest
from medstats import logistic_regression

DF = pd.read_csv("sample_data.csv")


def _json_safe(result):
    json.dumps(result)
    return result


def test_logistic_returns_keys():
    res = _json_safe(logistic_regression(DF, "outcome", ["age", "albumin"]))
    assert res["method"] == "logistic_regression"
    r = res["result"]
    assert "terms" in r
    preds = r["predicted_probabilities"]
    assert len(preds) == res["n_used"]
    # Each prediction must carry its row identity
    assert "id" in preds[0]
    assert "predicted_probability" in preds[0]


def test_logistic_id_col():
    df = DF.copy()
    df["patient_id"] = ["P" + str(i) for i in range(len(df))]
    res = logistic_regression(df, "outcome", ["age"], id_col="patient_id")
    ids = [r["id"] for r in res["result"]["predicted_probabilities"]]
    assert all(str(i).startswith("P") for i in ids)


def test_logistic_categorical_cols():
    res = _json_safe(logistic_regression(DF, "outcome", ["age", "stage"], categorical_cols=["stage"]))
    term_names = [t["name"] for t in res["result"]["terms"]]
    # stage dummy column should appear
    assert any("stage" in n for n in term_names)


def test_logistic_string_col_raises_without_categorical():
    with pytest.raises(ValueError, match="categorical_cols"):
        logistic_regression(DF, "outcome", ["age", "stage"])


def test_logistic_epv_warning():
    """Very few events relative to covariates should trigger EPV warning."""
    # Create a tiny dataset with rare outcome
    small = DF.head(30).copy()
    small["rare_outcome"] = 0
    small.loc[small.index[:2], "rare_outcome"] = 1
    res = logistic_regression(small, "rare_outcome", ["age", "albumin", "bilirubin"])
    assert any("EPV" in w for w in res["warnings"])


def test_logistic_age_or_direction():
    """age should have OR > 1 (older → higher risk) in this synthetic dataset."""
    res = logistic_regression(DF, "outcome", ["age"])
    age_term = next(t for t in res["result"]["terms"] if t["name"] == "age")
    assert age_term["or"] > 1.0


def test_logistic_bad_outcome():
    bad = DF.copy()
    bad["outcome"] = 3
    with pytest.raises(ValueError):
        logistic_regression(bad, "outcome", ["age"])
