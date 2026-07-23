import json
import pandas as pd
import pytest
from medstats import cox_regression, kaplan_meier

DF = pd.read_csv("sample_data.csv")


def _json_safe(result):
    """Assert result is json-serialisable and return it."""
    json.dumps(result)  # raises if not safe
    return result


def test_cox_returns_expected_keys():
    res = _json_safe(cox_regression(DF, "time", "event", ["age", "albumin"]))
    assert res["method"] == "cox_regression"
    assert res["status"] in ("ok", "warning")
    assert 0.5 < res["result"]["c_index"] < 1.0
    terms = res["result"]["terms"]
    assert any(t["name"] == "age" for t in terms)


def test_cox_age_hr_positive():
    """Older age should increase hazard (HR > 1) in this synthetic dataset."""
    res = cox_regression(DF, "time", "event", ["age"])
    age_term = next(t for t in res["result"]["terms"] if t["name"] == "age")
    assert age_term["hr"] > 1.0


def test_cox_requires_binary_event():
    bad = DF.copy()
    bad["event"] = 2
    with pytest.raises(ValueError):
        cox_regression(bad, "time", "event", ["age"])


def test_cox_categorical():
    res = _json_safe(cox_regression(DF, "time", "event", ["age", "stage"],
                                    categorical_cols=["stage"]))
    term_names = [t["name"] for t in res["result"]["terms"]]
    assert any("stage" in n for n in term_names)


def test_cox_string_col_raises_without_categorical():
    with pytest.raises(ValueError, match="categorical_cols"):
        cox_regression(DF, "time", "event", ["age", "stage"])


def test_km_no_group():
    res = _json_safe(kaplan_meier(DF, "time", "event"))
    groups = res["result"]["groups"]
    assert len(groups) == 1
    assert groups[0]["name"] == "overall"
    assert "overall_logrank_p" not in res["result"]


def test_km_two_groups_logrank():
    res = _json_safe(kaplan_meier(DF, "time", "event", group_col="treatment"))
    assert "overall_logrank_p" in res["result"]
    p = res["result"]["overall_logrank_p"]
    assert 0.0 <= p <= 1.0


def test_km_three_groups():
    df = DF.copy()
    df["grp3"] = pd.cut(df["age"], bins=3, labels=["low", "mid", "high"])
    res = _json_safe(kaplan_meier(df, "time", "event", group_col="grp3"))
    assert len(res["result"]["groups"]) == 3
    assert "overall_logrank_p" in res["result"]
