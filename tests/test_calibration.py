import numpy as np
import pandas as pd
import pytest
from src.stats.calibration import stratify_into_tertiles, km_logrank_test, calibration_table


def _strongly_separated_cohort(n=60, seed=0):
    """Score strongly, monotonically drives shorter survival -- built so
    the DIRECTION of every result is known in advance, even though the
    exact Cox-fitted numbers aren't hand-derivable."""
    rng = np.random.default_rng(seed)
    scores = np.linspace(0, 10, n)
    noise = rng.normal(0, 2, n)
    times = np.clip(100 - 5 * scores + noise, 1, None)
    events = np.ones(n, dtype=int)
    return pd.DataFrame({
        "followup_time_months": times, "event_recurrence": events, "score": scores,
    })


def test_tertile_groups_have_expected_relative_sizes():
    df = _strongly_separated_cohort()
    out, (q1, q2) = stratify_into_tertiles(df, "score")
    counts = out["risk_group"].value_counts()
    # roughly equal thirds by construction (tertile cutoffs) -- not exact
    # due to ties/rounding, but no group should be wildly unbalanced
    for group in ["Low", "Mid", "High"]:
        assert 15 <= counts[group] <= 25   # ~20 each out of 60


def test_logrank_detects_real_separation():
    df = _strongly_separated_cohort()
    out, _ = stratify_into_tertiles(df, "score")
    result = km_logrank_test(out, "risk_group")
    assert result["p_value"] < 0.01   # strongly separated by construction


def test_logrank_finds_no_separation_when_score_is_random_noise():
    # Negative control: score has NO relationship to survival here.
    rng = np.random.default_rng(1)
    n = 60
    times = rng.exponential(50, n)
    events = np.ones(n, dtype=int)
    scores = rng.normal(0, 1, n)   # unrelated to times
    df = pd.DataFrame({"followup_time_months": times, "event_recurrence": events, "score": scores})
    out, _ = stratify_into_tertiles(df, "score")
    result = km_logrank_test(out, "risk_group")
    assert result["p_value"] > 0.05   # should NOT find significance in pure noise


def test_calibration_table_is_directionally_correct():
    # Cannot hand-verify a Cox model's exact fitted output -- verifying the
    # property we actually need instead: higher-risk bins must show LOWER
    # predicted survival AND LOWER observed survival than lower-risk bins.
    df = _strongly_separated_cohort()
    horizon = np.median(df["followup_time_months"])
    table = calibration_table(df, "score", time_horizon=horizon, n_groups=4)

    assert len(table) <= 4
    assert table["n_patients"].sum() == len(df)

    # sorted by risk_bin ascending == ascending mean score by construction
    predicted = table["mean_predicted_survival"].values
    observed = table["observed_survival_km"].values
    assert all(predicted[i] >= predicted[i + 1] for i in range(len(predicted) - 1))
    assert all(observed[i] >= observed[i + 1] for i in range(len(observed) - 1))