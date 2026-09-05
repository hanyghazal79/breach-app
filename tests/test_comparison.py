import numpy as np
import pandas as pd
import pytest
from src.stats.comparison import bootstrap_compare_c_index


def test_comparing_score_to_itself_gives_zero_delta():
    # Strongest test available: if resampling is truly paired, comparing
    # any score to itself MUST give delta == 0.0 on every resample, exactly.
    n = 30
    times = np.arange(1, n + 1).astype(float)
    events = np.ones(n, dtype=int)
    score = -times   # perfectly decreasing with time -> C-index = 1.0

    df = pd.DataFrame({
        "followup_time_months": times, "event_recurrence": events,
        "score_a": score, "score_b": score,   # identical column
    })
    result = bootstrap_compare_c_index(df, "score_a", "score_b", n_bootstrap=200, seed=1)
    assert result["delta"] == pytest.approx(0.0)
    assert result["delta_ci_95"][0] == pytest.approx(0.0, abs=1e-9)
    assert result["delta_ci_95"][1] == pytest.approx(0.0, abs=1e-9)
    assert result["p_value"] == pytest.approx(1.0)


def test_superior_score_beats_random_score():
    n = 40
    times = np.arange(1, n + 1).astype(float)
    events = np.ones(n, dtype=int)
    score_good = -times                                   # C-index = 1.0
    rng = np.random.default_rng(0)
    score_random = rng.permutation(score_good)             # scrambled -> ~no relationship

    df = pd.DataFrame({
        "followup_time_months": times, "event_recurrence": events,
        "score_a": score_good, "score_b": score_random,
    })
    result = bootstrap_compare_c_index(df, "score_a", "score_b", n_bootstrap=500, seed=2)
    assert result["c_index_a"] == pytest.approx(1.0)
    assert result["delta"] > 0.3
    assert result["delta_ci_95"][0] > 0    # CI excludes zero
    assert result["p_value"] < 0.05


def test_reproducible_with_fixed_seed():
    n = 20
    times = np.arange(1, n + 1).astype(float)
    events = np.array(([1, 1, 1, 0] * 5))
    df = pd.DataFrame({
        "followup_time_months": times, "event_recurrence": events,
        "score_a": -times + 0.1, "score_b": -times,
    })
    r1 = bootstrap_compare_c_index(df, "score_a", "score_b", n_bootstrap=100, seed=7)
    r2 = bootstrap_compare_c_index(df, "score_a", "score_b", n_bootstrap=100, seed=7)
    assert r1["delta_ci_95"] == r2["delta_ci_95"]
    assert r1["p_value"] == r2["p_value"]


def test_too_few_patients_raises():
    df = pd.DataFrame({
        "followup_time_months": [1, 2, 3], "event_recurrence": [1, 1, 0],
        "score_a": [3, 2, 1], "score_b": [1, 2, 3],
    })
    with pytest.raises(ValueError, match="too few"):
        bootstrap_compare_c_index(df, "score_a", "score_b")