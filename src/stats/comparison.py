"""
Head-to-head statistical comparison between two risk scores' C-indices,
using paired bootstrap resampling.

WHY PAIRED: both scores are evaluated on the SAME patients. Resampling
patients (rows) with replacement and recomputing both scores' C-index on
each identical resample preserves the correlation between the two scores'
errors. Resampling each score independently would overstate uncertainty
and could show a "significant" difference between a score and itself.
"""

import numpy as np
import pandas as pd

from src.stats.discrimination import compute_c_index


def bootstrap_compare_c_index(
    df: pd.DataFrame,
    score_a_col: str,
    score_b_col: str,
    time_col: str = "followup_time_months",
    event_col: str = "event_recurrence",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap comparison of two scores' C-index on the SAME
    cohort. Caller must pass in the correct SUBSET already -- e.g. only
    CTS5-eligible patients when comparing BREACH vs CTS5. This function
    makes no eligibility decisions itself."""
    subset = df.dropna(
        subset=[score_a_col, score_b_col, time_col, event_col]
    ).reset_index(drop=True)
    n = len(subset)
    if n < 10:
        raise ValueError(
            f"Only {n} complete paired rows available -- too few for a "
            "meaningful bootstrap comparison (need at least 10)"
        )

    c_a = compute_c_index(subset, score_a_col, time_col, event_col)
    c_b = compute_c_index(subset, score_b_col, time_col, event_col)

    rng = np.random.default_rng(seed)
    deltas = []
    n_failed = 0
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)   # SAME resampled patients for both scores
        resample = subset.iloc[idx]
        try:
            ca = compute_c_index(resample, score_a_col, time_col, event_col)
            cb = compute_c_index(resample, score_b_col, time_col, event_col)
            deltas.append(ca - cb)
        except ValueError:
            n_failed += 1   # e.g. a resample with no comparable pairs
            continue

    deltas = np.array(deltas)
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])

    observed_delta = c_a - c_b
    if observed_delta >= 0:
        p_value = min(2 * np.mean(deltas <= 0), 1.0)
    else:
        p_value = min(2 * np.mean(deltas >= 0), 1.0)

    return {
        "c_index_a": c_a,
        "c_index_b": c_b,
        "delta": observed_delta,
        "delta_ci_95": (ci_low, ci_high),
        "p_value": p_value,
        "n_bootstrap_used": len(deltas),
        "n_bootstrap_failed": n_failed,
        "n_patients": n,
    }