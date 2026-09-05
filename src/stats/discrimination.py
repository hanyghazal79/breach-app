"""
Discrimination statistics: how well a risk score separates patients who
recur early from those who don't. Uses Harrell's concordance index (C-index)
via lifelines.

SIGN CONVENTION -- verified against lifelines source, not assumed from
memory: lifelines.utils.concordance_index expects predicted_scores where a
HIGHER value means LONGER predicted survival. BREACH/NPI/CTS5 are risk
scores -- higher means shorter survival. We negate before calling lifelines,
matching lifelines' own documented usage pattern for Cox partial hazards.
Getting this backwards doesn't crash -- it silently reports the score's
performance as its exact opposite.
"""

import pandas as pd
from lifelines.utils import concordance_index


def compute_c_index(df: pd.DataFrame, score_col: str,
                     time_col: str = "followup_time_months",
                     event_col: str = "event_recurrence") -> float:
    """C-index for one risk score column against observed outcomes.
    Drops rows where score_col is missing (e.g. CTS5 for ineligible
    patients) -- see Step 6 for whether that's a fair comparison."""
    subset = df.dropna(subset=[score_col, time_col, event_col])
    if len(subset) < 2:
        raise ValueError(f"Not enough complete rows to compute C-index for '{score_col}'")

    return concordance_index(
        event_times=subset[time_col],
        predicted_scores=-subset[score_col],   # negated -- see module docstring
        event_observed=subset[event_col],
    )