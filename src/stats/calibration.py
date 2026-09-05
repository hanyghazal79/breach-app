"""
Kaplan-Meier stratification + calibration for a continuous risk score.

KM STRATIFICATION: splits patients into tertiles by score, tests whether
the three groups' survival curves differ (log-rank test). This answers a
group-level question, separate from Step 5's pairwise C-index.

CALIBRATION: fits a Cox model with the score as its sole covariate to get
a predicted survival probability at a chosen time horizon, then compares
mean predicted survival against observed KM survival within score bins.
NOTE ON PRECISION: unlike the score formulas (Steps 2-3), a Cox partial
likelihood fit on realistic data cannot be hand-verified to an exact
number -- the tests for this module check DIRECTIONAL correctness
(higher risk -> lower survival), not a fabricated precise expected value.
"""

import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test


def stratify_into_tertiles(df: pd.DataFrame, score_col: str) -> tuple[pd.DataFrame, tuple[float, float]]:
    """Adds a 'risk_group' column (Low/Mid/High) based on score tertiles.
    Returns the augmented dataframe and the two cutoff values used."""
    q1, q2 = df[score_col].quantile([1 / 3, 2 / 3])

    def _group(v: float) -> str:
        if v <= q1:
            return "Low"
        elif v <= q2:
            return "Mid"
        return "High"

    out = df.copy()
    out["risk_group"] = out[score_col].apply(_group)
    return out, (q1, q2)


def km_logrank_test(df: pd.DataFrame, group_col: str,
                     time_col: str = "followup_time_months",
                     event_col: str = "event_recurrence") -> dict:
    """Tests whether survival differs across the risk groups. A small
    p-value means the groups are NOT just splitting noise."""
    result = multivariate_logrank_test(df[time_col], df[group_col], df[event_col])
    return {"test_statistic": result.test_statistic, "p_value": result.p_value}


def calibration_table(df: pd.DataFrame, score_col: str, time_horizon: float,
                       time_col: str = "followup_time_months",
                       event_col: str = "event_recurrence",
                       n_groups: int = 4) -> pd.DataFrame:
    """Fits a Cox model on score_col, predicts survival probability at
    time_horizon per patient, bins patients into n_groups by score, and
    compares mean predicted survival against observed KM survival per bin.

    CAVEAT: if time_horizon exceeds a bin's longest follow-up, the KM
    estimate for that bin is a flat extrapolation of its last known value,
    not a real observation at that horizon -- check n_patients and each
    bin's max follow-up before trusting a bin's row."""
    cph = CoxPHFitter()
    cph.fit(df[[time_col, event_col, score_col]], duration_col=time_col, event_col=event_col)

    predicted = cph.predict_survival_function(df[[score_col]], times=[time_horizon])
    work = df.copy()
    work["predicted_survival"] = predicted.iloc[0].values
    work["risk_bin"] = pd.qcut(work[score_col], q=n_groups, labels=False, duplicates="drop")

    rows = []
    for bin_id, group in work.groupby("risk_bin"):
        kmf = KaplanMeierFitter()
        kmf.fit(group[time_col], group[event_col])
        observed = kmf.survival_function_at_times(time_horizon).values[0]
        rows.append({
            "risk_bin": bin_id,
            "n_patients": len(group),
            "mean_score": group[score_col].mean(),
            "mean_predicted_survival": group["predicted_survival"].mean(),
            "observed_survival_km": observed,
            "max_followup_in_bin": group[time_col].max(),
        })

    return pd.DataFrame(rows).sort_values("risk_bin").reset_index(drop=True)