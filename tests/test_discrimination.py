import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stats.discrimination import compute_c_index


def test_perfect_concordance():
    # Hand-verified above: all 6 pairs concordant -> C-index = 1.0
    df = pd.DataFrame({
        "patient_id": ["P1", "P2", "P3", "P4"],
        "followup_time_months": [10, 20, 30, 5],
        "event_recurrence":      [1,  1,  0,  1],
        "risk_score":             [8,  5,  2,  9],
    })
    assert compute_c_index(df, score_col="risk_score") == pytest.approx(1.0)


def test_one_discordant_pair():
    # P2's score dropped 5 -> 1, breaking only pair (P2,P3):
    # P2 dies at t=20 (should outrank P3, censored at t=30) but 1 < 2.
    # Hand-counted: 5 concordant, 1 discordant -> C-index = 5/6.
    df = pd.DataFrame({
        "patient_id": ["P1", "P2", "P3", "P4"],
        "followup_time_months": [10, 20, 30, 5],
        "event_recurrence":      [1,  1,  0,  1],
        "risk_score":             [8,  1,  2,  9],
    })
    assert compute_c_index(df, score_col="risk_score") == pytest.approx(5 / 6, rel=1e-4)


def test_sign_convention_not_accidentally_inverted():
    # Regression guard: removing the negation on this exact perfect-
    # concordance example would silently produce ~0.0 instead of 1.0.
    df = pd.DataFrame({
        "patient_id": ["P1", "P2", "P3", "P4"],
        "followup_time_months": [10, 20, 30, 5],
        "event_recurrence":      [1,  1,  0,  1],
        "risk_score":             [8,  5,  2,  9],
    })
    assert compute_c_index(df, score_col="risk_score") > 0.9


def test_missing_scores_dropped():
    df = pd.DataFrame({
        "patient_id": ["P1", "P2", "P3"],
        "followup_time_months": [10, 20, 30],
        "event_recurrence": [1, 1, 0],
        "risk_score": [8.0, None, 2.0],   # e.g. a CTS5-ineligible patient
    })
    assert compute_c_index(df, score_col="risk_score") == pytest.approx(1.0)