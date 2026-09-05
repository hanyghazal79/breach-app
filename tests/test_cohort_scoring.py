import pandas as pd  # add this import if not already present
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.validation.schema import PatientRecord
from src.scores.cohort_scoring import build_scored_cohort

def _record(**overrides):
    base = dict(
        patient_id="SYN-0001", age_years=58, menopausal_status="post",
        tumor_size_mm=22, tumor_grade=2, nodes_positive_count=1,
        lvi_status="absent", ki67_percent=18.5, er_status="positive",
        neutrophils_10e9_L=4.2, lymphocytes_10e9_L=1.8,
        albumin_g_L=42, hemoglobin_g_dL=13.1,
        endocrine_therapy_completed_5y=True, disease_free_at_5y=True,
        followup_time_months=84, event_recurrence=0,
    )
    base.update(overrides)
    return PatientRecord(**base)

def test_build_scored_cohort_matches_hand_verified_values():
    syn0001 = _record()
    syn0002 = _record(
        patient_id="SYN-0002", age_years=47, menopausal_status="pre",
        tumor_size_mm=31, tumor_grade=3, nodes_positive_count=4,
        lvi_status="present", ki67_percent=45.0, er_status="negative",
        neutrophils_10e9_L=6.8, lymphocytes_10e9_L=1.1,
        albumin_g_L=35, hemoglobin_g_dL=10.9,
        endocrine_therapy_completed_5y=False, disease_free_at_5y=False,
        followup_time_months=36, event_recurrence=1,
    )
    df = build_scored_cohort([syn0001, syn0002])
    row1 = df[df.patient_id == "SYN-0001"].iloc[0]
    row2 = df[df.patient_id == "SYN-0002"].iloc[0]

    assert row1["breach"] == pytest.approx(0.156914, rel=1e-4)
    assert row1["npi"] == pytest.approx(4.44, rel=1e-4)
    assert row1["cts5"] == pytest.approx(3.696424, rel=1e-4)
    assert row1["cts5_eligible"] == True          # was: is True

    assert row2["breach"] == pytest.approx(4.37546, rel=1e-4)
    assert row2["npi"] == pytest.approx(6.62, rel=1e-4)
    assert pd.isna(row2["cts5"])                  # was: is None
    assert row2["cts5_eligible"] == False         # was: is False

def test_patients_without_outcome_excluded():
    no_outcome = _record(patient_id="SYN-NO-OUTCOME",
                          followup_time_months=None, event_recurrence=None)
    df = build_scored_cohort([_record(), no_outcome])
    assert len(df) == 1