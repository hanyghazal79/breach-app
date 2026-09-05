import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from src.validation.schema import validate_cohort

VALID_ROW = {
    "patient_id": "SYN-0001", "age_years": 58, "menopausal_status": "post",
    "tumor_size_mm": 22, "tumor_grade": 2, "nodes_positive_count": 1,
    "lvi_status": "absent", "ki67_percent": 18.5, "er_status": "positive",
    "neutrophils_10e9_L": 4.2, "lymphocytes_10e9_L": 1.8,
    "albumin_g_L": 42, "hemoglobin_g_dL": 13.1,
    "endocrine_therapy_completed_5y": True, "disease_free_at_5y": True,
    "followup_time_months": 84, "event_recurrence": 0,
}

def test_valid_row_passes():
    df = pd.DataFrame([VALID_ROW])
    report = validate_cohort(df)
    assert report.n_valid == 1
    assert report.n_invalid == 0

def test_albumin_unit_error_caught_at_cohort_level():
    # g/dL value (4.2) entered where g/L (42) was expected
    bad_row = dict(VALID_ROW, patient_id="SYN-BAD", albumin_g_L=4.2)
    df = pd.DataFrame([VALID_ROW, bad_row])
    report = validate_cohort(df)
    assert report.n_valid == 1
    assert report.n_invalid == 1
    assert 1 in report.row_errors
    assert "g/dL" in report.row_errors[1][0]

def test_missing_required_column_raises():
    df = pd.DataFrame([VALID_ROW]).drop(columns=["albumin_g_L"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_cohort(df)

def test_missingness_report():
    row_missing_hb = dict(VALID_ROW, hemoglobin_g_dL=None)
    df = pd.DataFrame([VALID_ROW, row_missing_hb])
    report = validate_cohort(df)
    assert report.missingness["hemoglobin_g_dL"] == 50.0
    
def test_hemoglobin_si_unit_error_caught():
    # SI unit (g/L, ~131) entered where g/dL (~13.1) was expected
    bad_row = dict(VALID_ROW, patient_id="SYN-BAD-HGB", hemoglobin_g_dL=131)
    df = pd.DataFrame([VALID_ROW, bad_row])
    report = validate_cohort(df)
    assert report.n_invalid == 1
    assert "g/L" in report.row_errors[1][0]

def test_neutrophils_cells_per_uL_error_caught():
    # 4200 cells/uL entered directly where 4.2 x10^9/L was expected
    bad_row = dict(VALID_ROW, patient_id="SYN-BAD-NEUT", neutrophils_10e9_L=4200)
    df = pd.DataFrame([VALID_ROW, bad_row])
    report = validate_cohort(df)
    assert report.n_invalid == 1
    assert "cells/uL" in report.row_errors[1][0]
    
def test_tumor_size_extreme_typo_caught():
    # a clear fat-finger typo (extra zero), not a unit-confusion case
    bad_row = dict(VALID_ROW, patient_id="SYN-BAD-SIZE", tumor_size_mm=2200)
    df = pd.DataFrame([VALID_ROW, bad_row])
    report = validate_cohort(df)
    assert report.n_invalid == 1

def test_tumor_size_small_but_plausible_value_passes():
    # a genuinely small tumor (e.g. T1a, 2.2mm) must NOT be rejected --
    # this documents the known limitation rather than hiding it
    small_row = dict(VALID_ROW, patient_id="SYN-SMALL", tumor_size_mm=2.2)
    df = pd.DataFrame([VALID_ROW, small_row])
    report = validate_cohort(df)
    assert report.n_invalid == 0