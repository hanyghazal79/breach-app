"""
Applies the three score formulas across a validated cohort, producing one
row per patient with all three risk scores plus outcome data. This is the
input the discrimination engine (src/stats/discrimination.py) consumes.
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scores.breach import BreachInputs, compute_breach
from src.scores.npi import NpiInputs, compute_npi
from src.scores.cts5 import Cts5Inputs, compute_cts5, cts5_eligible
from src.validation.schema import PatientRecord


def _to_breach_inputs(r: PatientRecord) -> BreachInputs:
    return BreachInputs(
        ki67_percent=r.ki67_percent,
        tumor_grade=r.tumor_grade,
        neutrophils_10e9_L=r.neutrophils_10e9_L,
        lymphocytes_10e9_L=r.lymphocytes_10e9_L,
        albumin_g_L=r.albumin_g_L,
        hemoglobin_g_dL=r.hemoglobin_g_dL,
        lvi_present=(r.lvi_status == "present"),
    )


def _to_npi_inputs(r: PatientRecord) -> NpiInputs:
    return NpiInputs(
        tumor_size_mm=r.tumor_size_mm,
        tumor_grade=r.tumor_grade,
        nodes_positive_count=r.nodes_positive_count,
    )


def _to_cts5_inputs(r: PatientRecord) -> Cts5Inputs:
    return Cts5Inputs(
        age_years=r.age_years,
        tumor_size_mm=r.tumor_size_mm,
        tumor_grade=r.tumor_grade,
        nodes_positive_count=r.nodes_positive_count,
        menopausal_status=r.menopausal_status,
    )


def build_scored_cohort(records: list[PatientRecord]) -> pd.DataFrame:
    """One row per patient with all three scores, plus outcomes. Patients
    missing outcome data are excluded here -- discrimination requires
    outcomes; the calculator-only use case (Step 8) never calls this."""
    rows = []
    for r in records:
        if r.followup_time_months is None or r.event_recurrence is None:
            continue

        cts5_inputs = _to_cts5_inputs(r)
        is_eligible = cts5_eligible(cts5_inputs)

        rows.append({
            "patient_id": r.patient_id,
            "followup_time_months": r.followup_time_months,
            "event_recurrence": r.event_recurrence,
            "breach": compute_breach(_to_breach_inputs(r)),
            "npi": compute_npi(_to_npi_inputs(r)),
            "cts5": compute_cts5(cts5_inputs) if is_eligible else None,
            "cts5_eligible": is_eligible,
        })

    return pd.DataFrame(rows)