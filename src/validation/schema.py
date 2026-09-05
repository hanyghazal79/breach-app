"""
Cohort data validation layer.

Validates an entire uploaded cohort against the project's data dictionary
BEFORE any score is computed. Produces a structured report -- which rows
passed, which failed and why, and per-column missingness -- rather than
letting a bad row crash score computation deep inside a loop.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator


def _check_plausible_range(value: float, low: float, high: float,
                            field_name: str, likely_mistake: str) -> float:
    """Shared range-check used by every unit-plausibility validator below.
    This is a heuristic, not a hard clinical truth -- see the caveat on
    tumor_size_mm for a case where that distinction actually matters."""
    if value < low or value > high:
        raise ValueError(
            f"{field_name}={value} is outside the plausible range "
            f"[{low}, {high}] -- {likely_mistake}"
        )
    return value


class PatientRecord(BaseModel):
    patient_id: str
    age_years: float = Field(gt=0, lt=120)
    menopausal_status: Literal["pre", "post"]
    tumor_size_mm: float = Field(gt=0, lt=250)
    tumor_grade: Literal[1, 2, 3]
    nodes_positive_count: int = Field(ge=0)
    lvi_status: Literal["absent", "present"]
    ki67_percent: float = Field(ge=0, le=100)
    er_status: Literal["positive", "negative"]
    neutrophils_10e9_L: float = Field(ge=0)
    lymphocytes_10e9_L: float = Field(gt=0)     # divisor in NLR -- must be > 0
    albumin_g_L: float = Field(gt=0)
    hemoglobin_g_dL: float = Field(gt=0)
    endocrine_therapy_completed_5y: bool
    disease_free_at_5y: bool
    followup_time_months: Optional[float] = Field(default=None, ge=0)
    event_recurrence: Optional[int] = Field(default=None, ge=0, le=1)

    @field_validator("albumin_g_L")
    @classmethod
    def check_albumin_plausible(cls, v: float) -> float:
        # Normal serum albumin is ~35-50 g/L. A g/dL value entered directly
        # (e.g. "4.2" instead of "42") lands far below this range.
        return _check_plausible_range(
            v, low=20, high=60, field_name="albumin_g_L",
            likely_mistake="check this wasn't entered in g/dL (multiply by 10)",
        )

    @field_validator("hemoglobin_g_dL")
    @classmethod
    def check_hemoglobin_plausible(cls, v: float) -> float:
        # Normal Hgb is ~7-20 g/dL even across anemia/polycythemia extremes.
        # The SI unit (g/L, ~70-180) entered here would be ~10x too high --
        # the mirror-image mistake of the albumin case.
        return _check_plausible_range(
            v, low=5, high=22, field_name="hemoglobin_g_dL",
            likely_mistake="check this wasn't entered in g/L (divide by 10)",
        )

    # @field_validator("tumor_size_mm")
    # @classmethod
    # def check_tumor_size_plausible(cls, v: float) -> float:
    #     # NOTE: this is a soft heuristic, not a certainty. Genuine
    #     # microinvasive cancers can be <2mm, so a low value here is
    #     # flagged, not silently rejected elsewhere -- the researcher
    #     # should confirm rather than have the row auto-dropped.
    #     return _check_plausible_range(
    #         v, low=1, high=200, field_name="tumor_size_mm",
    #         likely_mistake="check this wasn't entered in cm (multiply by 10)",
    #     )

    @field_validator("neutrophils_10e9_L")
    @classmethod
    def check_neutrophils_plausible(cls, v: float) -> float:
        # A value of 4200 instead of 4.2 (cells/uL vs x10^9/L) is a
        # realistic transcription error from a raw CBC printout.
        return _check_plausible_range(
            v, low=0, high=50, field_name="neutrophils_10e9_L",
            likely_mistake="check this wasn't entered in cells/uL (divide by 1000)",
        )

    @field_validator("lymphocytes_10e9_L")
    @classmethod
    def check_lymphocytes_plausible(cls, v: float) -> float:
        return _check_plausible_range(
            v, low=0.1, high=30, field_name="lymphocytes_10e9_L",
            likely_mistake="check this wasn't entered in cells/uL (divide by 1000)",
        )


REQUIRED_COLUMNS = list(PatientRecord.model_fields.keys())


@dataclass
class ValidationReport:
    n_total: int
    n_valid: int
    n_invalid: int
    valid_records: list[PatientRecord] = field(default_factory=list)
    row_errors: dict[int, list[str]] = field(default_factory=dict)
    missingness: dict[str, float] = field(default_factory=dict)


def validate_cohort(df: pd.DataFrame) -> ValidationReport:
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Uploaded file is missing required columns: {missing_cols}")

    missingness = {
        col: round(df[col].isna().mean() * 100, 1) for col in REQUIRED_COLUMNS
    }

    valid_records: list[PatientRecord] = []
    row_errors: dict[int, list[str]] = {}

    for idx, row in df.iterrows():
        row_clean = row.where(pd.notnull(row), None)
        try:
            record = PatientRecord(**row_clean.to_dict())
            valid_records.append(record)
        except ValidationError as e:
            row_errors[idx] = [f"{err['loc'][0]}: {err['msg']}" for err in e.errors()]

    return ValidationReport(
        n_total=len(df),
        n_valid=len(valid_records),
        n_invalid=len(row_errors),
        valid_records=valid_records,
        row_errors=row_errors,
        missingness=missingness,
    )