"""
BREACH Score — Breast Cancer Evolutionary Adaptation and Collapse of Host Defenses.

STATUS: Experimental / unvalidated. Proposed score, not yet tested against
outcome data. Discrimination performance relative to NPI and CTS5 is the
open question this application exists to help answer — do not present
BREACH output as a validated clinical risk estimate.
"""

from dataclasses import dataclass

SCORE_STATUS = "experimental_unvalidated"


@dataclass(frozen=True)
class BreachInputs:
    ki67_percent: float          # Ki67 proliferation index, 0-100
    tumor_grade: int             # Nottingham grade: 1, 2, or 3
    neutrophils_10e9_L: float    # absolute neutrophil count, x10^9/L
    lymphocytes_10e9_L: float    # absolute lymphocyte count, x10^9/L
    albumin_g_L: float           # serum albumin, g/L (NOT g/dL)
    hemoglobin_g_dL: float       # hemoglobin, g/dL
    lvi_present: bool            # lymphovascular invasion


def _validate(x: BreachInputs) -> None:
    if not (0 <= x.ki67_percent <= 100):
        raise ValueError(f"ki67_percent must be 0-100, got {x.ki67_percent}")
    if x.tumor_grade not in (1, 2, 3):
        raise ValueError(f"tumor_grade must be 1, 2, or 3, got {x.tumor_grade}")
    if x.lymphocytes_10e9_L <= 0:
        raise ValueError("lymphocytes_10e9_L must be > 0 (used as a divisor for NLR)")
    if x.albumin_g_L <= 0:
        raise ValueError("albumin_g_L must be > 0 (used as a divisor)")
    if x.hemoglobin_g_dL <= 0:
        raise ValueError("hemoglobin_g_dL must be > 0 (used as a divisor)")
    if x.neutrophils_10e9_L < 0:
        raise ValueError("neutrophils_10e9_L cannot be negative")
    # unit-plausibility guard, not a hard error — flagged, not blocked
    if x.albumin_g_L > 55:
        raise ValueError(
            f"albumin_g_L={x.albumin_g_L} is implausibly high — "
            "check this wasn't entered in g/dL instead of g/L"
        )


def compute_nlr(x: BreachInputs) -> float:
    """Neutrophil-to-lymphocyte ratio, exposed separately since it's
    clinically interpretable on its own, not just an internal step."""
    return x.neutrophils_10e9_L / x.lymphocytes_10e9_L


def compute_breach(x: BreachInputs) -> float:
    """Compute the BREACH score for one patient. Raises ValueError on
    out-of-range or physically invalid inputs — never silently proceeds
    with bad data."""
    _validate(x)
    nlr = compute_nlr(x)
    numerator = x.ki67_percent * x.tumor_grade * nlr
    denominator = x.albumin_g_L * x.hemoglobin_g_dL
    lvi_multiplier = 2.0 if x.lvi_present else 1.0
    return (numerator / denominator) * lvi_multiplier