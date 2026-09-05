"""
CTS5 — Clinical Treatment Score post-5-years.
Source: Dowsett M et al., J Clin Oncol 2018;36(19):1941-1948.
Derived and validated in postmenopausal, ER+/HER2- patients recurrence-free
at 5 years of endocrine therapy. Applying it outside that population is an
extrapolation — code surfaces this via cts5_eligible(), never hides it.
"""

from dataclasses import dataclass

SCORE_STATUS = "established_validated"
SIZE_CAP_MM = 30.0  # per original derivation — prevents non-monotonic artifact


@dataclass(frozen=True)
class Cts5Inputs:
    age_years: float
    tumor_size_mm: float
    tumor_grade: int
    nodes_positive_count: int
    menopausal_status: str   # "pre" or "post"


def _cts5_nodal_category(n: int) -> int:
    """CTS5's own 5-category nodal grouping — deliberately different
    granularity from NPI's 3-category scheme."""
    if n < 0:
        raise ValueError("nodes_positive_count cannot be negative")
    if n == 0:
        return 0
    if n == 1:
        return 1
    if n <= 3:
        return 2
    if n <= 9:
        return 3
    return 4


def cts5_eligible(x: Cts5Inputs) -> bool:
    """True only for the population CTS5 was validated in. Callers
    (the discrimination engine, the UI) must check this before treating
    a CTS5 value as clinically meaningful."""
    return x.menopausal_status == "post"


def compute_cts5(x: Cts5Inputs) -> float:
    if x.tumor_grade not in (1, 2, 3):
        raise ValueError(f"tumor_grade must be 1, 2, or 3, got {x.tumor_grade}")
    size_mm = min(x.tumor_size_mm, SIZE_CAP_MM)
    nodes_cat = _cts5_nodal_category(x.nodes_positive_count)
    inner = (0.093 * size_mm - 0.001 * (size_mm ** 2)
             + 0.375 * x.tumor_grade + 0.017 * x.age_years)
    return 0.438 * nodes_cat + 0.988 * inner