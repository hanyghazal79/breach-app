"""
Nottingham Prognostic Index (NPI).
Source: Haybittle JL et al., Br J Cancer 1982;45:361-366.
Established, validated score — coefficients are fixed, not tunable.
"""

from dataclasses import dataclass

SCORE_STATUS = "established_validated"


@dataclass(frozen=True)
class NpiInputs:
    tumor_size_mm: float       # canonical app unit is mm; NPI itself uses cm
    tumor_grade: int           # 1, 2, or 3
    nodes_positive_count: int  # raw positive node count, >= 0


def _npi_nodal_stage(nodes_positive_count: int) -> int:
    """NPI's own 3-category nodal grouping, derived from the raw count."""
    if nodes_positive_count < 0:
        raise ValueError("nodes_positive_count cannot be negative")
    if nodes_positive_count == 0:
        return 1
    if nodes_positive_count <= 3:
        return 2
    return 3


def compute_npi(x: NpiInputs) -> float:
    if x.tumor_grade not in (1, 2, 3):
        raise ValueError(f"tumor_grade must be 1, 2, or 3, got {x.tumor_grade}")
    tumor_size_cm = x.tumor_size_mm / 10.0
    nodal_stage = _npi_nodal_stage(x.nodes_positive_count)
    return 0.2 * tumor_size_cm + x.tumor_grade + nodal_stage