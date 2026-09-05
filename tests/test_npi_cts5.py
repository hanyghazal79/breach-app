import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from src.scores.npi import NpiInputs, compute_npi
from src.scores.cts5 import Cts5Inputs, compute_cts5, cts5_eligible

def test_npi_syn0001():
    # size 22mm = 2.2cm, grade 2, 1 node -> nodal stage 2 (1-3 range)
    # NPI = 0.2*2.2 + 2 + 2 = 4.44
    x = NpiInputs(tumor_size_mm=22, tumor_grade=2, nodes_positive_count=1)
    assert compute_npi(x) == pytest.approx(4.44, rel=1e-4)

def test_npi_syn0002():
    # size 31mm = 3.1cm, grade 3, 4 nodes -> nodal stage 3 (>=4)
    # NPI = 0.2*3.1 + 3 + 3 = 6.62
    x = NpiInputs(tumor_size_mm=31, tumor_grade=3, nodes_positive_count=4)
    assert compute_npi(x) == pytest.approx(6.62, rel=1e-4)

def test_cts5_syn0001():
    # age 58, size 22mm (under cap), grade 2, 1 node -> nodes_cat 1
    # inner = 0.093*22 - 0.001*484 + 0.375*2 + 0.017*58 = 3.298
    # CTS5 = 0.438*1 + 0.988*3.298 = 3.696424
    
    x = Cts5Inputs(age_years=58, tumor_size_mm=22, tumor_grade=2,
                    nodes_positive_count=1, menopausal_status="post")
    assert compute_cts5(x) == pytest.approx(3.696424, rel=1e-4)
    assert cts5_eligible(x) is True

def test_cts5_syn0002_size_capped_and_ineligible():
    # size 31mm -> capped to 30mm; 4 nodes -> nodes_cat 3
    # inner = 0.093*30 - 0.001*900 + 0.375*3 + 0.017*47 = 3.814
    # CTS5 = 0.438*3 + 0.988*3.814 = 5.082232
    """
        0.988 × 3.814 = 3.814 − (3.814 × 0.012) = 3.814 − 0.045768 = 3.768232
        0.438 × 3    = 1.314
        Total        = 1.314 + 3.768232 = 5.082232
    """
    x = Cts5Inputs(age_years=47, tumor_size_mm=31, tumor_grade=3,
                    nodes_positive_count=4, menopausal_status="pre")
    assert compute_cts5(x) == pytest.approx(5.082232, rel=1e-4)
    # premenopausal -> outside validated population, even though it computes
    assert cts5_eligible(x) is False

def test_npi_invalid_grade_raises():
    x = NpiInputs(tumor_size_mm=22, tumor_grade=4, nodes_positive_count=1)
    with pytest.raises(ValueError, match="tumor_grade"):
        compute_npi(x)