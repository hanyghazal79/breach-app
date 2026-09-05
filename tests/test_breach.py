import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from src.scores.breach import BreachInputs, compute_breach, compute_nlr

def test_breach_lvi_absent():
    # Hand calculation:
    # NLR = 4.2 / 1.8 = 2.33333...
    # numerator = 18.5 * 2 * 2.33333 = 86.3333
    # denominator = 42 * 13.1 = 550.2
    # ratio = 86.3333 / 550.2 = 0.156914
    # x (1 + 0) = 0.156914
    x = BreachInputs(
        ki67_percent=18.5, tumor_grade=2,
        neutrophils_10e9_L=4.2, lymphocytes_10e9_L=1.8,
        albumin_g_L=42, hemoglobin_g_dL=13.1, lvi_present=False,
    )
    assert compute_breach(x) == pytest.approx(0.156914, rel=1e-4)

def test_breach_lvi_present_doubles_score():
    # Same inputs as above but LVI present — score must exactly double.
    x_absent = BreachInputs(18.5, 2, 4.2, 1.8, 42, 13.1, lvi_present=False)
    x_present = BreachInputs(18.5, 2, 4.2, 1.8, 42, 13.1, lvi_present=True)
    assert compute_breach(x_present) == pytest.approx(2 * compute_breach(x_absent))

def test_breach_second_example():
    # Hand calculation:
    # NLR = 6.8 / 1.1 = 6.181818
    # numerator = 45 * 3 * 6.181818 = 834.545
    # denominator = 35 * 10.9 = 381.5
    # ratio = 834.545 / 381.5 = 2.18773
    # x (1 + 1) = 4.37546
    x = BreachInputs(
        ki67_percent=45.0, tumor_grade=3,
        neutrophils_10e9_L=6.8, lymphocytes_10e9_L=1.1,
        albumin_g_L=35, hemoglobin_g_dL=10.9, lvi_present=True,
    )
    assert compute_breach(x) == pytest.approx(4.37546, rel=1e-4)

def test_zero_lymphocytes_raises():
    x = BreachInputs(18.5, 2, 4.2, 0, 42, 13.1, False)
    with pytest.raises(ValueError, match="lymphocytes"):
        compute_breach(x)

def test_invalid_grade_raises():
    x = BreachInputs(18.5, 5, 4.2, 1.8, 42, 13.1, False)
    with pytest.raises(ValueError, match="tumor_grade"):
        compute_breach(x)

def test_albumin_unit_error_caught():
    # 4.2 g/dL entered where g/L was expected — plausible real mistake
    x = BreachInputs(18.5, 2, 4.2, 1.8, albumin_g_L=100, hemoglobin_g_dL=13.1, lvi_present=False)
    with pytest.raises(ValueError, match="implausibly high"):
        compute_breach(x)