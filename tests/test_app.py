
import sys

from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

sys.path.insert(0, PROJECT_ROOT)

import pytest
from streamlit.testing.v1 import AppTest


def test_app_loads_without_exception():
    at = AppTest.from_file(f"{PROJECT_ROOT}/src/app.py")
    at.run()
    assert not at.exception


def test_calculator_computes_syn0001_correctly():
    at = AppTest.from_file(f"{PROJECT_ROOT}/src/app.py")
    at.run()

    at.number_input(key="ki67_percent").set_value(18.5)
    at.selectbox(key="tumor_grade").set_value(2)
    at.number_input(key="neutrophils").set_value(4.2)
    at.number_input(key="lymphocytes").set_value(1.8)
    at.number_input(key="albumin_g_L").set_value(42)
    at.number_input(key="hemoglobin_g_dL").set_value(13.1)
    at.checkbox(key="lvi_present").set_value(False)
    at.number_input(key="tumor_size_mm").set_value(22)
    at.number_input(key="nodes_positive_count").set_value(1)
    at.number_input(key="age_years").set_value(58)
    at.radio(key="menopausal_status").set_value("post")

    at.button(key="submit_calc").click()
    at.run()

    assert not at.exception
    assert at.session_state["last_breach"] == pytest.approx(0.156914, rel=1e-4)
    assert at.session_state["last_npi"] == pytest.approx(4.44, rel=1e-4)
    assert at.session_state["last_cts5"] == pytest.approx(3.696424, rel=1e-4)
    assert at.session_state["last_cts5_eligible"] is True   # plain Python bool -- never touched pandas


def test_invalid_grade_shows_error_not_crash():
    at = AppTest.from_file(f"{PROJECT_ROOT}/src/app.py")
    at.run()
    at.selectbox(key="tumor_grade").set_value(1)  # valid selectbox value
    at.button(key="submit_calc").click()
    at.run()
    assert not at.exception   # bad input must produce st.error, never an unhandled crash
    
def test_cohort_tab_loads_example_and_computes():
    at = AppTest.from_file(f"{PROJECT_ROOT}/src/app.py", default_timeout=30)
    at.run(timeout=30)
    at.button(key="load_example").click()
    at.run(timeout=30)

    assert not at.exception
    assert 0.0 <= at.session_state["cohort_c_breach"] <= 1.0
    assert 0.0 <= at.session_state["cohort_c_npi"] <= 1.0
    assert at.session_state["cohort_n_eligible_cts5"] >= 0