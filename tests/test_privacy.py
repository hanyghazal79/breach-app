"""
Privacy-by-design guarantee: the cohort upload -> validate -> score ->
discriminate -> compare -> plot pipeline must never write patient data (or
anything derived from it) to disk. Two independent checks:

1. End-to-end: run the full app flow inside an empty sandboxed directory
   and confirm it stays empty (catches evidence left behind).
2. Unit-level: block any write-mode file open during the pipeline
   functions directly (catches the ATTEMPT, even if later cleaned up).

SCOPE: this proves our own code and Streamlit's rendering for this flow
never write patient-data-adjacent files. It does not certify that no
third-party library ever writes an unrelated cache file (e.g. matplotlib's
font cache in the user's home directory) -- that's a different concern
from patient data persistence.
"""

import builtins
import shutil
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_cohort_analysis_never_writes_to_disk(tmp_path, monkeypatch):
    """End-to-end: the full app flow, sandboxed in an empty directory,
    must leave that directory exactly as it started (plus the one input
    file the 'researcher' explicitly provided up front)."""
    sandbox_data = tmp_path / "data"
    sandbox_data.mkdir()
    shutil.copy(PROJECT_ROOT / "data" / "example_cohort.csv", sandbox_data / "example_cohort.csv")

    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))

    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(PROJECT_ROOT / "src" / "app.py"), default_timeout=30)
    at.run(timeout=30)
    at.button(key="load_example").click()
    at.run(timeout=30)

    assert not at.exception

    after = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    new_or_changed = set(after) - set(before)
    assert not new_or_changed, f"Unexpected files created during cohort analysis: {new_or_changed}"


def test_pipeline_never_opens_a_file_for_writing(monkeypatch):
    """Defense in depth, independent of the UI layer: blocks any WRITE-mode
    file open during validate -> score -> discriminate -> compare ->
    stratify. Read-mode opens (loading the input CSV itself) are
    explicitly allowed -- only writes are forbidden."""
    real_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(c in mode for c in ("w", "a", "x", "+")):
            raise AssertionError(f"Unexpected WRITE-mode file open attempted: {file!r} (mode={mode!r})")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)

    from src.validation.schema import validate_cohort
    from src.scores.cohort_scoring import build_scored_cohort
    from src.stats.discrimination import compute_c_index
    from src.stats.comparison import bootstrap_compare_c_index
    from src.stats.calibration import stratify_into_tertiles, km_logrank_test

    df = pd.read_csv(PROJECT_ROOT / "data" / "example_cohort.csv")   # read -- allowed
    report = validate_cohort(df)
    scored = build_scored_cohort(report.valid_records)
    compute_c_index(scored, "breach")
    compute_c_index(scored, "npi")
    bootstrap_compare_c_index(scored, "breach", "npi", n_bootstrap=50)  # small n, this is a speed/safety check, not a stats result
    strat, _ = stratify_into_tertiles(scored, "breach")
    km_logrank_test(strat, "risk_group")
    # Reaching this line without an AssertionError IS the proof --
    # guarded_open raises immediately on any write attempt.