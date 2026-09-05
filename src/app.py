"""
BREACH Score Web App -- entrypoint.
Deliberately thin: collects input, calls src/scores/ functions (already
tested in Steps 2-3), displays results. Cohort mode (Step 9) validates,
scores, and statistically compares an uploaded cohort.
"""
import pandas as pd
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scores.breach import BreachInputs, compute_breach, compute_nlr
from src.scores.breach import SCORE_STATUS as BREACH_STATUS
from src.scores.npi import NpiInputs, compute_npi
from src.scores.npi import SCORE_STATUS as NPI_STATUS
from src.scores.cts5 import Cts5Inputs, compute_cts5, cts5_eligible
from src.scores.cts5 import SCORE_STATUS as CTS5_STATUS
from src.validation.schema import validate_cohort
from src.scores.cohort_scoring import build_scored_cohort
from src.stats.discrimination import compute_c_index
from src.stats.comparison import bootstrap_compare_c_index
from src.stats.calibration import stratify_into_tertiles, km_logrank_test
from src.stats.plots import plot_km_by_group


st.set_page_config(page_title="BREACH Score Calculator", layout="centered")
st.title("BREACH Score Web App")
st.caption("Research tool -- not for clinical decision-making.")

tab_calc, tab_cohort = st.tabs(["Single-Patient Calculator", "Cohort Validation"])

with tab_calc:
    st.header("Single-Patient Calculator")

    with st.form("calculator_form"):
        st.subheader("Shared inputs")
        col1, col2 = st.columns(2)
        with col1:
            age_years = st.number_input("Age (years)", 18, 110, 60, key="age_years")
            tumor_grade = st.selectbox("Tumor grade", [1, 2, 3], index=1, key="tumor_grade")
            tumor_size_mm = st.number_input(
                "Tumor size (millimeters)", 1.0, 200.0, 20.0,
                help="Enter in millimeters, not centimeters.", key="tumor_size_mm",
            )
        with col2:
            menopausal_status = st.radio("Menopausal status", ["pre", "post"], key="menopausal_status")
            nodes_positive_count = st.number_input(
                "Number of positive nodes", 0, 50, 0, step=1, key="nodes_positive_count",
            )
            lvi_present = st.checkbox("Lymphovascular invasion (LVI) present", key="lvi_present")

        st.subheader("BREACH-specific inputs")
        col3, col4 = st.columns(2)
        with col3:
            ki67_percent = st.number_input("Ki67 (%)", 0.0, 100.0, 20.0, key="ki67_percent")
            neutrophils = st.number_input("Neutrophils (x10^9/L)", 0.0, 50.0, 4.0, key="neutrophils")
            lymphocytes = st.number_input("Lymphocytes (x10^9/L)", 0.1, 30.0, 2.0, key="lymphocytes")
        with col4:
            albumin_g_L = st.number_input(
                "Albumin (g/L -- NOT g/dL)", 1.0, 60.0, 40.0,
                help="Normal range ~35-50 g/L. If your lab reports g/dL, multiply by 10.",
                key="albumin_g_L",
            )
            hemoglobin_g_dL = st.number_input("Hemoglobin (g/dL)", 1.0, 22.0, 13.0, key="hemoglobin_g_dL")

        submitted = st.form_submit_button("Calculate scores", key="submit_calc")

    if submitted:
        try:
            breach_inputs = BreachInputs(
                ki67_percent=ki67_percent, tumor_grade=tumor_grade,
                neutrophils_10e9_L=neutrophils, lymphocytes_10e9_L=lymphocytes,
                albumin_g_L=albumin_g_L, hemoglobin_g_dL=hemoglobin_g_dL,
                lvi_present=lvi_present,
            )
            breach_val = compute_breach(breach_inputs)
            nlr_val = compute_nlr(breach_inputs)

            npi_val = compute_npi(NpiInputs(
                tumor_size_mm=tumor_size_mm, tumor_grade=tumor_grade,
                nodes_positive_count=nodes_positive_count,
            ))

            cts5_inputs = Cts5Inputs(
                age_years=age_years, tumor_size_mm=tumor_size_mm, tumor_grade=tumor_grade,
                nodes_positive_count=nodes_positive_count, menopausal_status=menopausal_status,
            )
            is_eligible = cts5_eligible(cts5_inputs)
            cts5_val = compute_cts5(cts5_inputs) if is_eligible else None

        except ValueError as e:
            st.error(f"Input error: {e}")
        else:
            # Stored for testability (Step 8's AppTest reads these
            # directly, instead of parsing rendered widget text).
            st.session_state["last_breach"] = breach_val
            st.session_state["last_npi"] = npi_val
            st.session_state["last_cts5"] = cts5_val
            st.session_state["last_cts5_eligible"] = is_eligible

            st.subheader("Results")
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("BREACH", f"{breach_val:.3f}")
                st.caption(f"Status: {BREACH_STATUS}")
                st.caption(f"NLR used: {nlr_val:.2f}")
            with r2:
                st.metric("NPI", f"{npi_val:.3f}")
                st.caption(f"Status: {NPI_STATUS}")
            with r3:
                if is_eligible:
                    st.metric("CTS5", f"{cts5_val:.3f}")
                    st.caption(f"Status: {CTS5_STATUS}")
                else:
                    st.metric("CTS5", "N/A")
                    st.warning("Not eligible: postmenopausal only.")

            if not is_eligible:
                st.info(
                    "CTS5 is not shown because this patient falls outside its "
                    "validated population (postmenopausal). Computing it anyway "
                    "would be an unsupported extrapolation."
                )

            st.caption("BREACH is an experimental, unvalidated score. Do not use for clinical decisions.")

with tab_cohort:
    st.header("Cohort Validation")
    st.warning(
        "Upload de-identified data only. Files are processed in memory "
        "only, never saved to disk, and are discarded when this session ends."
    )

    col_a, col_b = st.columns(2)
    load_example = col_a.button("Load example dataset (synthetic)", key="load_example")
    uploaded_file = col_b.file_uploader("Or upload your own CSV", type="csv", key="cohort_upload")

    cohort_df = None
    if load_example:
        cohort_df = pd.read_csv("data/example_cohort.csv")
        st.info("Loaded the synthetic example dataset -- for exploring the tool, not real evidence about BREACH.")
    elif uploaded_file is not None:
        cohort_df = pd.read_csv(uploaded_file)

    if cohort_df is not None:
        report = validate_cohort(cohort_df)

        st.subheader("Validation Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total rows", report.n_total)
        c2.metric("Valid rows", report.n_valid)
        c3.metric("Invalid rows", report.n_invalid)

        with st.expander("Missingness per column"):
            st.dataframe(pd.Series(report.missingness, name="% missing").sort_values(ascending=False))

        if report.row_errors:
            with st.expander(f"Row-level errors ({len(report.row_errors)} rows excluded)"):
                for idx, errs in list(report.row_errors.items())[:20]:
                    st.text(f"Row {idx}: {'; '.join(errs)}")
                if len(report.row_errors) > 20:
                    st.caption(f"...and {len(report.row_errors) - 20} more")

        if report.n_valid < 10:
            st.error("Fewer than 10 valid patients -- not enough for a meaningful discrimination analysis.")
        else:
            scored_df = build_scored_cohort(report.valid_records)
            st.info(f"{len(scored_df)} of {report.n_valid} valid patients have outcome data "
                    f"and are included in the analysis below.")

            st.subheader("Discrimination: C-index per score")
            c_breach = compute_c_index(scored_df, "breach")
            c_npi = compute_c_index(scored_df, "npi")
            eligible_df = scored_df[scored_df["cts5_eligible"]]
            n_eligible = len(eligible_df)

            st.session_state["cohort_c_breach"] = c_breach
            st.session_state["cohort_c_npi"] = c_npi
            st.session_state["cohort_n_eligible_cts5"] = n_eligible

            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("BREACH C-index", f"{c_breach:.3f}", help=f"n={len(scored_df)}")
            cc2.metric("NPI C-index", f"{c_npi:.3f}", help=f"n={len(scored_df)}")
            if n_eligible >= 10:
                c_cts5 = compute_c_index(eligible_df, "cts5")
                st.session_state["cohort_c_cts5"] = c_cts5
                cc3.metric("CTS5 C-index", f"{c_cts5:.3f}", help=f"n={n_eligible} eligible (postmenopausal) patients")
            else:
                cc3.metric("CTS5 C-index", "N/A", help=f"Only {n_eligible} eligible patients -- need >=10")
            st.caption("BREACH and NPI are compared on the full cohort. CTS5 uses only the postmenopausal-eligible subset -- these are NOT the same N.")

            st.subheader("Head-to-head comparison (paired bootstrap, 1000 resamples)")
            with st.spinner("Running bootstrap comparison (1000 resamples)..."):
                r_npi = bootstrap_compare_c_index(scored_df, "breach", "npi")
            st.write(f"**BREACH vs NPI** — Δ = {r_npi['delta']:.3f}, "
                     f"95% CI [{r_npi['delta_ci_95'][0]:.3f}, {r_npi['delta_ci_95'][1]:.3f}], "
                     f"p = {r_npi['p_value']:.4f}, n = {r_npi['n_patients']}")

            if n_eligible >= 10:
                with st.spinner("Running bootstrap comparison for CTS5 subset..."):
                    r_cts5 = bootstrap_compare_c_index(eligible_df, "breach", "cts5")
                st.write(f"**BREACH vs CTS5** — Δ = {r_cts5['delta']:.3f}, "
                         f"95% CI [{r_cts5['delta_ci_95'][0]:.3f}, {r_cts5['delta_ci_95'][1]:.3f}], "
                         f"p = {r_cts5['p_value']:.4f}, n = {r_cts5['n_patients']} (eligible subset only)")

            st.subheader("Kaplan-Meier stratification (BREACH tertiles)")
            strat_df, cutoffs = stratify_into_tertiles(scored_df, "breach")
            logrank = km_logrank_test(strat_df, "risk_group")
            st.write(f"Log-rank test across Low/Mid/High BREACH tertiles: p = {logrank['p_value']:.4f}")
            fig = plot_km_by_group(strat_df, "risk_group")
            st.pyplot(fig)