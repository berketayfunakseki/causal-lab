"""
dashboard.py — interactive Streamlit app.
Lets anyone (recruiter included) run a full A/B test end-to-end:
configure an experiment, see the data, see the test, see the verdict.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.simulate import ExperimentConfig, simulate_experiment, simulate_panel_for_did
from src.power import required_sample_size, estimate_test_duration_days
from src.ab_test import run_from_dataframe, check_sample_ratio_mismatch
from src.did import estimate_did, parallel_trends_check

st.set_page_config(page_title="Causal Lab", layout="wide")
st.title("🧪 Causal Lab")
st.caption("A/B testing, power analysis, and difference-in-differences — built to practice the exact methods Google's experimentation interview round tests.")

tab1, tab2, tab3 = st.tabs(["📐 Power Analysis", "🔬 A/B Test Simulator", "📊 Difference-in-Differences"])

with tab1:
    st.subheader("How many users do I need?")
    col1, col2 = st.columns(2)
    with col1:
        baseline = st.slider("Baseline conversion rate", 0.01, 0.5, 0.12, 0.01)
        mde = st.slider("Minimum detectable effect (absolute)", 0.001, 0.1, 0.015, 0.001)
    with col2:
        alpha = st.select_slider("Significance level (alpha)", [0.01, 0.05, 0.1], value=0.05)
        power = st.select_slider("Statistical power", [0.7, 0.8, 0.9, 0.95], value=0.8)

    n = required_sample_size(baseline, mde, alpha, power)
    st.metric("Required sample size per group", f"{n:,}")

    daily_traffic = st.number_input("Daily traffic per group (optional)", min_value=0, value=500)
    if daily_traffic > 0:
        days = estimate_test_duration_days(n, daily_traffic)
        st.info(f"At {daily_traffic:,} users/day/group, this test needs **{days} days** to reach significance-detecting power.")

with tab2:
    st.subheader("Simulate an experiment, then test it")
    col1, col2, col3 = st.columns(3)
    with col1:
        n_per_group = st.number_input("Users per group", 500, 50000, 5000, step=500)
        baseline_rate = st.slider("True baseline rate", 0.01, 0.5, 0.12, key="t2_baseline")
    with col2:
        true_lift = st.slider("True injected lift", -0.05, 0.05, 0.015, 0.001, key="t2_lift")
        novelty = st.checkbox("Simulate novelty effect (lift decays over time)")
    with col3:
        srm_bug = st.checkbox("Inject a sample ratio mismatch bug")
        srm_pct = st.slider("SRM severity", 0.0, 0.1, 0.03) if srm_bug else None

    if st.button("Run experiment", type="primary"):
        cfg = ExperimentConfig(
            n_control=n_per_group, n_treatment=n_per_group,
            baseline_conversion=baseline_rate, true_lift=true_lift,
            novelty_decay_days=10 if novelty else None,
            sample_ratio_mismatch=srm_pct,
        )
        df = simulate_experiment(cfg)

        srm_result = check_sample_ratio_mismatch(df)
        if srm_result["srm_detected"]:
            st.error(
                f"⚠️ Sample Ratio Mismatch detected! "
                f"Observed split {srm_result['observed_ratio']:.1%} vs expected 50%. "
                f"**Do not trust the test result below until this is fixed.**"
            )
        else:
            st.success("✅ No sample ratio mismatch — randomization looks healthy.")

        result = run_from_dataframe(df)
        c1, c2, c3 = st.columns(3)
        c1.metric("Control rate", f"{result.control_rate:.2%}")
        c2.metric("Treatment rate", f"{result.treatment_rate:.2%}", f"{result.absolute_lift:+.2%}")
        c3.metric("p-value", f"{result.p_value:.4f}", "significant" if result.significant_at_05 else "not significant")

        st.text(result.summary())

        daily = df.groupby(["day", "variant"])["converted"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(9, 3.5))
        for variant, grp in daily.groupby("variant"):
            ax.plot(grp["day"], grp["converted"], marker="o", label=variant)
        ax.set_xlabel("Day"); ax.set_ylabel("Conversion rate"); ax.legend()
        ax.set_title("Daily conversion rate by variant" + (" (novelty decay visible)" if novelty else ""))
        st.pyplot(fig)

with tab3:
    st.subheader("Difference-in-Differences — for when you can't randomize")
    st.caption("Same estimator family as my thesis (regional panel, DiD, event-study) — applied here to a generic treated/control panel.")

    col1, col2 = st.columns(2)
    with col1:
        n_units = st.slider("Number of units (e.g. regions/stores)", 10, 100, 40)
        n_periods = st.slider("Number of time periods", 4, 24, 12)
    with col2:
        treated_from = st.slider("Treatment starts at period", 1, 20, 6)
        true_effect = st.slider("True treatment effect", -10.0, 10.0, 3.5)

    if st.button("Run DiD", type="primary"):
        panel = simulate_panel_for_did(n_units, n_periods, treated_from, true_effect)
        result = estimate_did(panel)

        c1, c2 = st.columns(2)
        c1.metric("Estimated effect (ATT)", f"{result.att:+.3f}", f"true={true_effect:+.2f}")
        c2.metric("p-value", f"{result.p_value:.4f}")
        st.text(result.summary())

        trends = parallel_trends_check(panel)
        fig, ax = plt.subplots(figsize=(9, 3.5))
        trends.plot(ax=ax, marker="o")
        ax.axvline(treated_from - 1, color="gray", linestyle="--", label="treatment starts")
        ax.set_title("Pre-treatment parallel trends check")
        ax.set_xlabel("Period"); ax.set_ylabel("Mean outcome")
        st.pyplot(fig)
        st.caption("If control and treated lines move roughly in parallel *before* the dashed line, the parallel-trends assumption holds and the DiD estimate above is credible.")

st.divider()
st.caption("Berke Tayfun Akseki — [berketayfunakseki.com](https://berketayfunakseki.com)")
