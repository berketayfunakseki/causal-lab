"""
ab_test.py — the actual hypothesis testing layer.

Implements what Google DS interviews explicitly probe: not just "which test
do I run" but the mechanics — can you compute the test statistic by hand,
explain what the p-value means, and interpret a confidence interval correctly.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ABTestResult:
    control_n: int
    treatment_n: int
    control_rate: float
    treatment_rate: float
    absolute_lift: float
    relative_lift_pct: float
    z_statistic: float
    p_value: float
    ci_95_low: float
    ci_95_high: float
    significant_at_05: bool

    def summary(self) -> str:
        sig = "YES" if self.significant_at_05 else "NO"
        return (
            f"Control:   {self.control_rate:.4f}  (n={self.control_n})\n"
            f"Treatment: {self.treatment_rate:.4f}  (n={self.treatment_n})\n"
            f"Absolute lift: {self.absolute_lift:+.4f}  "
            f"({self.relative_lift_pct:+.2f}% relative)\n"
            f"95% CI on lift: [{self.ci_95_low:+.4f}, {self.ci_95_high:+.4f}]\n"
            f"z = {self.z_statistic:.3f},  p = {self.p_value:.4f}\n"
            f"Significant at alpha=0.05: {sig}"
        )


def two_proportion_z_test(
    control_conversions: int,
    control_n: int,
    treatment_conversions: int,
    treatment_n: int,
    alpha: float = 0.05,
) -> ABTestResult:
    """
    Two-proportion z-test — the standard test for conversion-rate experiments.

    H0: p_control == p_treatment
    H1: p_control != p_treatment  (two-sided)
    """
    p1 = control_conversions / control_n
    p2 = treatment_conversions / treatment_n
    p_pool = (control_conversions + treatment_conversions) / (control_n + treatment_n)

    se_pooled = np.sqrt(p_pool * (1 - p_pool) * (1 / control_n + 1 / treatment_n))
    z = (p2 - p1) / se_pooled
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # CI on the *difference* uses unpooled variance (standard practice)
    se_diff = np.sqrt(p1 * (1 - p1) / control_n + p2 * (1 - p2) / treatment_n)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lift = p2 - p1
    ci_low, ci_high = lift - z_crit * se_diff, lift + z_crit * se_diff

    return ABTestResult(
        control_n=control_n,
        treatment_n=treatment_n,
        control_rate=p1,
        treatment_rate=p2,
        absolute_lift=lift,
        relative_lift_pct=(lift / p1 * 100) if p1 > 0 else float("nan"),
        z_statistic=z,
        p_value=p_value,
        ci_95_low=ci_low,
        ci_95_high=ci_high,
        significant_at_05=p_value < alpha,
    )


def run_from_dataframe(df: pd.DataFrame, alpha: float = 0.05) -> ABTestResult:
    """Convenience wrapper: takes the simulate.py output format directly."""
    c = df[df.variant == "control"]
    t = df[df.variant == "treatment"]
    return two_proportion_z_test(
        control_conversions=int(c.converted.sum()),
        control_n=len(c),
        treatment_conversions=int(t.converted.sum()),
        treatment_n=len(t),
        alpha=alpha,
    )


def check_sample_ratio_mismatch(df: pd.DataFrame, expected_ratio: float = 0.5, alpha: float = 0.001) -> dict:
    """
    Sample Ratio Mismatch (SRM) check — a real production data-quality test.
    If group sizes deviate significantly from the expected 50/50 split (or
    whatever ratio was configured), the experiment's randomization is broken
    and results should not be trusted, no matter how significant the p-value.

    This is a chi-squared goodness-of-fit test, checked at a strict alpha
    (0.001, not 0.05) because false alarms here are costly to chase down.
    """
    n_control = (df.variant == "control").sum()
    n_treatment = (df.variant == "treatment").sum()
    total = n_control + n_treatment

    expected_control = total * expected_ratio
    expected_treatment = total * (1 - expected_ratio)

    chi2, p_value = stats.chisquare(
        f_obs=[n_control, n_treatment],
        f_exp=[expected_control, expected_treatment],
    )
    return {
        "n_control": int(n_control),
        "n_treatment": int(n_treatment),
        "observed_ratio": n_control / total,
        "expected_ratio": expected_ratio,
        "chi2_statistic": float(chi2),
        "p_value": float(p_value),
        "srm_detected": p_value < alpha,
    }
