"""
did.py — Difference-in-Differences estimator.

This is the quasi-experimental method used when you *can't* randomize
(e.g. a feature rolled out by region, not by random assignment — exactly
the structure of the "Tourism Recovery Asymmetry" analysis in my thesis,
applied here to a generic product/regional panel instead of NUTS-2 regions).

Google's 2026 experimentation rounds explicitly test DiD, geo-randomized
trials, and synthetic control as alternatives to standard A/B testing —
this module is that.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


@dataclass
class DiDResult:
    att: float                    # average treatment effect on the treated
    std_error: float
    p_value: float
    ci_95_low: float
    ci_95_high: float
    n_obs: int
    n_units: int
    n_periods: int
    model_summary: str

    def summary(self) -> str:
        sig = "significant" if self.p_value < 0.05 else "not significant"
        return (
            f"DiD estimate (ATT): {self.att:+.3f}  (SE={self.std_error:.3f})\n"
            f"95% CI: [{self.ci_95_low:+.3f}, {self.ci_95_high:+.3f}]\n"
            f"p = {self.p_value:.4f}  →  {sig} at alpha=0.05\n"
            f"Panel: {self.n_units} units × {self.n_periods} periods "
            f"({self.n_obs} observations)"
        )


def estimate_did(df: pd.DataFrame, y_col: str = "y") -> DiDResult:
    """
    Two-way fixed effects DiD:

        y_it = alpha_i + lambda_t + beta * (treated_i * post_t) + eps_it

    alpha_i: unit fixed effects (absorbs any time-invariant unit differences)
    lambda_t: time fixed effects (absorbs any shock common to all units)
    beta: the DiD estimate — the causal effect of treatment on the treated,
          under the parallel-trends assumption.

    df must have columns: unit, period, treated_unit, post, y
    """
    data = df.copy()
    data["interaction"] = data["treated_unit"] * data["post"]

    # unit and period fixed effects via categorical dummies (standard DiD spec)
    formula = f"{y_col} ~ interaction + C(unit) + C(period)"
    model = smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["unit"]}
    )  # cluster SEs at the unit level — standard practice, avoids overstating precision

    beta = model.params["interaction"]
    se = model.bse["interaction"]
    p = model.pvalues["interaction"]
    ci = model.conf_int().loc["interaction"]

    return DiDResult(
        att=float(beta),
        std_error=float(se),
        p_value=float(p),
        ci_95_low=float(ci[0]),
        ci_95_high=float(ci[1]),
        n_obs=len(data),
        n_units=data["unit"].nunique(),
        n_periods=data["period"].nunique(),
        model_summary=str(model.summary()),
    )


def parallel_trends_check(df: pd.DataFrame, y_col: str = "y") -> pd.DataFrame:
    """
    Pre-treatment trend check — the key identifying assumption of DiD.
    Returns average outcome by group and period, pre-treatment only,
    so you can visually/numerically confirm treated and control units
    were moving in parallel *before* treatment started.
    """
    pre = df[df.post == 0]
    return (
        pre.groupby(["period", "treated_unit"])[y_col]
        .mean()
        .reset_index()
        .pivot(index="period", columns="treated_unit", values=y_col)
        .rename(columns={0: "control_mean", 1: "treated_mean"})
    )
