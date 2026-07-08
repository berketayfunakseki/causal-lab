"""
api.py — FastAPI service wrapping the toolkit.
Consistent with the rest of my stack (Jobscope, Regulatory RAG): every
project ships as a real, runnable API — not just a notebook.
"""
from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .simulate import ExperimentConfig, simulate_experiment, simulate_panel_for_did
from .power import required_sample_size, minimum_detectable_effect, estimate_test_duration_days
from .ab_test import two_proportion_z_test, run_from_dataframe, check_sample_ratio_mismatch
from .did import estimate_did

app = FastAPI(
    title="Causal Lab API",
    description="A/B testing, power analysis, and difference-in-differences as a service.",
    version="1.0.0",
)


class PowerRequest(BaseModel):
    baseline_rate: float = Field(..., gt=0, lt=1, example=0.12)
    minimum_detectable_effect: float = Field(..., gt=0, lt=1, example=0.015)
    alpha: float = 0.05
    power: float = 0.8
    daily_traffic_per_group: int | None = None


class ZTestRequest(BaseModel):
    control_conversions: int
    control_n: int
    treatment_conversions: int
    treatment_n: int
    alpha: float = 0.05


@app.get("/")
def root():
    return {"service": "causal-lab", "status": "ok"}


@app.post("/power/sample-size")
def sample_size(req: PowerRequest):
    n = required_sample_size(req.baseline_rate, req.minimum_detectable_effect, req.alpha, req.power)
    resp = {"required_n_per_group": n, "required_n_total": n * 2}
    if req.daily_traffic_per_group:
        resp["estimated_duration_days"] = estimate_test_duration_days(n, req.daily_traffic_per_group)
    return resp


@app.post("/ab-test/z-test")
def z_test(req: ZTestRequest):
    result = two_proportion_z_test(
        req.control_conversions, req.control_n, req.treatment_conversions, req.treatment_n, req.alpha
    )
    return result.__dict__


@app.get("/ab-test/simulate-and-test")
def simulate_and_test(
    n_per_group: int = 5000,
    baseline_rate: float = 0.12,
    true_lift: float = 0.015,
    novelty_decay_days: int | None = None,
    sample_ratio_mismatch: float | None = None,
):
    """
    End-to-end demo: simulate an experiment with a known ground-truth effect,
    then recover it with the z-test — proves the statistical machinery works.
    """
    cfg = ExperimentConfig(
        n_control=n_per_group, n_treatment=n_per_group,
        baseline_conversion=baseline_rate, true_lift=true_lift,
        novelty_decay_days=novelty_decay_days, sample_ratio_mismatch=sample_ratio_mismatch,
    )
    df = simulate_experiment(cfg)
    result = run_from_dataframe(df)
    srm = check_sample_ratio_mismatch(df)
    return {"ground_truth_lift": true_lift, "test_result": result.__dict__, "srm_check": srm}


@app.get("/did/simulate-and-estimate")
def did_simulate_and_estimate(
    n_units: int = 40, n_periods: int = 12, treated_from_period: int = 6, true_effect: float = 3.5
):
    """Simulate a regional panel and recover the injected treatment effect via DiD."""
    df = simulate_panel_for_did(n_units, n_periods, treated_from_period, true_effect)
    result = estimate_did(df)
    return {"ground_truth_effect": true_effect, "did_result": result.__dict__}
