"""
simulate.py — generates synthetic A/B test data with realistic quirks:
novelty effects, sample ratio mismatch, and configurable true effect size.

This exists so the rest of the toolkit (power analysis, hypothesis testing,
DiD) has something honest to run against — we know the ground-truth effect
we injected, so we can verify our own statistical methods recover it.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    n_control: int = 5000
    n_treatment: int = 5000
    baseline_conversion: float = 0.12       # control group true conversion rate
    true_lift: float = 0.015                # absolute lift injected into treatment (e.g. +1.5pp)
    novelty_decay_days: int | None = None    # if set, treatment effect decays over N days (novelty effect)
    days: int = 14
    sample_ratio_mismatch: float | None = None  # if set (e.g. 0.02), injects an SRM bug
    seed: int = 42


def simulate_experiment(cfg: ExperimentConfig = ExperimentConfig()) -> pd.DataFrame:
    """
    Returns a user-level DataFrame: user_id, variant, day, converted (0/1).
    Ground truth: control converts at cfg.baseline_conversion; treatment converts
    at baseline + true_lift (optionally decaying over time to simulate novelty effect).
    """
    rng = np.random.default_rng(cfg.seed)

    n_control, n_treatment = cfg.n_control, cfg.n_treatment
    if cfg.sample_ratio_mismatch:
        # simulate a real-world SRM bug: treatment group is under/over-assigned
        n_treatment = int(n_treatment * (1 - cfg.sample_ratio_mismatch))

    rows = []
    for day in range(cfg.days):
        # decaying novelty effect: lift shrinks toward 0 as days increase
        if cfg.novelty_decay_days:
            decay = max(0.0, 1 - day / cfg.novelty_decay_days)
        else:
            decay = 1.0

        daily_n_c = n_control // cfg.days
        daily_n_t = n_treatment // cfg.days

        control_conv = rng.binomial(1, cfg.baseline_conversion, daily_n_c)
        treat_conv = rng.binomial(
            1, min(0.999, cfg.baseline_conversion + cfg.true_lift * decay), daily_n_t
        )

        for i, c in enumerate(control_conv):
            rows.append({"user_id": f"c_{day}_{i}", "variant": "control", "day": day, "converted": int(c)})
        for i, t in enumerate(treat_conv):
            rows.append({"user_id": f"t_{day}_{i}", "variant": "treatment", "day": day, "converted": int(t)})

    return pd.DataFrame(rows)


def simulate_panel_for_did(
    n_units: int = 40,
    n_periods: int = 12,
    treated_from_period: int = 6,
    true_effect: float = 3.5,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Panel data for difference-in-differences: half the units are "treated"
    starting at treated_from_period, the rest are controls throughout.
    Mirrors the structure used in the NUTS-2 regional thesis analysis —
    unit fixed effects + time fixed effects + a treatment*post interaction.
    """
    rng = np.random.default_rng(seed)
    unit_fe = rng.normal(50, 8, n_units)          # unit-level baseline (region fixed effect)
    time_fe = np.linspace(0, 5, n_periods)          # common time trend
    treated_units = set(range(n_units // 2))         # first half = treated group

    rows = []
    for unit in range(n_units):
        for t in range(n_periods):
            is_treated_unit = unit in treated_units
            is_post = t >= treated_from_period
            treatment_effect = true_effect if (is_treated_unit and is_post) else 0.0
            noise = rng.normal(0, 2.5)
            y = unit_fe[unit] + time_fe[t] + treatment_effect + noise
            rows.append({
                "unit": unit, "period": t, "y": y,
                "treated_unit": int(is_treated_unit), "post": int(is_post),
            })
    return pd.DataFrame(rows)
