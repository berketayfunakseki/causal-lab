"""
power.py — sample size / statistical power calculations.

Answers the question every DS is asked before running a test:
"How many users do I need to reliably detect this effect?"
"""
from __future__ import annotations
import math
from scipy import stats


def required_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """
    Sample size per group for a two-proportion z-test.

    baseline_rate: control group conversion rate (e.g. 0.12)
    minimum_detectable_effect: smallest absolute lift worth detecting (e.g. 0.015 = 1.5pp)
    alpha: significance level (Type I error rate)
    power: desired statistical power (1 - Type II error rate)
    """
    p1 = baseline_rate
    p2 = baseline_rate + minimum_detectable_effect
    p_bar = (p1 + p2) / 2

    z_alpha = stats.norm.ppf(1 - alpha / 2)   # two-sided
    z_beta = stats.norm.ppf(power)

    numerator = (z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) +
                 z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    denominator = (p2 - p1) ** 2

    n = numerator / denominator
    return math.ceil(n)


def minimum_detectable_effect(
    baseline_rate: float,
    n_per_group: int,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """
    Inverse problem: given a fixed sample size (e.g. limited traffic),
    what's the smallest effect we could reliably detect?
    Solved via binary search since there's no closed form.
    """
    lo, hi = 0.0001, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        n_needed = required_sample_size(baseline_rate, mid, alpha, power)
        if n_needed > n_per_group:
            lo = mid   # need a bigger effect to detect with this sample size
        else:
            hi = mid
    return hi


def estimate_test_duration_days(
    required_n_per_group: int,
    daily_traffic_per_group: int,
) -> int:
    """How many days to reach the required sample size, given daily traffic."""
    if daily_traffic_per_group <= 0:
        raise ValueError("daily_traffic_per_group must be positive")
    return math.ceil(required_n_per_group / daily_traffic_per_group)
