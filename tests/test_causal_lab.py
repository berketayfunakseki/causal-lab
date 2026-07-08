"""
Tests verify the statistical machinery actually recovers known ground-truth
effects from simulated data — not just that the code runs without crashing.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulate import ExperimentConfig, simulate_experiment, simulate_panel_for_did
from src.power import required_sample_size, minimum_detectable_effect
from src.ab_test import run_from_dataframe, two_proportion_z_test, check_sample_ratio_mismatch
from src.did import estimate_did


def test_z_test_detects_known_lift():
    """With a large sample and a real 3pp lift, the test should find significance."""
    cfg = ExperimentConfig(n_control=20000, n_treatment=20000, baseline_conversion=0.10, true_lift=0.03, days=1)
    df = simulate_experiment(cfg)
    result = run_from_dataframe(df)
    assert result.significant_at_05, "should detect a large, well-powered effect"
    assert result.absolute_lift > 0.015, "recovered lift should be in the right ballpark"


def test_z_test_no_false_positive_on_null():
    """With zero true effect, most runs should NOT reject the null (sanity check on Type I error)."""
    false_positives = 0
    for seed in range(20):
        cfg = ExperimentConfig(n_control=3000, n_treatment=3000, baseline_conversion=0.10, true_lift=0.0, days=1, seed=seed)
        df = simulate_experiment(cfg)
        result = run_from_dataframe(df)
        if result.significant_at_05:
            false_positives += 1
    # at alpha=0.05 we expect ~5% false positive rate; allow generous margin for a 20-run sample
    assert false_positives <= 4, f"too many false positives under the null: {false_positives}/20"


def test_sample_size_increases_for_smaller_effect():
    n_large_effect = required_sample_size(0.10, 0.05)
    n_small_effect = required_sample_size(0.10, 0.01)
    assert n_small_effect > n_large_effect, "detecting a smaller effect requires more samples"


def test_srm_detects_injected_mismatch():
    cfg = ExperimentConfig(n_control=10000, n_treatment=10000, sample_ratio_mismatch=0.15, days=1)
    df = simulate_experiment(cfg)
    srm = check_sample_ratio_mismatch(df)
    assert srm["srm_detected"], "a 15% injected mismatch should be caught"


def test_srm_clean_when_balanced():
    cfg = ExperimentConfig(n_control=10000, n_treatment=10000, days=1)
    df = simulate_experiment(cfg)
    srm = check_sample_ratio_mismatch(df)
    assert not srm["srm_detected"], "a balanced 50/50 split should not trigger SRM"


def test_did_recovers_known_treatment_effect():
    panel = simulate_panel_for_did(n_units=60, n_periods=12, treated_from_period=6, true_effect=4.0, seed=1)
    result = estimate_did(panel)
    assert abs(result.att - 4.0) < 1.0, f"DiD estimate {result.att:.2f} too far from true effect 4.0"
    assert result.p_value < 0.05, "a true effect of this size with this much data should be significant"


def test_did_null_effect_not_significant():
    panel = simulate_panel_for_did(n_units=60, n_periods=12, treated_from_period=6, true_effect=0.0, seed=2)
    result = estimate_did(panel)
    assert result.p_value > 0.05, "with zero true effect, DiD should not find significance"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
