# Causal Lab — A/B Testing & Difference-in-Differences Toolkit

A statistics-first project built to practice the exact methods used in experimentation
and causal-inference interviews: hypothesis testing mechanics (not just "which test to run"),
power analysis, sample-ratio-mismatch detection, and difference-in-differences for when
randomization isn't possible.

Try it live in the [Streamlit dashboard](#running-locally) — simulate an experiment with a
known ground-truth effect and watch the statistics recover it.

## Why this exists

Most "data science portfolio" projects are prediction models. This one is different on purpose:
it's about **causal inference and experimental design** — the parts of the DS toolkit that
prediction-focused projects skip. It's the same estimator family as my bachelor's thesis
(difference-in-differences, event-study panel analysis on 21 Italian NUTS-2 regions), applied
here to a generic experimentation setting.

## What's inside

| Module | What it does |
|---|---|
| `src/simulate.py` | Generates synthetic experiment data with configurable effect size, novelty-effect decay, and sample-ratio-mismatch bugs |
| `src/power.py` | Sample size / minimum-detectable-effect calculations |
| `src/ab_test.py` | Two-proportion z-test with confidence intervals, plus a chi-squared SRM (Sample Ratio Mismatch) check |
| `src/did.py` | Two-way fixed-effects difference-in-differences estimator with clustered standard errors |
| `src/api.py` | FastAPI service exposing all of the above as REST endpoints |
| `dashboard.py` | Interactive Streamlit app — configure and run experiments live |

All statistical claims are backed by tests in `tests/test_causal_lab.py` that verify the
methods recover a **known, injected ground-truth effect** from simulated data — not just that
the code runs without errors.

## Running locally

```bash
pip install -r requirements.txt

# interactive dashboard
streamlit run dashboard.py

# or the API
uvicorn src.api:app --reload
# → http://127.0.0.1:8000/docs
```

## Running with Docker

```bash
docker build -t causal-lab .
docker run -p 8000:8000 causal-lab
```

## Example: does my sample size actually work?

```python
from src.power import required_sample_size
from src.simulate import ExperimentConfig, simulate_experiment
from src.ab_test import run_from_dataframe

n = required_sample_size(baseline_rate=0.12, minimum_detectable_effect=0.015)
print(f"Need {n} users per group")

cfg = ExperimentConfig(n_control=n, n_treatment=n, baseline_conversion=0.12, true_lift=0.015, days=1)
df = simulate_experiment(cfg)
result = run_from_dataframe(df)
print(result.summary())
```

## Why these choices

- **Two-proportion z-test over a generic t-test** — conversion data is Bernoulli, not
  continuous; the z-test is the correctly-specified test for proportions and gives an
  exact closed-form variance rather than an approximation.
- **Cluster-robust standard errors in the DiD model** — without clustering at the unit level,
  standard errors are understated whenever outcomes are correlated within a unit over time
  (they almost always are), which overstates significance. This mirrors a mistake that's easy
  to make and a strong Google interview signal to *not* make.
- **A dedicated SRM check, run automatically before trusting any test result** — a real
  production practice: if your randomization is broken, your p-value is meaningless no matter
  how small it is.

## Lessons learned

- **statsmodels emits a rank-deficiency warning** on the fixed-effects DiD spec when the panel
  is small relative to the number of unit dummies — this is a known quirk of dummy-variable
  fixed effects with clustered SEs, not a bug in the estimator. A production version would use
  a proper within-transformation (demeaning) instead of dummy variables to avoid it, e.g. via
  the `linearmodels` package's `PanelOLS`.
- **Simulating the null hypothesis and checking the false-positive rate** was more useful for
  catching bugs than testing the alternative hypothesis alone — it's easy to write a test that
  correctly detects a real effect but silently violates alpha under the null.

## What I'd improve next

- Sequential testing correction (to handle "peeking" at results before the test ends)
- Bayesian A/B testing as an alternative framework
- `linearmodels.PanelOLS` for the DiD estimator instead of dummy-variable OLS

---
Berke Tayfun Akseki — [berketayfunakseki.com](https://berketayfunakseki.com)
