⚠️ **This package has been merged into [`insurance-severity`](https://github.com/burning-cost/insurance-severity).** This repository is archived. Install `insurance-severity` instead.

# insurance-drn

**Distributional Refinement Network for insurance pricing.**

GLMs are good at the mean. The problem is that the mean isn't enough — Solvency II SCR needs the 99.5th percentile, FCA governance needs distributional fairness checks, and reinsurance pricing needs the full tail. You can bolt a parametric assumption (Gamma, LogNormal) onto your GLM, but the parametric shape is often wrong. This library fixes that.

`insurance-drn` takes your fitted GLM and uses a small neural network to refine its distributional predictions into a proper non-parametric histogram distribution. The neural network learns what the GLM gets systematically wrong about shape. The GLM baseline is frozen — the DRN only adjusts it. If the network has nothing to add, you get the GLM back. If it has something to add, you get better calibration.

Based on Avanzi, Dong, Laub, Wong (2024), *Distributional Refinement Network: Distributional Forecasting via Deep Learning.* arXiv:2406.00998.

---

## The problem this solves

Standard actuarial GLM workflow for motor severity:

1. Fit Gamma GLM with log link
2. Get predicted means E[Y | X]
3. Assume Gamma shape parameter is constant across all risks
4. Use Gamma distribution for VaR / SCR calculations

Step 3 is almost always wrong. Shape varies by risk group — young drivers have different severity distributions than mature drivers, not just different means. The DRN learns these distributional differences from data without requiring you to model them explicitly.

---

## Quick start

```python
import statsmodels.formula.api as smf
import statsmodels.api as sm
from insurance_drn import GLMBaseline, DRN

# Step 1: fit your GLM as normal
glm = smf.glm(
    "claim_cost ~ vehicle_age + C(region) + driver_age",
    data=df_train,
    family=sm.families.Gamma(sm.families.links.Log()),
).fit()

# Step 2: wrap it
baseline = GLMBaseline(glm)

# Step 3: fit DRN
drn = DRN(
    baseline,
    hidden_size=64,
    max_epochs=200,
    patience=20,
    kl_alpha=1e-4,    # don't stray too far from baseline
    mean_alpha=1e-4,  # keep predicted means aligned with GLM
    scr_aware=True,   # set c_K above 99.7th percentile for SCR
)
drn.fit(X_train, y_train)

# Step 4: full predictive distributions
dist = drn.predict_distribution(X_test)
print(dist.mean())           # (n,) — expected claims per policy
print(dist.quantile(0.995))  # (n,) — 99.5th VaR for Solvency II SCR
print(dist.expected_shortfall(0.995))  # (n,) — TVaR
```

---

## What you get

### `ExtendedHistogramBatch`

The predictive distribution for all n test observations, vectorised:

```python
dist = drn.predict_distribution(X_test)
dist.mean()                    # (n,)
dist.var()                     # (n,)
dist.std()                     # (n,)
dist.quantile(0.995)           # (n,) — VaR
dist.quantile([0.5, 0.75, 0.99])  # (n, 3)
dist.expected_shortfall(0.995) # (n,) — CVaR / TVaR
dist.crps(y_true)              # (n,) — CRPS for model comparison
dist.cdf(threshold)            # (n,) — CDF at a threshold
dist.summary()                 # polars DataFrame with standard quantiles
```

### Interpretability

```python
# Per-bin adjustment factors: where does DRN differ from GLM?
af = drn.adjustment_factors(X_test)
# polars DataFrame, shape (n, K)
# Values > 1: DRN assigns more probability than GLM baseline
# Values < 1: DRN assigns less probability than GLM baseline
```

### Calibration diagnostics

```python
from insurance_drn import DRNDiagnostics
diag = DRNDiagnostics(drn)

diag.pit_histogram(X_test, y_test)           # matplotlib Figure
diag.quantile_calibration(X_test, y_test)    # polars DataFrame
diag.crps_by_segment(X_test, y_test, "region")  # polars DataFrame
diag.summary(X_test, y_test)                 # one-row summary
```

---

## Architecture

The DRN is a three-part system:

1. **Frozen baseline** (your GLM): provides parametric bin probabilities b_k(x) for each histogram bin and observation.

2. **Neural network**: a standard feedforward MLP that outputs K log-adjustment values delta_k(x).

3. **Refinement equation**: `p_k(x) = softmax(log(b_k(x)) + delta_k(x))`

When all delta_k = 0 (at initialisation with `baseline_start=True`), the DRN recovers the GLM exactly. Training teaches delta_k to capture what the GLM misses.

The output is an **extended histogram distribution**: piecewise uniform over K bins between cutpoints [c_0, c_K], with the baseline GLM distribution governing the tails outside this range. For Solvency II, set `scr_aware=True` so c_K exceeds the 99.7th percentile and the 99.5th VaR falls in the histogram region where the DRN has full control.

**Loss**: JBCE (Joint Binary Cross-Entropy). For each cutpoint c_k, treat the problem as binary classification: does y fall below c_k? Average BCE over all cutpoints and observations. More stable than NLL for histogram models because it evaluates the CDF, not the PDF.

---

## Supported baselines

| Family | GLMBaseline | CatBoostBaseline |
|--------|------------|-----------------|
| Gamma | Yes | Yes |
| Gaussian | Yes | Yes |
| LogNormal | Yes | Yes |
| InverseGaussian | Yes | Yes |

```python
from insurance_drn import GLMBaseline, CatBoostBaseline

# Wrap a fitted statsmodels GLM
baseline_glm = GLMBaseline(glm_result)

# Wrap a CatBoost mean predictor with manual dispersion
baseline_cb = CatBoostBaseline(catboost_model, family="gamma")
baseline_cb.fit_dispersion(y_train, X_train)  # estimate phi from residuals

# Use insurance-distributional for full CDF predictions
from insurance_distributional import CatBoostDistributional
dist_model = CatBoostDistributional(...)
dist_model.fit(X_train, y_train)
baseline_dist = CatBoostBaseline(dist_model)  # delegates to dist_model.predict_cdf()
```

---

## Installation

```bash
pip install insurance-drn
pip install insurance-drn[glm]     # + statsmodels for GLMBaseline.from_formula()
pip install insurance-drn[plots]   # + matplotlib for diagnostic plots
pip install insurance-drn[all]     # everything
```

Requires Python 3.10+, PyTorch 2.0+.

---

## Regularisation

Training without constraints can cause the DRN to move the mean away from the GLM — which breaks actuarial pricing consistency. Use these to control it:

| Parameter | Typical value | Effect |
|-----------|--------------|--------|
| `kl_alpha` | 1e-4 | Penalise deviation from baseline shape |
| `mean_alpha` | 1e-4 | Force DRN mean close to GLM mean |
| `tv_alpha` | 0 | Penalise jagged density (rarely needed) |
| `dv_alpha` | 1e-3 | Penalise non-smooth curvature |

Start with `kl_alpha=1e-4, mean_alpha=1e-4`. If the GLM is well-calibrated for the mean, `mean_alpha` is the most important constraint.

---

## Saving and loading

```python
drn.save("models/motor_severity_drn.pt")

# Load (baseline must be provided separately)
drn2 = DRN.load("models/motor_severity_drn.pt", baseline=baseline)
```

The baseline model is not serialised — only the network weights and cutpoints are saved. You must provide the same fitted baseline on load.

---

## SCR application

```python
drn = DRN(baseline, scr_aware=True, max_epochs=500)
drn.fit(X_train, y_train)

# Solvency II VaR at 99.5th percentile
var_995 = drn.predict_quantile(X_portfolio, 0.995)

# Expected Shortfall (TVaR) — for internal model use
tvar_995 = drn.predict_distribution(X_portfolio).expected_shortfall(0.995)
```

With `scr_aware=True`, the upper cutpoint c_K is set to 1.1x the 99.7th percentile of training claims. This means the 99.5th VaR falls inside the histogram region where the neural network has full distributional control.

---

## Notes on zero-inflation

DRN cannot handle a point mass at zero within the histogram. For UK personal lines datasets with no-claim records:

1. **Recommended**: model only positive claims (use `y > 0` training data). Combine with a separate frequency model (Poisson/negative binomial GLM).
2. **Alternative**: set `c_0` slightly below zero and let the baseline handle zero mass — but this means the DRN cannot adjust the zero-mass probability.

This is a fundamental limitation of histogram-based distributional models. Document it in your model validation.
