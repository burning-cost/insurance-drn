"""
insurance-drn: Distributional Refinement Network for insurance pricing.

Takes a frozen GLM or GBM baseline distribution and refines it into a full
predictive distribution using a neural network. Based on Avanzi, Dong, Laub,
Wong (2024), arXiv:2406.00998.

Key classes:
    BaselineDistribution  — Protocol for baseline models
    GLMBaseline           — Wraps statsmodels GLM results
    CatBoostBaseline      — Wraps insurance-distributional / CatBoost models
    DRN                   — Main model: fit, predict_distribution, etc.
    ExtendedHistogramBatch — Vectorised predictive distributions for n obs
    DRNDiagnostics        — Calibration plots (PIT, coverage, CRPS segments)

Quick start::

    from insurance_drn import GLMBaseline, DRN
    import statsmodels.formula.api as smf

    glm = smf.glm("claims ~ age + vehicle_age", data=df,
                  family=sm.families.Gamma(sm.families.links.Log())).fit()
    baseline = GLMBaseline(glm)
    drn = DRN(baseline, hidden_size=64, num_hidden_layers=2)
    drn.fit(X_train, y_train)
    dist = drn.predict_distribution(X_test)
    print(dist.mean())           # numpy array, shape (n,)
    print(dist.quantile(0.995))  # 99.5th percentile for SCR
"""

from insurance_drn.baseline import BaselineDistribution, GLMBaseline
from insurance_drn.catboost_baseline import CatBoostBaseline
from insurance_drn.histogram import ExtendedHistogramBatch
from insurance_drn.cutpoints import drn_cutpoints
from insurance_drn.loss import jbce_loss, drn_regularisation
from insurance_drn.network import DRNNetwork
from insurance_drn.drn import DRN
from insurance_drn.diagnostics import DRNDiagnostics

__version__ = "0.1.0"
__all__ = [
    "BaselineDistribution",
    "GLMBaseline",
    "CatBoostBaseline",
    "ExtendedHistogramBatch",
    "DRNNetwork",
    "DRN",
    "DRNDiagnostics",
    "drn_cutpoints",
    "jbce_loss",
    "drn_regularisation",
]
