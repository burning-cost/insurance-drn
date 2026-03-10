"""
Shared fixtures for insurance-drn tests.

All data is synthetic — generated from known distributions so we can
test correctness analytically.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def gamma_data(rng):
    """
    Synthetic Gamma severity dataset.
    shape=2.0, scale=1000.0 => mean=2000, CV=0.707
    n=2000 observations, 3 features.
    """
    n = 2000
    X = pd.DataFrame({
        "age": rng.integers(18, 75, size=n).astype(float),
        "vehicle_age": rng.integers(0, 15, size=n).astype(float),
        "region": rng.integers(0, 5, size=n).astype(float),
    })
    # True log-mu depends on features
    log_mu = (
        7.5
        + 0.005 * (X["age"] - 40)
        - 0.02 * X["vehicle_age"]
        + 0.1 * X["region"]
    )
    mu = np.exp(log_mu)
    shape = 2.0
    scale = mu / shape
    y = rng.gamma(shape=shape, scale=scale)
    return X, y, mu


@pytest.fixture(scope="session")
def small_gamma_data(rng):
    """Smaller dataset for faster tests — n=300."""
    n = 300
    X = pd.DataFrame({
        "age": rng.integers(18, 75, size=n).astype(float),
        "region": rng.integers(0, 3, size=n).astype(float),
    })
    mu = np.exp(7.0 + 0.003 * (X["age"] - 40))
    shape = 2.0
    scale = mu / shape
    y = rng.gamma(shape=shape, scale=scale)
    return X, y, mu


@pytest.fixture(scope="session")
def mock_glm_baseline(gamma_data):
    """
    A mock GLM baseline that returns known Gamma parameters.
    Avoids statsmodels dependency for core unit tests.
    """
    X, y, mu = gamma_data

    class MockGLMBaseline:
        distribution_family = "gamma"

        def predict_params(self, X_new):
            n = len(X_new)
            return {"mu": np.full(n, float(np.mean(mu))), "dispersion": 0.5}

        def predict_cdf(self, X_new, cutpoints):
            from scipy import stats
            params = self.predict_params(X_new)
            mu_val = params["mu"][:, np.newaxis]
            disp = params["dispersion"]
            alpha = 1.0 / disp
            scale = mu_val * disp
            return stats.gamma.cdf(cutpoints[np.newaxis, :], a=alpha, scale=scale)

    return MockGLMBaseline()


@pytest.fixture(scope="session")
def mock_glm_baseline_small(small_gamma_data):
    """Mock GLM baseline for small dataset."""
    X, y, mu = small_gamma_data

    class MockGLMBaseline:
        distribution_family = "gamma"

        def predict_params(self, X_new):
            n = len(X_new)
            return {"mu": np.full(n, float(np.mean(mu))), "dispersion": 0.5}

        def predict_cdf(self, X_new, cutpoints):
            from scipy import stats
            params = self.predict_params(X_new)
            mu_val = params["mu"][:, np.newaxis]
            disp = params["dispersion"]
            alpha = 1.0 / disp
            scale = mu_val * disp
            return stats.gamma.cdf(cutpoints[np.newaxis, :], a=alpha, scale=scale)

    return MockGLMBaseline()
