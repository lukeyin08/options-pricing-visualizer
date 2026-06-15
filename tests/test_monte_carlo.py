"""Tests for the Monte Carlo pricer, variance reduction, convergence, MC Greeks.

All randomness is seeded, so every assertion below is deterministic.
"""
from __future__ import annotations

import numpy as np
import pytest

from options_viz import greeks, monte_carlo as mc
from options_viz.black_scholes import bs_price

# (S, K, T, r, sigma, q, option_type)
CASES = [
    (100.0, 90.0, 1.0, 0.05, 0.20, 0.00, "call"),
    (100.0, 100.0, 1.0, 0.05, 0.20, 0.00, "call"),
    (100.0, 110.0, 0.5, 0.05, 0.30, 0.00, "call"),
    (100.0, 100.0, 1.0, 0.05, 0.20, 0.02, "put"),
    (100.0, 80.0, 2.0, 0.03, 0.25, 0.00, "put"),
]
SEED = 12345
N = 300_000


@pytest.mark.parametrize("S,K,T,r,sigma,q,ot", CASES)
@pytest.mark.parametrize("estimator", [mc.mc_price, mc.mc_price_antithetic, mc.mc_price_control_variate])
def test_price_within_3_standard_errors_of_bs(estimator, S, K, T, r, sigma, q, ot):
    bs = float(bs_price(S, K, T, r, sigma, q, ot))
    res = estimator(S, K, T, r, sigma, q, ot, n=N, seed=SEED)
    assert abs(res.estimate - bs) < 3.0 * res.std_error
    assert np.isfinite(res.estimate) and np.isfinite(res.std_error)


def test_estimates_are_seed_reproducible():
    a = mc.mc_price(100, 100, 1.0, 0.05, 0.2, 0.0, "call", n=50_000, seed=7)
    b = mc.mc_price(100, 100, 1.0, 0.05, 0.2, 0.0, "call", n=50_000, seed=7)
    c = mc.mc_price(100, 100, 1.0, 0.05, 0.2, 0.0, "call", n=50_000, seed=8)
    assert a.estimate == b.estimate and a.std_error == b.std_error
    assert a.estimate != c.estimate


@pytest.mark.parametrize("S,K,T,r,sigma,q,ot", CASES)
def test_variance_reduction_factors_exceed_one(S, K, T, r, sigma, q, ot):
    rep = mc.variance_reduction_report(S, K, T, r, sigma, q, ot, n=200_000, seed=0)
    assert rep["vrf_antithetic"] > 1.0
    assert rep["vrf_control_variate"] > 1.0


def test_variance_reduction_is_substantial_atm_call():
    # Concrete, comfortable margins for the canonical ATM call.
    rep = mc.variance_reduction_report(100, 100, 1.0, 0.05, 0.2, 0.0, "call", n=200_000, seed=0)
    assert rep["vrf_antithetic"] > 1.5
    assert rep["vrf_control_variate"] > 3.0
    assert rep["se_control_variate"] < rep["se_antithetic"] < rep["se_naive"]


def test_convergence_is_root_n():
    cs = mc.convergence_study(100, 100, 1.0, 0.05, 0.2, 0.0, "call",
                              n_grid=(10**3, 10**4, 10**5, 10**6), seed=0)
    # Half-width shrinks monotonically across decades.
    hw = cs["ci_halfwidth"]
    assert np.all(np.diff(hw) < 0)
    # Log-log slope of half-width vs N is ~ -1/2.
    slope = np.polyfit(np.log(cs["N"]), np.log(hw), 1)[0]
    assert -0.6 < slope < -0.4


# ----------------------------- MC Greeks -----------------------------------
def test_pathwise_delta_matches_analytic():
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0
    an = float(greeks.delta(S, K, T, r, sigma, q, "call"))
    res = mc.mc_delta_pathwise(S, K, T, r, sigma, q, "call", n=400_000, seed=2)
    assert abs(res.estimate - an) < 3.0 * res.std_error
    assert abs(res.estimate - an) < 0.01


def test_lr_gamma_matches_analytic():
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0
    an = float(greeks.gamma(S, K, T, r, sigma, q))
    res = mc.mc_gamma_lr(S, K, T, r, sigma, q, "call", n=1_000_000, seed=3)
    assert abs(res.estimate - an) < 3.0 * res.std_error


def test_crn_gamma_matches_analytic_and_is_low_variance():
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0
    an = float(greeks.gamma(S, K, T, r, sigma, q))
    res = mc.mc_gamma_crn(S, K, T, r, sigma, q, "call", n=400_000, seed=4)
    assert abs(res.estimate - an) < 3.0 * res.std_error
    assert abs(res.estimate - an) < 2e-3
    # CRN keeps the second-difference standard error small (well under the Greek).
    assert res.std_error < 1e-3


def test_pathwise_delta_put_sign():
    # Put delta is negative.
    res = mc.mc_delta_pathwise(100, 100, 1.0, 0.05, 0.2, 0.0, "put", n=200_000, seed=5)
    assert res.estimate < 0.0


def test_variance_reduction_report_handles_zero_variance():
    # Deep OTM, low vol, short T: every simulated payoff is 0, so all standard
    # errors are 0. The report must not raise ZeroDivisionError.
    import math

    rep = mc.variance_reduction_report(100, 200, 0.1, 0.05, 0.10, 0.0, "call", n=20_000, seed=0)
    assert rep["se_naive"] == 0.0
    for key in ("vrf_antithetic", "vrf_control_variate"):
        assert math.isinf(rep[key]) or math.isnan(rep[key])
