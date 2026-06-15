"""Tests for Black-Scholes pricing: reference values, parity, edges, IV."""
from __future__ import annotations

import numpy as np
import pytest

from options_viz.black_scholes import (
    bs_price,
    call_price,
    implied_vol,
    put_call_parity_gap,
    put_price,
)

# (S, K, T, r, sigma, q, call_ref, put_ref). Reference values are the exact
# Black-Scholes prices (cross-checked against scipy's norm.cdf to 10 dp).
REFERENCE_CASES = [
    (42.0, 40.0, 0.50, 0.10, 0.20, 0.00, 4.7594, 0.8086),   # Hull, classic example
    (100.0, 100.0, 1.00, 0.05, 0.20, 0.00, 10.4506, 5.5735),  # ATM
    (50.0, 50.0, 0.25, 0.05, 0.30, 0.00, 3.2915, 2.6704),     # short-dated ATM
    (100.0, 95.0, 0.75, 0.04, 0.25, 0.03, 11.2783, 5.6955),   # with dividend yield
]


@pytest.mark.parametrize("S,K,T,r,sigma,q,call_ref,put_ref", REFERENCE_CASES)
def test_reference_prices(S, K, T, r, sigma, q, call_ref, put_ref):
    assert call_price(S, K, T, r, sigma, q) == pytest.approx(call_ref, abs=1e-3)
    assert put_price(S, K, T, r, sigma, q) == pytest.approx(put_ref, abs=1e-3)


def test_put_call_parity_grid():
    S = np.array([80.0, 100.0, 120.0])
    K = np.array([90.0, 100.0, 110.0])
    T = np.array([0.1, 0.5, 1.0, 2.0])
    r_vals = [0.0, 0.03, 0.07]
    q_vals = [0.0, 0.02, 0.05]
    sig_vals = [0.1, 0.25, 0.5]
    for r in r_vals:
        for q in q_vals:
            for sig in sig_vals:
                # Broadcast S (3,) against K (3,) against T (4,1) -> grid.
                gap = put_call_parity_gap(
                    S[:, None], K[:, None], T[None, :], r, sig, q
                )
                assert np.max(np.abs(np.asarray(gap))) < 1e-10


def test_edge_T_zero_is_intrinsic():
    # T = 0 -> payoff today.
    assert call_price(110.0, 100.0, 0.0, 0.05, 0.2) == pytest.approx(10.0, abs=1e-12)
    assert put_price(110.0, 100.0, 0.0, 0.05, 0.2) == pytest.approx(0.0, abs=1e-12)
    assert call_price(90.0, 100.0, 0.0, 0.05, 0.2) == pytest.approx(0.0, abs=1e-12)
    assert put_price(90.0, 100.0, 0.0, 0.05, 0.2) == pytest.approx(10.0, abs=1e-12)


def test_edge_sigma_zero_is_discounted_intrinsic():
    S, K, T, r, q = 100.0, 95.0, 1.0, 0.05, 0.02
    expected_call = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    expected_put = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
    assert call_price(S, K, T, r, 0.0, q) == pytest.approx(expected_call, abs=1e-12)
    assert put_price(S, K, T, r, 0.0, q) == pytest.approx(expected_put, abs=1e-12)


def test_deep_itm_otm_no_nan():
    # Extreme moneyness must stay finite and non-negative.
    prices = [
        call_price(1000.0, 100.0, 1.0, 0.05, 0.2),  # deep ITM call
        call_price(1.0, 100.0, 1.0, 0.05, 0.2),      # deep OTM call
        put_price(1000.0, 100.0, 1.0, 0.05, 0.2),    # deep OTM put
        put_price(1.0, 100.0, 1.0, 0.05, 0.2),       # deep ITM put
    ]
    for p in prices:
        assert np.isfinite(p) and p >= 0.0
    # Deep ITM call ~ discounted intrinsic.
    assert call_price(1000.0, 100.0, 1.0, 0.05, 0.2) == pytest.approx(
        1000.0 - 100.0 * np.exp(-0.05), abs=1e-6
    )


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("sigma_true", [0.10, 0.20, 0.45])
def test_implied_vol_roundtrip(option_type, sigma_true):
    S, K, T, r, q = 100.0, 105.0, 0.75, 0.03, 0.01
    price = bs_price(S, K, T, r, sigma_true, q, option_type)
    iv = implied_vol(price, S, K, T, r, q, option_type)
    assert iv == pytest.approx(sigma_true, abs=1e-6)


def test_implied_vol_out_of_bounds_raises():
    S, K, T, r = 100.0, 100.0, 1.0, 0.05
    upper = S  # call price can never exceed spot (q = 0)
    with pytest.raises(ValueError):
        implied_vol(upper + 1.0, S, K, T, r, 0.0, "call")


def test_implied_vol_above_reachable_bound_raises():
    # For S=K=100, T=1, r=0.05 the max call price at sigma=5 is ~98.79, below the
    # no-arbitrage ceiling of 100. A target in between needs vol > hi: the solver
    # must raise, not silently return hi.
    assert bs_price(100, 100, 1.0, 0.05, 5.0, 0.0, "call") < 99.0 < 100.0
    with pytest.raises(ValueError):
        implied_vol(99.0, 100.0, 100.0, 1.0, 0.05, 0.0, "call")


@pytest.mark.parametrize("sigma_true", [0.12, 0.35, 0.80, 1.50])
def test_implied_vol_roundtrip_high_vol(sigma_true):
    # Any vol within the bracket round-trips exactly (no silent non-convergence).
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0
    price = bs_price(S, K, T, r, sigma_true, q, "call")
    assert implied_vol(price, S, K, T, r, q, "call") == pytest.approx(sigma_true, abs=1e-6)


def test_vectorized_matches_scalar():
    S = np.linspace(60, 140, 9)
    K, T, r, sigma, q = 100.0, 0.5, 0.04, 0.25, 0.0
    vec = np.asarray(call_price(S, K, T, r, sigma, q))
    scal = np.array([call_price(float(s), K, T, r, sigma, q) for s in S])
    assert np.allclose(vec, scal, atol=1e-12)


def test_monotonicity():
    # Call rises with spot; falls with strike.
    S = np.linspace(80, 120, 21)
    calls = np.asarray(call_price(S, 100.0, 1.0, 0.05, 0.2))
    assert np.all(np.diff(calls) > 0)
    K = np.linspace(80, 120, 21)
    calls_k = np.asarray(call_price(100.0, K, 1.0, 0.05, 0.2))
    assert np.all(np.diff(calls_k) < 0)
