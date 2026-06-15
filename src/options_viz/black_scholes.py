"""Black-Scholes-Merton pricing for European options.

All formulas are built on the from-scratch standard normal CDF in
``normal.py``. No external pricing or statistics library is used.

Model assumptions: the underlying follows geometric Brownian motion under the
risk-neutral measure with constant volatility ``sigma``, constant risk-free
rate ``r`` and continuous dividend yield ``q``. Options are European.

Notation::

    d1 = (ln(S/K) + (r - q + sigma^2 / 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    Call = S e^{-qT} N(d1) - K e^{-rT} N(d2)
    Put  = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)
"""
from __future__ import annotations

import math
from typing import Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .normal import norm_cdf, norm_pdf

__all__ = [
    "d1_d2",
    "bs_price",
    "call_price",
    "put_price",
    "put_call_parity_gap",
    "implied_vol",
]

FloatOrArray = Union[float, NDArray[np.float64]]


def _unwrap(x: np.ndarray) -> FloatOrArray:
    """Return a Python float for 0-d arrays, otherwise the array itself."""
    return x.item() if x.ndim == 0 else x


def _normalize_type(option_type: str) -> str:
    """Map assorted spellings to canonical 'call'/'put'."""
    s = str(option_type).strip().lower()
    if s in ("c", "call"):
        return "call"
    if s in ("p", "put"):
        return "put"
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def d1_d2(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> Tuple[FloatOrArray, FloatOrArray]:
    """Compute the Black-Scholes d1 and d2 terms.

    Where the total volatility ``sigma * sqrt(T)`` is zero (T -> 0 or
    sigma -> 0) the ratio is undefined; the denominator is replaced by 1 to
    avoid divide-by-zero. Callers that price options treat that regime
    separately (intrinsic value), so the specific d1/d2 returned there is not
    used downstream.

    Returns:
        (d1, d2), broadcast over the inputs.
    """
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    vol = sigma * np.sqrt(T)  # total (not annualized) volatility over [0, T]
    safe_vol = np.where(vol > 0.0, vol, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / safe_vol
        d2 = d1 - safe_vol
    return _unwrap(np.asarray(d1)), _unwrap(np.asarray(d2))


def bs_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: str = "call",
) -> FloatOrArray:
    """Black-Scholes-Merton price of a European call or put.

    Fully vectorized and free of NaNs on the documented edge cases:

    * ``T -> 0`` or ``sigma -> 0`` collapses to the discounted intrinsic value
      ``max(S e^{-qT} - K e^{-rT}, 0)`` (call) which equals ``max(S - K, 0)``
      at ``T = 0``. This single expression is the common limit of both
      degeneracies, so no separate branch is needed for each.
    * Deep in-/out-of-the-money inputs push N(.) to 0 or 1 smoothly.

    Args:
        S, K, T, r, sigma, q: Spot, strike, time to expiry (years), risk-free
            rate, volatility, continuous dividend yield. Array-like and
            broadcastable.
        option_type: 'call' or 'put'.

    Returns:
        Option price(s); float for scalar inputs, else an ndarray.
    """
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    ot = _normalize_type(option_type)

    vol = sigma * np.sqrt(T)
    safe_vol = np.where(vol > 0.0, vol, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / safe_vol
        d2 = d1 - safe_vol

    disc_S = S * np.exp(-q * T)  # dividend-discounted spot (= forward * e^{-rT})
    disc_K = K * np.exp(-r * T)  # PV of strike

    if ot == "call":
        formula = disc_S * norm_cdf(d1) - disc_K * norm_cdf(d2)
        intrinsic = np.maximum(disc_S - disc_K, 0.0)
    else:
        formula = disc_K * norm_cdf(-d2) - disc_S * norm_cdf(-d1)
        intrinsic = np.maximum(disc_K - disc_S, 0.0)

    price = np.where(vol > 0.0, formula, intrinsic)
    return _unwrap(np.asarray(price))


def call_price(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike = 0.0
) -> FloatOrArray:
    """European call price (see :func:`bs_price`)."""
    return bs_price(S, K, T, r, sigma, q, "call")


def put_price(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike = 0.0
) -> FloatOrArray:
    """European put price (see :func:`bs_price`)."""
    return bs_price(S, K, T, r, sigma, q, "put")


def put_call_parity_gap(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike = 0.0
) -> FloatOrArray:
    """Residual of put-call parity: (C - P) - (S e^{-qT} - K e^{-rT}).

    For arbitrage-free European prices this is identically zero. Our
    implementation satisfies it to floating-point precision because the
    from-scratch ``norm_cdf`` is exactly odd-symmetric, so N(x) + N(-x) = 1.
    """
    c = np.asarray(bs_price(S, K, T, r, sigma, q, "call"), dtype=np.float64)
    p = np.asarray(bs_price(S, K, T, r, sigma, q, "put"), dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    gap = (c - p) - (S * np.exp(-q * T) - K * np.exp(-r * T))
    return _unwrap(np.asarray(gap))


def _vega_scalar(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
    """Vega for scalar inputs (per 1.00 change in vol). Local copy to keep this
    module independent of greeks.py and avoid a circular import."""
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    phi = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    return S * math.exp(-q * T) * phi * math.sqrt(T)


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: str = "call",
    *,
    tol: float = 1e-8,
    max_iter: int = 100,
    lo: float = 1e-8,
    hi: float = 5.0,
) -> float:
    """Implied volatility from an observed price (scalar inputs).

    Newton-Raphson on price(sigma) - target, using vega as the derivative,
    with a bisection fallback. Bisection is guaranteed to converge because the
    price is strictly increasing in sigma, so the target is bracketed by
    ``[lo, hi]`` whenever it is inside the no-arbitrage bounds.

    Args:
        price: Observed option price.
        S, K, T, r, q: Standard Black-Scholes inputs.
        option_type: 'call' or 'put'.
        tol: Absolute price tolerance for convergence.
        max_iter: Maximum iterations for each stage.
        lo, hi: Volatility search bracket.

    Returns:
        The implied volatility.

    Raises:
        ValueError: If the price violates the no-arbitrage bounds.
    """
    price = float(price)
    S, K, T, r, q = float(S), float(K), float(T), float(r), float(q)
    ot = _normalize_type(option_type)

    disc_S = S * math.exp(-q * T)
    disc_K = K * math.exp(-r * T)
    lower = max(disc_S - disc_K, 0.0) if ot == "call" else max(disc_K - disc_S, 0.0)
    upper = disc_S if ot == "call" else disc_K

    if price < lower - 1e-10 or price > upper + 1e-10:
        raise ValueError(
            f"price {price:.6g} is outside the no-arbitrage bounds "
            f"[{lower:.6g}, {upper:.6g}] for a {ot}."
        )
    def f(sig: float) -> float:
        return float(bs_price(S, K, T, r, sig, q, ot)) - price

    # At the lower bound the option is worth its intrinsic value to within
    # floating point: time value (and vega) have vanished, so the vol is not
    # identifiable and 0 is the only sensible answer.
    if price <= lower + tol:
        return lo
    # The price is strictly increasing in sigma and approaches `upper` only as
    # sigma -> infinity. If the target exceeds the most we can reach at hi, the
    # implied vol is outside the search bracket; say so rather than guess.
    price_at_hi = f(hi) + price
    if price > price_at_hi + tol:
        raise ValueError(
            f"implied vol exceeds the search bound hi={hi:g}: price {price:.6g} is above "
            f"the maximum reachable {price_at_hi:.6g}. Increase hi."
        )

    # --- Newton-Raphson -------------------------------------------------
    sigma = 0.2  # neutral starting guess
    for _ in range(max_iter):
        diff = f(sigma)
        if abs(diff) < tol:
            return sigma
        v = _vega_scalar(S, K, T, r, sigma, q)
        if v < 1e-8:  # vega too flat for a reliable Newton step
            break
        step = diff / v
        sigma_new = sigma - step
        if not (lo < sigma_new < hi):  # left the bracket: hand off to bisection
            break
        sigma = sigma_new

    # --- Bisection fallback (now guaranteed bracketed: f(lo) <= 0 <= f(hi)) --
    a, b = lo, hi
    fa = f(a)
    m = 0.5 * (a + b)
    for _ in range(200):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol or (b - a) < 1e-13:
            return m
        if (fm > 0.0) == (fa > 0.0):
            a, fa = m, fm
        else:
            b = m
    if abs(f(m)) > max(tol, 1e-7):  # never silently return a non-converged value
        raise ValueError(f"implied_vol failed to converge (residual {f(m):.2e}).")
    return m
