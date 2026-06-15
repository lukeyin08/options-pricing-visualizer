"""Monte Carlo pricing of European options under risk-neutral GBM.

Terminal sampling (single step, exact for European payoffs)::

    S_T = S_0 * exp((r - q - sigma^2/2) * T + sigma * sqrt(T) * Z),  Z ~ N(0, 1)
    price = exp(-r T) * mean(payoff(S_T))

The terminal law of S_T is exact under geometric Brownian motion, so no time
discretization is needed here. For path-dependent payoffs (Asian, barrier,
lookback) one would instead simulate the whole path on a time grid; the
sampling below would become a loop-free (N, n_steps) array of increments.

Variance reduction implemented and measured:
  * Antithetic variates: pair each Z with -Z.
  * Control variates: use the terminal stock S_T as the control, exploiting the
    known mean E[S_T] = S_0 exp((r - q) T).

Everything that touches the N samples is vectorized (no Python loop over draws).
scipy is not used; the 95% normal quantile is hard-coded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from .black_scholes import _normalize_type, bs_price

__all__ = [
    "MCResult",
    "mc_price",
    "mc_price_antithetic",
    "mc_price_control_variate",
    "variance_reduction_report",
    "convergence_study",
    "mc_delta_pathwise",
    "mc_gamma_lr",
    "mc_gamma_crn",
]

# 97.5th percentile of N(0,1): the 95% two-sided CI multiplier.
Z95: float = 1.959963984540054


@dataclass(frozen=True)
class MCResult:
    """A Monte Carlo estimate with its sampling error.

    ``estimate`` is the point estimate (a price or a Greek). ``std_error`` is
    the standard error of that estimate. ``price`` is an alias for ``estimate``
    so price results read naturally.
    """

    estimate: float
    std_error: float
    n_samples: int
    method: str = "naive"

    @property
    def price(self) -> float:
        """Alias for ``estimate`` so price results read naturally."""
        return self.estimate

    @property
    def ci95(self) -> Tuple[float, float]:
        """The 95% confidence interval, ``estimate +/- 1.96 * std_error``."""
        h = Z95 * self.std_error
        return (self.estimate - h, self.estimate + h)

    @property
    def ci_halfwidth(self) -> float:
        """Half-width of the 95% confidence interval (``1.96 * std_error``)."""
        return Z95 * self.std_error


# --------------------------------------------------------------------------
# Core helpers (vectorized)
# --------------------------------------------------------------------------
def _terminal_price(S: float, T: float, r: float, sigma: float, q: float, Z: NDArray) -> NDArray:
    """Vectorized risk-neutral terminal price S_T for standard normals Z."""
    drift = (r - q - 0.5 * sigma * sigma) * T
    diffusion = sigma * np.sqrt(T) * Z
    return S * np.exp(drift + diffusion)


def _payoff(S_T: NDArray, K: float, option_type: str) -> NDArray:
    if option_type == "call":
        return np.maximum(S_T - K, 0.0)
    return np.maximum(K - S_T, 0.0)


def _summarize(discounted: NDArray, method: str) -> MCResult:
    """Mean, standard error (ddof=1), and sample count of a discounted payoff."""
    n = discounted.size
    est = float(np.mean(discounted))
    se = float(np.std(discounted, ddof=1) / np.sqrt(n))
    return MCResult(estimate=est, std_error=se, n_samples=n, method=method)


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------
def mc_price(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
    option_type: str = "call", n: int = 100_000, seed: int | None = None,
) -> MCResult:
    """Naive Monte Carlo price with a 95% confidence interval."""
    ot = _normalize_type(option_type)
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n)
    S_T = _terminal_price(S, T, r, sigma, q, Z)
    discounted = np.exp(-r * T) * _payoff(S_T, K, ot)
    return _summarize(discounted, method="naive")


def mc_price_antithetic(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
    option_type: str = "call", n: int = 100_000, seed: int | None = None,
) -> MCResult:
    """Antithetic-variates price.

    Draw n/2 normals Z and reuse -Z. Averaging the discounted payoff of each
    (Z, -Z) pair cancels the part of the payoff that is odd in Z, lowering
    variance whenever the payoff is monotone in Z (true for vanilla calls/puts).
    The standard error is computed from the m = n/2 independent pair averages.
    """
    ot = _normalize_type(option_type)
    m = n // 2
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(m)
    disc = np.exp(-r * T)
    p_plus = disc * _payoff(_terminal_price(S, T, r, sigma, q, Z), K, ot)
    p_minus = disc * _payoff(_terminal_price(S, T, r, sigma, q, -Z), K, ot)
    pair_avg = 0.5 * (p_plus + p_minus)  # m independent, lower-variance samples
    est = float(np.mean(pair_avg))
    se = float(np.std(pair_avg, ddof=1) / np.sqrt(m))
    return MCResult(estimate=est, std_error=se, n_samples=2 * m, method="antithetic")


def mc_price_control_variate(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
    option_type: str = "call", n: int = 100_000, seed: int | None = None,
) -> MCResult:
    """Control-variate price using the terminal stock S_T as the control.

    We know E[S_T] = S_0 exp((r - q) T) exactly. The estimator
    ``D - c (S_T - E[S_T])`` has the same mean as the discounted payoff D for
    any c, and the variance-minimizing c* = Cov(D, S_T)/Var(S_T) is estimated
    from the sample. Because the discounted call payoff is strongly correlated
    with S_T, this removes a large share of the variance.
    """
    ot = _normalize_type(option_type)
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n)
    S_T = _terminal_price(S, T, r, sigma, q, Z)
    D = np.exp(-r * T) * _payoff(S_T, K, ot)
    EST = S * np.exp((r - q) * T)  # known control mean
    cov = np.cov(D, S_T, ddof=1)
    c_star = cov[0, 1] / cov[1, 1]
    adjusted = D - c_star * (S_T - EST)
    return _summarize(adjusted, method="control_variate")


# --------------------------------------------------------------------------
# Measuring variance reduction
# --------------------------------------------------------------------------
def variance_reduction_report(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
    option_type: str = "call", n: int = 200_000, seed: int = 0,
) -> Dict[str, float]:
    """Compare naive, antithetic, and control-variate standard errors at equal n.

    The variance reduction factor is (se_naive / se_method)^2: how many times
    more naive samples you would need to match the method's accuracy.
    """
    naive = mc_price(S, K, T, r, sigma, q, option_type, n=n, seed=seed)
    anti = mc_price_antithetic(S, K, T, r, sigma, q, option_type, n=n, seed=seed)
    cv = mc_price_control_variate(S, K, T, r, sigma, q, option_type, n=n, seed=seed)

    def _vrf(se_base: float, se_method: float) -> float:
        # Guard the degenerate case (e.g. deep OTM where every payoff is 0, so
        # all standard errors are 0): infinite reduction if there was variance
        # to remove, otherwise undefined.
        if se_method == 0.0:
            return float("inf") if se_base > 0.0 else float("nan")
        return (se_base / se_method) ** 2

    return {
        "se_naive": naive.std_error,
        "se_antithetic": anti.std_error,
        "se_control_variate": cv.std_error,
        "vrf_antithetic": _vrf(naive.std_error, anti.std_error),
        "vrf_control_variate": _vrf(naive.std_error, cv.std_error),
    }


# --------------------------------------------------------------------------
# Convergence study
# --------------------------------------------------------------------------
def convergence_study(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
    option_type: str = "call", n_grid: Tuple[int, ...] = (10**3, 10**4, 10**5, 10**6),
    seed: int = 0, method: str = "naive",
) -> Dict[str, NDArray]:
    """Price, standard error, and CI half-width as a function of sample size N.

    Returns arrays suitable for a log-log plot of half-width vs N (the slope is
    -1/2) and of the error vs the analytical Black-Scholes price. The loop is
    over the handful of N values only; each individual estimate is vectorized.
    """
    estimator: Callable[..., MCResult] = {
        "naive": mc_price,
        "antithetic": mc_price_antithetic,
        "control_variate": mc_price_control_variate,
    }[method]
    bs = float(bs_price(S, K, T, r, sigma, q, option_type))
    Ns, prices, ses, halfs, errs = [], [], [], [], []
    for i, N in enumerate(n_grid):
        res = estimator(S, K, T, r, sigma, q, option_type, n=int(N), seed=seed + i)
        Ns.append(res.n_samples)
        prices.append(res.estimate)
        ses.append(res.std_error)
        halfs.append(res.ci_halfwidth)
        errs.append(abs(res.estimate - bs))
    return {
        "N": np.asarray(Ns, dtype=float),
        "price": np.asarray(prices),
        "std_error": np.asarray(ses),
        "ci_halfwidth": np.asarray(halfs),
        "abs_error": np.asarray(errs),
        "bs_price": np.full(len(n_grid), bs),
    }


# --------------------------------------------------------------------------
# Monte Carlo Greeks (stretch)
# --------------------------------------------------------------------------
def mc_delta_pathwise(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
    option_type: str = "call", n: int = 200_000, seed: int | None = None,
) -> MCResult:
    """Pathwise delta.

    The payoff is (a.s.) differentiable in S_0, and dS_T/dS_0 = S_T/S_0, so by
    the chain rule the pathwise estimator for a call is
    ``exp(-rT) * E[ 1{S_T > K} * S_T / S_0 ]`` (the indicator is the derivative
    of the payoff). For a put it is ``-exp(-rT) * E[ 1{S_T < K} * S_T / S_0 ]``.
    Pathwise estimators reuse the simulated paths and are low variance.
    """
    ot = _normalize_type(option_type)
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n)
    S_T = _terminal_price(S, T, r, sigma, q, Z)
    disc = np.exp(-r * T)
    if ot == "call":
        contrib = disc * (S_T > K) * (S_T / S)
    else:
        contrib = -disc * (S_T < K) * (S_T / S)
    return _summarize(contrib, method="pathwise_delta")


def mc_gamma_lr(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
    option_type: str = "call", n: int = 1_000_000, seed: int | None = None,
) -> MCResult:
    """Gamma via the likelihood ratio (score) method.

    Gamma has no pathwise estimator for vanilla payoffs (the payoff's second
    derivative is a delta function), so we differentiate the *density* instead.
    For log S_T ~ N(m(S_0), sigma^2 T) the second-order score in S_0 is

        w = (Z^2 - 1)/(S_0^2 sigma^2 T) - Z/(S_0^2 sigma sqrt(T)),

    giving ``gamma = exp(-rT) * E[payoff * w]``. This is unbiased but high
    variance (hence the large default n).
    """
    ot = _normalize_type(option_type)
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n)
    S_T = _terminal_price(S, T, r, sigma, q, Z)
    payoff = _payoff(S_T, K, ot)
    sqrtT = np.sqrt(T)
    weight = (Z * Z - 1.0) / (S * S * sigma * sigma * T) - Z / (S * S * sigma * sqrtT)
    contrib = np.exp(-r * T) * payoff * weight
    return _summarize(contrib, method="lr_gamma")


def mc_gamma_crn(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
    option_type: str = "call", h: float | None = None, n: int = 200_000,
    seed: int | None = None,
) -> MCResult:
    """Gamma by a second-order bump using common random numbers (CRN).

    Estimate ``gamma ~ [V(S+h) - 2 V(S) + V(S-h)] / h^2`` where all three prices
    are computed from the *same* normals Z. With independent draws the variance
    of this difference explodes like O(1/h^2) because three noisy numbers are
    differenced; CRN makes the three estimators almost perfectly correlated, so
    the simulation noise cancels in the numerator and only the (tiny) true
    curvature survives. The per-path bumped payoffs are differenced first, then
    averaged, so the standard error reflects the variance of that difference.
    """
    ot = _normalize_type(option_type)
    if h is None:
        h = 0.01 * S
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n)
    disc = np.exp(-r * T)
    # Same Z for all three spots (common random numbers).
    p_up = disc * _payoff(_terminal_price(S + h, T, r, sigma, q, Z), K, ot)
    p_mid = disc * _payoff(_terminal_price(S, T, r, sigma, q, Z), K, ot)
    p_dn = disc * _payoff(_terminal_price(S - h, T, r, sigma, q, Z), K, ot)
    per_path = (p_up - 2.0 * p_mid + p_dn) / (h * h)
    return _summarize(per_path, method="crn_gamma")
