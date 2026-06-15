"""Visualizing how the Greeks move across strike and time to expiry.

This is the centerpiece. Each function returns a figure object (matplotlib
``Figure`` or plotly ``go.Figure``) so it can be embedded, tested, or saved by
``scripts/generate_figures.py``. Nothing here calls ``plt.show()``.

The default base case is S=100, r=4%, q=0, sigma=20%, varying strike and T.
The visuals are tuned to make three textbook facts unmistakable:

* Gamma peaks at the money and spikes sharply as T -> 0.
* Vega peaks at the money and grows with maturity.
* Delta is a smooth S-curve in moneyness that steepens as T -> 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from . import greeks

try:  # plotly is only needed for the interactive surfaces.
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - plotly always present per requirements
    go = None  # type: ignore


# --------------------------------------------------------------------------
# Base case and Greek dispatch
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class BaseCase:
    """Default market parameters for the figures."""

    S: float = 100.0
    r: float = 0.04
    q: float = 0.0
    sigma: float = 0.20


BASE = BaseCase()

# Pretty labels and the maturities / moneyness lines used across the figures.
GREEK_LABELS: Dict[str, str] = {
    "delta": "Delta",
    "gamma": "Gamma",
    "vega": "Vega (per 1.00 vol)",
    "theta": "Theta (per year)",
    "vanna": "Vanna",
    "volga": "Volga",
}
DEFAULT_MATURITIES: Tuple[float, ...] = (0.05, 0.25, 0.50, 1.00)


def eval_greek(
    name: str,
    S: NDArray | float,
    K: NDArray | float,
    T: NDArray | float,
    r: float,
    sigma: float,
    q: float,
    option_type: str = "call",
) -> NDArray:
    """Dispatch to a Greek by name and return a numpy array."""
    name = name.lower()
    if name == "delta":
        out = greeks.delta(S, K, T, r, sigma, q, option_type)
    elif name == "gamma":
        out = greeks.gamma(S, K, T, r, sigma, q)
    elif name == "vega":
        out = greeks.vega(S, K, T, r, sigma, q)
    elif name == "theta":
        out = greeks.theta(S, K, T, r, sigma, q, option_type)
    elif name == "vanna":
        out = greeks.vanna(S, K, T, r, sigma, q)
    elif name == "volga":
        out = greeks.volga(S, K, T, r, sigma, q)
    else:
        raise ValueError(f"unknown greek {name!r}")
    return np.asarray(out, dtype=np.float64)


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
def apply_style() -> None:
    """Apply a clean, readable matplotlib style for all static figures."""
    plt.rcParams.update(
        {
            "figure.figsize": (9.0, 5.5),
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "lines.linewidth": 2.0,
            "font.family": "DejaVu Sans",
        }
    )


def _maturity_colors(n: int) -> NDArray:
    return plt.cm.viridis(np.linspace(0.15, 0.9, n))


# --------------------------------------------------------------------------
# A) Greek vs strike, one line per maturity
# --------------------------------------------------------------------------
def plot_greek_vs_strike(
    name: str,
    maturities: Sequence[float] = DEFAULT_MATURITIES,
    base: BaseCase = BASE,
    strikes: NDArray | None = None,
    option_type: str = "call",
) -> Figure:
    """2D line plot of a Greek versus strike, one line per time to expiry.

    The shorter maturities show the sharp at-the-money structure (tall narrow
    gamma, steep delta); the longer ones are broad and smooth.
    """
    if strikes is None:
        strikes = np.linspace(0.6 * base.S, 1.4 * base.S, 401)
    colors = _maturity_colors(len(maturities))

    fig, ax = plt.subplots()
    for T, c in zip(maturities, colors):
        y = eval_greek(name, base.S, strikes, T, base.r, base.sigma, base.q, option_type)
        ax.plot(strikes, y, color=c, label=f"T = {T:g}y")
    ax.axvline(base.S, color="0.5", ls="--", lw=1, label="ATM (K = S)")
    ax.set_xlabel("Strike K  (spot S = 100)")
    ax.set_ylabel(GREEK_LABELS.get(name, name))
    ax.set_title(f"{GREEK_LABELS.get(name, name)} vs strike ({option_type}), by maturity")
    ax.legend(title="maturity")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# B) Greek vs time to expiry, one line per moneyness
# --------------------------------------------------------------------------
def plot_greek_vs_time(
    name: str,
    base: BaseCase = BASE,
    strikes_by_label: Dict[str, float] | None = None,
    T_grid: NDArray | None = None,
    option_type: str = "call",
) -> Figure:
    """2D line plot of a Greek versus time to expiry, one line per moneyness.

    Moneyness is set by the strike at fixed spot S = 100: ITM/ATM/OTM for a
    call correspond to K = 90 / 100 / 110.
    """
    if strikes_by_label is None:
        strikes_by_label = {"ITM (K=90)": 90.0, "ATM (K=100)": 100.0, "OTM (K=110)": 110.0}
    if T_grid is None:
        T_grid = np.linspace(0.02, 1.5, 400)
    colors = {"ITM (K=90)": "#1b7837", "ATM (K=100)": "#2166ac", "OTM (K=110)": "#b2182b"}

    fig, ax = plt.subplots()
    for label, K in strikes_by_label.items():
        y = eval_greek(name, base.S, K, T_grid, base.r, base.sigma, base.q, option_type)
        ax.plot(T_grid, y, label=label, color=colors.get(label))
    ax.set_xlabel("Time to expiry T (years)")
    ax.set_ylabel(GREEK_LABELS.get(name, name))
    ax.set_title(f"{GREEK_LABELS.get(name, name)} vs time to expiry ({option_type})")
    ax.legend(title="moneyness")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Grid helper for surfaces and heatmaps
# --------------------------------------------------------------------------
def _surface_grid(
    name: str,
    base: BaseCase,
    option_type: str,
    n_strike: int = 120,
    n_time: int = 120,
    strike_lo: float = 0.6,
    strike_hi: float = 1.4,
    T_lo: float = 0.02,
    T_hi: float = 1.0,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Return (K_grid, T_grid, Z) with Z[i, j] = greek at (T_grid[i], K_grid[j])."""
    K = np.linspace(strike_lo * base.S, strike_hi * base.S, n_strike)
    T = np.linspace(T_lo, T_hi, n_time)
    KK, TT = np.meshgrid(K, T)  # shapes (n_time, n_strike)
    Z = eval_greek(name, base.S, KK, TT, base.r, base.sigma, base.q, option_type)
    return K, T, Z


# --------------------------------------------------------------------------
# C) 3D surface (matplotlib static)
# --------------------------------------------------------------------------
def surface_matplotlib(
    name: str, base: BaseCase = BASE, option_type: str = "call", **grid_kwargs
) -> Figure:
    """Static 3D surface of a Greek over (strike, time-to-expiry)."""
    K, T, Z = _surface_grid(name, base, option_type, **grid_kwargs)
    KK, TT = np.meshgrid(K, T)

    fig = plt.figure(figsize=(9.5, 7.0), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(KK, TT, Z, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Time to expiry T (y)")
    ax.set_zlabel(GREEK_LABELS.get(name, name))
    ax.set_title(f"{GREEK_LABELS.get(name, name)} surface ({option_type})")
    ax.view_init(elev=28, azim=-122)
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label=GREEK_LABELS.get(name, name))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# C') 3D surface (plotly interactive)
# --------------------------------------------------------------------------
def surface_plotly(
    name: str, base: BaseCase = BASE, option_type: str = "call", **grid_kwargs
) -> "go.Figure":
    """Interactive plotly 3D surface (returns a go.Figure)."""
    if go is None:  # pragma: no cover
        raise RuntimeError("plotly is not installed")
    K, T, Z = _surface_grid(name, base, option_type, **grid_kwargs)
    fig = go.Figure(data=[go.Surface(x=K, y=T, z=Z, colorscale="Viridis", colorbar=dict(title=name))])
    fig.update_layout(
        title=f"{GREEK_LABELS.get(name, name)} surface ({option_type}) - S={base.S:g}, r={base.r:g}, sigma={base.sigma:g}",
        scene=dict(
            xaxis_title="Strike K",
            yaxis_title="Time to expiry T (y)",
            zaxis_title=GREEK_LABELS.get(name, name),
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


# --------------------------------------------------------------------------
# D) Heatmap (2D read of the surface)
# --------------------------------------------------------------------------
def heatmap(
    name: str, base: BaseCase = BASE, option_type: str = "call", **grid_kwargs
) -> Figure:
    """Heatmap of a Greek over strike (x) and time to expiry (y)."""
    K, T, Z = _surface_grid(name, base, option_type, **grid_kwargs)
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    mesh = ax.pcolormesh(K, T, Z, cmap="viridis", shading="auto")
    ax.axvline(base.S, color="white", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("Strike K  (spot S = 100)")
    ax.set_ylabel("Time to expiry T (years)")
    ax.set_title(f"{GREEK_LABELS.get(name, name)} heatmap ({option_type})")
    fig.colorbar(mesh, ax=ax, label=GREEK_LABELS.get(name, name))
    fig.tight_layout()
    return fig
