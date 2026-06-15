# Options Pricing and Greeks Visualizer

A from-scratch Black-Scholes engine in Python: pricing, the full set of analytical Greeks, a Monte Carlo pricer with variance reduction, and a Greek visualizer across strike and time to expiry. The standard normal distribution, the pricing formulas, and every Greek are implemented from first principles. scipy is used only in the test suite as an independent reference, never in the core. 1763 tests pass, including every analytical Greek checked against finite differences and Monte Carlo checked against Black-Scholes inside its confidence interval.

## 1. Overview

The library prices European calls and puts under geometric Brownian motion with a continuous dividend yield, returns closed-form Greeks (delta, gamma, vega, theta, rho, plus the second-order vanna and volga), and cross-checks everything two ways: analytical Greeks against central finite differences, and Black-Scholes prices against a Monte Carlo estimator with antithetic and control variates. The visualizer turns the Greeks into 2D slices, 3D surfaces, and heatmaps that make the textbook behavior obvious: gamma spikes at the money as expiry approaches, vega peaks at the money and grows with maturity, and delta is an S-curve that steepens near expiry.

## 2. Quickstart

```bash
# 1. install (Python 3.11+ recommended; developed on 3.10+)
pip install -r requirements.txt

# 2. run the full test suite
python -m pytest            # 1763 tests

# 3. regenerate every figure into figures/
python scripts/generate_figures.py

# 4. print the BS vs MC tables and write the comparison figures
python scripts/run_comparison.py

# 5. launch the interactive dashboard (bonus)
streamlit run app/streamlit_app.py
```

The package uses a `src/` layout. Tests find it through the `pythonpath = ["src"]` setting in `pyproject.toml`; the scripts and app add `src/` to the path themselves, so nothing needs to be installed first.

## 3. The math

### 3.1 Risk-neutral pricing in one paragraph

Black-Scholes assumes the underlying follows geometric Brownian motion, volatility and rates are constant, and trading is continuous and frictionless so a position can be hedged perfectly. Under those assumptions there is a unique self-financing hedge, which forces a unique arbitrage-free price. The clean way to compute that price is to switch to the risk-neutral measure $\mathbb{Q}$, under which the discounted asset price is a martingale and the stock drifts at the risk-free rate net of dividends:

$$ dS_t = (r - q)\, S_t\, dt + \sigma\, S_t\, dW_t^{\mathbb{Q}}. $$

The solution is lognormal,

$$ S_T = S_0 \exp\!\Big( (r - q - \tfrac{1}{2}\sigma^2) T + \sigma \sqrt{T}\, Z \Big), \qquad Z \sim \mathcal{N}(0,1), $$

and the price of any European payoff is its discounted expectation under $\mathbb{Q}$:

$$ V_0 = e^{-rT}\, \mathbb{E}^{\mathbb{Q}}\!\left[\, \text{payoff}(S_T) \,\right]. $$

Black-Scholes evaluates this expectation in closed form; Monte Carlo evaluates it by sampling. Both are computing the same integral.

### 3.2 The Black-Scholes formula

With

$$ d_1 = \frac{\ln(S/K) + (r - q + \tfrac{1}{2}\sigma^2) T}{\sigma \sqrt{T}}, \qquad d_2 = d_1 - \sigma \sqrt{T}, $$

the call and put prices are

$$ C = S e^{-qT} \Phi(d_1) - K e^{-rT} \Phi(d_2), \qquad P = K e^{-rT} \Phi(-d_2) - S e^{-qT} \Phi(-d_1). $$

$\Phi$ is the standard normal CDF, implemented here through the Abramowitz and Stegun 7.1.26 polynomial for $\mathrm{erf}$ (maximum absolute error about $1.5\times 10^{-7}$), then $\Phi(x) = \tfrac{1}{2}(1 + \mathrm{erf}(x/\sqrt{2}))$. Measured against scipy on a 2401-point grid over $[-6, 6]$, the implemented $\Phi$ has a maximum absolute error of $6.97\times 10^{-8}$.

Put-call parity, $C - P = S e^{-qT} - K e^{-rT}$, holds to floating-point precision (residual about $10^{-14}$) because the from-scratch $\mathrm{erf}$ is exactly odd, so $\Phi(x) + \Phi(-x) = 1$ identically. Edge cases are handled without NaNs: $T \to 0$ and $\sigma \to 0$ both collapse to the discounted intrinsic value $\max(S e^{-qT} - K e^{-rT}, 0)$, which is the common limit of both degeneracies.

### 3.3 The Greeks

All formulas include the dividend yield $q$. Here $\phi$ is the normal PDF.

| Greek | Formula (call / put) | Shape across strike and time |
|---|---|---|
| Delta | $e^{-qT}\Phi(d_1)$ / $e^{-qT}(\Phi(d_1)-1)$ | S-curve in moneyness from $0$ to $e^{-qT}$; steepens as $T \to 0$. |
| Gamma | $\dfrac{e^{-qT}\phi(d_1)}{S\sigma\sqrt{T}}$ | Peaks at the money, spikes like $1/\sqrt{T}$ as $T \to 0$, vanishes in the wings at expiry. |
| Vega | $S e^{-qT}\phi(d_1)\sqrt{T}$ | Peaks at the money, grows with $\sqrt{T}$, so long-dated at-the-money is most vol-sensitive. |
| Theta | $-\dfrac{S\phi(d_1)\sigma e^{-qT}}{2\sqrt{T}} \mp \dots$ | Time decay, most negative for at-the-money near expiry (the mirror of gamma). |
| Rho | $K T e^{-rT}\Phi(d_2)$ / $-K T e^{-rT}\Phi(-d_2)$ | Grows with maturity; positive for calls, negative for puts. |
| Vanna | $-e^{-qT}\phi(d_1)\, d_2/\sigma$ | $\partial\Delta/\partial\sigma = \partial\text{Vega}/\partial S$; zero at the money, sign flips across it. |
| Volga | $\text{Vega}\cdot d_1 d_2/\sigma$ | $\partial\text{Vega}/\partial\sigma$; near zero at the money, positive in the wings (vega is convex in vol). |

Full theta:

$$ \Theta_{\text{call}} = -\frac{S\phi(d_1)\sigma e^{-qT}}{2\sqrt{T}} - rKe^{-rT}\Phi(d_2) + qSe^{-qT}\Phi(d_1), $$

$$ \Theta_{\text{put}} = -\frac{S\phi(d_1)\sigma e^{-qT}}{2\sqrt{T}} + rKe^{-rT}\Phi(-d_2) - qSe^{-qT}\Phi(-d_1). $$

Every one of these is validated against a central finite difference of a machine-precision reference price (see Section 3.5). The worst-case absolute disagreement over the test grid is $1.4\times 10^{-7}$ for delta, $3.1\times 10^{-6}$ for gamma, $5.3\times 10^{-8}$ for vega, $5.1\times 10^{-7}$ for theta, and $1.3\times 10^{-5}$ for rho, all far below the size of the Greeks themselves.

### 3.4 Monte Carlo

The estimator samples the exact terminal law and discounts the mean payoff:

$$ \hat{V} = e^{-rT}\,\frac{1}{N}\sum_{i=1}^{N} \text{payoff}(S_T^{(i)}), \qquad \text{SE} = \frac{\mathrm{std}(e^{-rT}\,\text{payoff})}{\sqrt{N}}, $$

with a 95% confidence interval $\hat{V} \pm 1.96\,\text{SE}$. The error decays like $1/\sqrt{N}$ regardless of dimension, which is the entire reason Monte Carlo is the tool of choice for high-dimensional and path-dependent problems.

**Antithetic variates.** Draw $N/2$ normals $Z$ and reuse $-Z$. Averaging the discounted payoff of each $(Z, -Z)$ pair cancels the component of the payoff that is odd in $Z$. Whenever the payoff is monotone in $Z$ (true for vanilla calls and puts) the two legs are negatively correlated and the pair average has lower variance. Measured factor for the at-the-money call: about $2.0\times$.

**Control variates.** Use the terminal stock $S_T$ as a control, whose mean is known exactly, $\mathbb{E}[S_T] = S_0 e^{(r-q)T}$. The estimator $D - c(S_T - \mathbb{E}[S_T])$ has the same mean as the discounted payoff $D$ for any $c$, and the variance-minimizing $c^{\star} = \mathrm{Cov}(D, S_T)/\mathrm{Var}(S_T)$ removes the part of $D$ that is linearly explained by $S_T$. Because a call payoff is strongly correlated with $S_T$, this cuts variance by about $6.9\times$ at the money (standard error about $2.6\times$ smaller, so roughly seven times fewer paths for the same confidence interval).

**Monte Carlo Greeks.** Delta uses the pathwise method: the payoff is almost surely differentiable in $S_0$ and $\partial S_T/\partial S_0 = S_T/S_0$, giving the call estimator $e^{-rT}\,\mathbb{E}[\mathbf{1}\{S_T > K\}\,S_T/S_0]$. Gamma has no pathwise estimator for vanilla payoffs (the second derivative of the payoff is a delta function), so it uses either the likelihood ratio (score) method or a finite-difference bump with common random numbers. The likelihood ratio weight is the second-order score of the lognormal density in $S_0$,

$$ \gamma_{\text{LR}} = e^{-rT}\,\mathbb{E}\!\left[\text{payoff}\cdot\left(\frac{Z^2 - 1}{S_0^2\sigma^2 T} - \frac{Z}{S_0^2\sigma\sqrt{T}}\right)\right]. $$

The common-random-numbers bump computes $[V(S+h) - 2V(S) + V(S-h)]/h^2$ where all three prices use the same draws $Z$. With independent draws this second difference has variance of order $1/h^2$ and is useless; sharing $Z$ makes the three estimators almost perfectly correlated, the simulation noise cancels in the numerator, and only the true curvature survives. All three MC Greeks land within one standard error of the analytical values.

### 3.5 A note on validating Greeks against finite differences

The Greeks are differenced against a machine-precision reference price built on the standard library's exact `math.erf`, not against the library's own price. The shipped price uses the Abramowitz and Stegun polynomial, whose derivative is not exactly the normal density $\phi$ that the analytical Greeks use. Differencing that price reintroduces (and for rho amplifies, through the large sensitivity $\partial d_2/\partial r \approx 9$ at long maturity and low vol) the polynomial's $10^{-7}$ wobble, which would mask the thing being tested: whether the closed-form Greek formulas are correct. Differencing an exact price isolates the formula check. Coverage is complete in tandem: the price tests show the shipped price matches the true price to better than $10^{-5}$ (about $8\times10^{-6}$ on the reference cases), and the Greek tests show the formulas match the exact derivatives of the true price to $10^{-7}$ to $10^{-5}$.

## 4. Results

### Gamma vs strike, by maturity
![Gamma vs strike](figures/gamma_vs_strike.png)

At one week to expiry (T = 0.05) gamma is a tall, narrow spike right at the strike, reaching about 0.089 at the money. As maturity lengthens the peak collapses and broadens: at one year the at-the-money gamma is only about 0.019. This is the visual statement that a short-dated at-the-money option's delta flips fastest as spot moves.

### Gamma vs time to expiry, by moneyness
![Gamma vs time](figures/gamma_vs_time.png)

The at-the-money line (blue) blows up as $T \to 0$, while the in- and out-of-the-money lines fall to zero. Away from the strike, the option becomes nearly all-or-nothing near expiry, so its delta stops changing and gamma vanishes. At the strike the opposite happens: delta is about to jump between 0 and 1, so gamma diverges like $1/\sqrt{T}$.

### Vega vs strike, by maturity
![Vega vs strike](figures/vega_vs_strike.png)

Vega peaks at the money and grows with maturity through the $\sqrt{T}$ factor: the one-year line tops out near 40 while the one-week line barely reaches 9. Long-dated at-the-money options carry the most volatility exposure, which is why they are the natural instruments for trading vol.

### Delta vs strike, by maturity
![Delta vs strike](figures/delta_vs_strike.png)

Call delta is a smooth S-curve in strike, running from about $e^{-qT}$ deep in the money to 0 deep out of the money. As expiry approaches the transition sharpens toward a step at the strike (the T = 0.05 line is nearly vertical at K = 100), because the option is collapsing into a digital bet on finishing in the money.

### Gamma surface and heatmap
![Gamma surface](figures/gamma_surface.png)
![Gamma heatmap](figures/gamma_heatmap.png)

The same object in 3D and as a 2D heatmap. The ridge along K = 100 rising sharply toward T = 0 is the at-the-money gamma spike; everywhere off the ridge the surface is low and flat. The interactive plotly versions (`figures/*_surface.html`) let you rotate the surfaces.

## 5. Black-Scholes vs Monte Carlo

Antithetic Monte Carlo with $N = 200{,}000$ paths against the closed form, for a call with $S = 100$, $r = 0.05$, $\sigma = 0.20$, $q = 0$:

| K | T | BS price | MC price | MC SE | abs error | BS in 95% CI |
|---:|---:|---:|---:|---:|---:|:--:|
| 90 | 0.25 | 11.6701 | 11.6598 | 0.0063 | 0.0103 | yes |
| 100 | 0.25 | 4.6150 | 4.6080 | 0.0106 | 0.0070 | yes |
| 110 | 0.25 | 1.1911 | 1.1884 | 0.0072 | 0.0028 | yes |
| 90 | 1 | 16.6994 | 16.6993 | 0.0195 | 0.0001 | yes |
| 100 | 1 | 10.4506 | 10.4276 | 0.0232 | 0.0230 | yes |
| 110 | 1 | 6.0401 | 6.0479 | 0.0222 | 0.0079 | yes |
| 100 | 2 | 16.1268 | 16.1281 | 0.0353 | 0.0013 | yes |
| 110 | 2 | 11.4555 | 11.4817 | 0.0360 | 0.0262 | yes |

Across the full 15-cell grid the Black-Scholes price falls inside the Monte Carlo 95% confidence interval in every cell. The convergence study confirms the rate:

![MC convergence](figures/mc_convergence.png)

The left panel shows the price with its 95% interval tightening onto the Black-Scholes line as $N$ grows from $10^2$ to $10^7$. The right panel plots the interval half-width on log-log axes against $N$; the fitted slope is $-0.492$, matching the theoretical $-1/2$. The error-bar view across strikes:

![BS vs MC error bars](figures/bs_vs_mc_errorbars.png)

**Timing and the tradeoff.** One Black-Scholes price takes about 17 microseconds and is exact. Monte Carlo costs about 0.18 ms at $N = 10^4$, 1.0 ms at $N = 10^5$, and 10 ms at $N = 10^6$, so it is hundreds of times slower for a single European price and carries $1/\sqrt{N}$ noise on top. What you buy for that cost is generality: Monte Carlo prices path-dependent and exotic payoffs (Asian, barrier, lookback, basket) and arbitrary dynamics that have no closed form, simply by changing the payoff or the simulation. Variance reduction recovers much of the speed gap, control variates cutting the at-the-money variance about seven times here. This is exactly the tradeoff an interviewer probes: closed form is fast and exact but rigid, Monte Carlo is slow and noisy but general.

## 6. Conventions

- **Vega** is reported per 1.00 change in volatility (a move from 0.20 to 1.20). The per-1% figure that traders quote is `vega_pct = vega / 100`, the price change for a vol move from 0.20 to 0.21.
- **Theta** is reported per year, as $\partial V/\partial t = -\partial V/\partial T$. The per-calendar-day figure is `theta_per_day = theta / 365`. Theta is normally negative for long options: holding all else fixed, an option loses value as expiry approaches.
- **Rho** is reported per 1.00 change in the rate. The per-1% figure is `rho_pct = rho / 100`.

## 7. Limitations and extensions

The model's assumptions are also its limitations. Volatility is constant, which the volatility smile observed in real markets directly contradicts; a stochastic-volatility model (Heston) or a local-volatility surface relaxes this. The terminal distribution is lognormal with no jumps; jump-diffusion (Merton) adds fat tails. Hedging is assumed continuous and frictionless, which ignores transaction costs and gap risk. Only European exercise is priced here; American options need a binomial or trinomial tree or Longstaff-Schwartz least-squares Monte Carlo for the early-exercise boundary. Path-dependent payoffs (Asian, barrier, lookback) need multi-step path simulation, which the Monte Carlo module is structured to extend to (the single-step terminal sampler becomes a loop-free `(N, steps)` array of increments).

## 8. References

- John C. Hull, *Options, Futures, and Other Derivatives*. Pricing, Greeks, and the parity and convention details.
- Paul Wilmott, *Paul Wilmott on Quantitative Finance*. The PDE and hedging view of Black-Scholes.
- Paul Glasserman, *Monte Carlo Methods in Financial Engineering*. The estimators, variance reduction, and the pathwise and likelihood-ratio Greek methods.
- Milton Abramowitz and Irene Stegun, *Handbook of Mathematical Functions*, formula 7.1.26. The erf approximation behind the normal CDF.

## Repository layout

```
options-pricing-visualizer/
  README.md  requirements.txt  pyproject.toml  LICENSE  .gitignore
  src/options_viz/
    normal.py          # standard normal PDF/CDF from scratch (A&S 7.1.26)
    black_scholes.py   # pricing, parity, implied vol (Newton + bisection)
    greeks.py          # analytical Greeks incl. vanna, volga
    monte_carlo.py     # MC pricer, antithetic, control variates, MC Greeks
    visualize.py       # 2D slices, 3D surfaces, heatmaps
    comparison.py      # BS vs MC tables, MC-Greek checks, timing
  scripts/
    generate_figures.py  # regenerate every figure into figures/
    run_comparison.py    # print BS vs MC tables, write comparison figures
  app/streamlit_app.py   # interactive dashboard (bonus)
  tests/                 # 1763 tests (normal, BS, Greeks, MC, comparison, app)
  figures/               # generated PNGs and interactive HTML surfaces
  results/               # generated comparison table (markdown)
```
