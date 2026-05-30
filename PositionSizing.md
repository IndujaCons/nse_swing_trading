# Position Sizing Study — Mom20 (Nifty200, Monthly Rebalance)

**Date:** 2026-05-30  
**Fixed params across all tests:** Top 20 | Buffer 15/40 | Sector cap ≤4 | Regime SMA200 | PIT 11yr (Jan 2015 → Apr 2026) | ₹20L start

---

## Approaches Tested

### 1. Kelly Criterion (3 variants)
`f* = μ / σ²`, normalized across basket, capped at 3× equal weight, floored at 0.3× equal weight.

| Variant | Formula |
|---|---|
| K-12m | f\* = ret₁₂m / σ₁₂² |
| K-Blend | f\* = 0.5×(ret₁₂m/σ₁₂²) + 0.5×(ret₃m/σ₃m²) |
| K-3m | f\* = ret₃m / σ₃m² |

**All Kelly variants underperform equal weight.** Best Kelly was K-Blend at 25.4% CAGR vs equal weight's 28.5%.  
See `KellyStudy.md` for full Kelly results.

---

### 2. Z-Score Proportional (Z-Prop)

`w_i = max(wz_i, 0) / Σ max(wz_j, 0)`

Where `wz` is the raw weighted z-score recovered from `norm_score`:
- If `norm_score ≥ 1`: `wz = norm_score − 1`
- If `norm_score < 1`: `wz = 1 − 1/norm_score`

Weights are then clipped to [floor, cap] and re-normalized.

**Why Z-Prop instead of Kelly:** Kelly uses return/σ² which overweights high-momentum/high-vol names — precisely the ones that mean-revert hardest. Z-Prop amplifies the same signal used for selection (the z-score), without the volatility denominator that causes over-concentration.

---

## Z-Prop Results: Band Comparison (β≤1.2, sector cap ≤4)

| Metric | Equal Wt | ZP 1.5%/15% | **ZP 2.5%/10%** |
|---|---|---|---|
| **CAGR** | 29.2% | 28.7% | **30.2%** |
| **Final NAV** | ₹3.62Cr | ₹3.47Cr | **₹3.96Cr** |
| Total Return | 1709% | 1636% | **1879%** |
| Max Drawdown | **22.0%** | 23.1% | 22.3% |
| Total Trades | 429 | 357 | 336 |
| Win Rate | **58.7%** | 55.2% | 58.3% |
| Profit Factor | **4.24** | 3.72 | 4.19 |
| Avg Hold (days) | 161 | 163 | **172** |
| Negative Years | **1** | 2 | 2 |

### Year-by-Year: Equal Wt vs ZP 2.5%/10% (with β≤1.2)

| Year | Equal% | ZP 2.5/10% | Δ | Best |
|---|---|---|---|---|
| 2015 | +13.2% | +5.7% | −7.5% | Eq |
| 2016 | +15.3% | +16.0% | +0.7% | ZP |
| 2017 | +51.8% | +60.0% | +8.2% | ZP |
| 2018 | +0.3% | −4.5% | −4.8% | Eq |
| 2019 | +17.3% | +15.0% | −2.3% | Eq |
| 2020 | +30.5% | +22.9% | −7.6% | Eq |
| 2021 | +78.7% | +98.2% | **+19.5%** | ZP |
| 2022 | +42.7% | +50.0% | +7.3% | ZP |
| 2023 | +42.6% | +56.5% | **+13.8%** | ZP |
| 2024 | +59.0% | +56.5% | −2.5% | Eq |
| 2025 | +3.9% | +5.0% | +1.2% | ZP |
| 2026 | −1.0% | −2.2% | −1.3% | Eq |
| **Years won** | **6** | **6** | | |
| **Neg years** | **1** | **2** | | |

**Pattern:** Equal weight wins in choppy/down years (2018, 2019, 2020, 2024). Z-Prop wins large in strong trending years (2021 +19.5pp, 2023 +13.8pp, 2017 +8.2pp). Z-Prop's wins are larger in magnitude than equal weight's wins — that asymmetry drives the +1% CAGR advantage over 11 years.

---

## Effect of Removing Beta Cap (β≤1.2 → no cap)

With no beta cap, Z-Prop **underperforms** equal weight significantly:

| Metric | Equal Wt (no β cap) | ZP 2.5/10% (no β cap) |
|---|---|---|
| CAGR | **27.7%** | 25.5% |
| Negative Years | **1** | 3 |
| Years Won | **8** | 3 |

**The β≤1.2 cap is essential for Z-Prop to work.** Without it, high-beta stocks enter with inflated z-scores (high return, not quality). Z-Prop overweights them; they mean-revert; equal weight spreads the damage. The cap filters out noisy high-beta names, leaving a universe where z-score is a reliable sizing signal.

---

## Academic Basis

- **Z-Prop** is derived from signal-proportional weighting used internally in Asness, Moskowitz & Pedersen (2013) *"Value and Momentum Everywhere"* (JFE).
- Other approaches tested: **Inverse Volatility** (Kirby & Ostdiek 2012) — tested and rejected below.
- Not yet tested: **Volatility-Managed Momentum** (Barroso & Santa-Clara 2015).

---

## 3. Inverse Volatility

`w_i = (1/σ_i) / Σ(1/σ_j)`, same 2.5%/10% band for fair comparison.

**Clearly rejected.** CAGR 23.2% vs equal weight 29.0%. Lost 2021 catastrophically (+43% vs +80%) because it underweights high-vol momentum winners by design — the opposite of what you want in a strong trend year. Wins only in low-vol defensive years.

| Metric | Equal Wt | ZP 2.5/10% | InvVol 2.5/10% |
|---|---|---|---|
| CAGR | 29.0% | 28.2%* | **23.2%** |
| 2021 | +80.5% | +91.7% | +43.0% |
| 2022 | +41.8% | +42.6% | +20.4% |
| Years Won | **6** | 2 | 2 |

*ZP result varied 28–31% across runs due to yfinance data pull variance.

---

## Conclusions

1. **Equal weight remains the production rule** — simplest, most robust across runs, 1 negative year, best risk-adjusted stats.

2. **ZP 2.5%/10% shows a marginal edge** (+0 to +1.5% CAGR depending on run). Not definitive at 12 annual data points. Best tested alternative if sizing is ever revisited.

3. **ZP edge is β-cap dependent** — remove β≤1.2 and equal weight wins 8/12 years. The cap is load-bearing for any non-equal sizing.

4. **InvVol is rejected** — fights the selection signal, destroys trending year returns.

5. **Kelly is rejected** — all three variants underperform. Full Kelly study in `KellyStudy.md`.

6. **Next step if pursuing ZP**: lock the cache with `--refresh` once, run a single canonical comparison, treat that as the definitive result.

---

## Parameters for ZP 2.5%/10% (if adopted)

```
Basket selection:  identical to Mom20 production (top20, buffer 15/40, β≤1.2, sector cap ≤4, SMA200 regime)
Weight formula:    w_i = max(wz_i, 0) / Σ max(wz_j, 0)
wz recovery:       wz = norm_score − 1  (if norm_score ≥ 1)
                   wz = 1 − 1/norm_score  (if norm_score < 1)
Floor:             2.5% (= 0.5× equal weight)
Cap:               10.0% (= 2× equal weight)
Applies to:        new entries only — existing holds not resized at rebalance
```

Script: `/tmp/kelly_vs_equal_mom20.py` (not committed — research only)
