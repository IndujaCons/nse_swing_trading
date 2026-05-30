# Kelly Criterion Sizing Study — Mom20

**Date:** 2026-05-30  
**Conclusion: Equal weight is optimal. Kelly sizing rejected.**

---

## Objective

Test whether Kelly criterion position sizing improves Mom20 returns vs equal weight, using the same basket selection (top 20, buffer 15/40, β≤1.2, monthly, Nifty200 PIT).

## Kelly Variants Tested

| Variant | Formula |
|---|---|
| **K-12m** | f\* = ret₁₂m / σ₁₂² |
| **K-Blend** | f\* = 0.5 × (ret₁₂m / σ₁₂²) + 0.5 × (ret₃m / σ₃m²) |
| **K-3m** | f\* = ret₃m / σ₃m² |

- σ₁₂ = annualized vol from 252-day log-return std dev  
- σ₃m = annualized vol from 63-day log-return std dev  
- All f\* floored at 0, normalized across basket, capped at 3× equal weight, floored at 0.3× equal weight, then re-normalized  
- Only **new entries** are Kelly-sized; existing holds are not resized at rebalance  
- Basket selection identical across all portfolios — only capital per slot differs

## Results (11yr PIT, Jan 2015 → Apr 2026, ₹20L start)

| Metric | Equal Wt | K-12m | K-Blend | K-3m |
|---|---|---|---|---|
| **Final NAV** | **₹3.19Cr** | ₹2.22Cr | ₹2.58Cr | ₹2.11Cr |
| **CAGR** | **27.8%** | 23.7% | 25.4% | 23.2% |
| Total Return | **1493%** | 1009% | 1191% | 957% |
| Max Drawdown | **23.7%** | 24.1% | 26.6% | 26.7% |
| Total Trades | 442 | 481 | 425 | 337 |
| Win Rate | **57.2%** | 53.6% | 56.2% | 53.4% |
| Profit Factor | **4.00** | 3.44 | 3.67 | 3.62 |
| Avg Hold (days) | 158 | 149 | 159 | 151 |
| Negative Years | **2** | 3 | 3 | 4 |

## Per-Year Comparison

| Year | Equal Wt | K-12m | K-Blend | K-3m |
|---|---|---|---|---|
| 2015 | +13.2% | +7.7% | +8.1% | **+22.8%** |
| 2016 | +11.9% | **+19.4%** | +17.9% | −4.8% |
| 2017 | +49.3% | **+57.0%** | **+57.2%** | +45.6% |
| 2018 | −1.8% | −3.4% | −2.6% | **−1.5%** |
| 2019 | **+16.1%** | +11.9% | +13.3% | +14.1% |
| 2020 | **+27.7%** | +18.5% | +23.6% | +25.9% |
| 2021 | **+84.6%** | +65.5% | +76.5% | +60.4% |
| 2022 | **+39.9%** | +27.6% | +30.6% | +39.3% |
| 2023 | +42.9% | +43.3% | **+47.0%** | +39.7% |
| 2024 | **+55.0%** | +50.0% | +49.2% | +50.1% |
| 2025 | **+2.8%** | −1.5% | −3.9% | −3.3% |
| 2026 | **−1.7%** | −4.2% | −2.6% | −2.9% |
| **Neg years** | **2** | 3 | 3 | 4 |

Bold = best performer that year.

## Key Findings

1. **Equal weight wins 8 of 12 years** and dominates on all aggregate metrics.

2. **Kelly hurts in the big compounding years (2020, 2021, 2022).** These are precisely the years where the strategy generates the most alpha. Kelly overweights high-momentum names going in — those stocks then mean-revert hardest after a strong run, causing underperformance vs equal weight.

3. **K-3m is the worst overall** (4 negative years, lowest CAGR) but wins strongly in 2015 (+22.8%). Short-horizon Kelly is too reactive — it concentrates into recent movers who are often at peak.

4. **K-Blend is the best Kelly variant** (CAGR 25.4%, 3 neg years) but still trails equal weight by −2.4% CAGR over 11 years and has worse max drawdown (+2.9pp).

5. **Kelly's edge disappears on a momentum basket.** Kelly's μ/σ² formula rewards the same signal (past return / vol) that already drove stock selection. Overweighting within an already-ranked high-momentum basket adds no information — it just concentrates risk.

6. **Equal weight gets more out of equal diversification.** With only 20 slots, each position is already meaningful. Spreading evenly lets the portfolio capture the full compounding of the basket's winners without betting more on what's already peaked.

## Conclusion

Kelly criterion position sizing does not improve Mom20. **Equal weight (NAV/20 per new entry) remains the production sizing rule.** No change to strategy.

Script used: `/tmp/kelly_vs_equal_mom20.py` (not committed — research only).
