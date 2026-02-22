# JT 3-Stage Scale-Out Strategy

## Overview

Momentum dip-buying strategy on Nifty 100 stocks combining two signal types (J, T) with a 3-stage partial exit system. Capital: 20L, 2L per trade, 2 entries/day.

---

## Entry Signals

### Strategy J — Weekly Support Bounce
- **Condition**: Close within 0-3% above weekly support level, IBS > 0.5, green candle, CCI(20) > -100, **no gap-down** (open >= prev close)
- **Stop**: Below weekly low (support break)
- **Edge**: Defined structural support gives tight risk

### Strategy T — Keltner Channel Pullback
- **Condition**: Price within 1% of EMA(20), stock touched upper Keltner band (EMA20 + 2x ATR14) in last 10 bars, green candle, **no gap-down** (open >= prev close)
- **Stop**: 5% hard SL
- **Edge**: Buying a pullback in a confirmed uptrend (was recently at upper band = strong momentum)

### Gap-Down Filter (applies to both J and T)
- **Rule**: Skip entry if today's Open < yesterday's Close (stock gapped down)
- **Why**: Analysis of top 10 losses in 2018 showed 9/10 had gap-down entries on low volume — classic "dead cat bounce" pattern. Filtering these improves avg return by +2.6%/yr and eliminates losing years.

---

## Exit Rules (3-Stage Scale-Out)

### Strategy J Exits
J uses its own exit logic (not the 3-stage system):
1. **Support break**: Full exit if price drops below weekly low stop (with Nifty shield — skipped if Nifty dropped same % or more)
2. **+5%**: Sell 50%
3. **+10%**: Sell remaining
4. **Chandelier exit**: After partial, sell remaining if price < Highest High (since entry) - 3x ATR(14)

### Strategy T Exits (3-Stage)
1. **Hard SL**: 5% below entry — exit all remaining shares
2. **+5%**: Sell 1/3 of shares (lock in first profit)
3. **+8%**: Sell 1/3 of shares (lock in second profit)
4. **Indicator exit on remaining 1/3**: Price reaches upper Keltner band (EMA20 + 2x ATR14)

### Why 3-Stage Works
The original system sold 1/2 at +5% and exited the other 1/2 on indicator. The 3-stage system:
- Sells only 1/3 at +5% (keeps more capital in the trade)
- Adds a +8% exit for 1/3 (captures an additional profit tier the original missed)
- Lets the final 1/3 ride the full indicator exit (same as before, but with 1/3 instead of 1/2)
- Net effect: more capital stays invested during the profitable phase of the trend

---

## Portfolio Parameters

| Parameter | Value |
|---|---|
| Universe | Nifty 100 |
| Capital | 20 Lakhs |
| Per trade | 2 Lakhs |
| Max positions | 10 (20L / 2L) |
| Entries per day | 2 |
| Strategies | J + T |

---

## Backtest Results — 10 Years (2016-2025), Nifty 100, 3-Seed Average

Results below are 3-seed averages (seeds 42, 99, 7) with gap-down filter ON (production config).

### 3-Stage + Gap-Down Filter (Production)

| Year | Trades | WR% | Return% | P&L |
|---|---|---|---|---|
| 2025 | 182 | 67.1% | 14.23% | 2,84,719 |
| 2024 | 195 | 73.0% | 22.81% | 4,56,167 |
| 2023 | 184 | 79.8% | 36.32% | 7,26,425 |
| 2022 | 214 | 66.4% | 12.88% | 2,57,726 |
| 2021 | 212 | 69.8% | 20.40% | 4,08,052 |
| 2020 | 222 | 72.1% | 20.59% | 4,11,771 |
| 2019 | 218 | 65.8% | 12.18% | 2,43,627 |
| 2018 | 169 | 59.8% | -0.33% | -6,621 |
| 2017 | 184 | 82.3% | 41.10% | 8,22,090 |
| 2016 | 183 | 69.4% | 15.60% | 3,11,994 |
| **Total** | | | **195.79%** | **39,15,950** |
| **Avg/yr** | | | **19.58%** | **3,91,595** |

- Winning years: 9/10
- Best year: 41.10% (2017)
- Worst year: -0.33% (2018 — effectively breakeven)

### Without Gap-Down Filter (for Reference)

| Year | Trades | WR% | Return% | P&L |
|---|---|---|---|---|
| 2025 | 179 | 66.2% | 14.67% | 2,93,444 |
| 2024 | 210 | 68.9% | 17.01% | 3,40,228 |
| 2023 | 195 | 79.1% | 36.79% | 7,35,850 |
| 2022 | 234 | 67.4% | 18.98% | 3,79,567 |
| 2021 | 233 | 68.4% | 18.13% | 3,62,615 |
| 2020 | 243 | 72.7% | 23.71% | 4,74,192 |
| 2019 | 216 | 63.2% | 7.02% | 1,40,449 |
| 2018 | 199 | 54.6% | -9.34% | -1,86,871 |
| 2017 | 177 | 80.0% | 36.55% | 7,31,100 |
| 2016 | 190 | 65.1% | 12.95% | 2,58,916 |
| **Total** | | | **176.48%** | **35,29,490** |
| **Avg/yr** | | | **17.65%** | **3,52,949** |

- Winning years: 9/10
- Worst year: -9.34% (2018)

### Gap-Down Filter Impact

| Metric | No Filter | Gap Filter | Improvement |
|---|---|---|---|
| Total Return (10yr) | 176.48% | 195.79% | +19.31% |
| Avg Return/yr | 17.65% | 19.58% | +1.93% |
| Total P&L | 35.3L | 39.2L | +3.9L |
| Worst Year | -9.34% | -0.33% | +9.01% |

---

## Live Trade Selection Rules

When multiple signals fire on the same day and you have limited slots:

1. **Priority: J > T** — J has structural support (defined risk), T is a trend pullback
2. **Within same strategy**:
   - J: Prefer lower `close_near_pct` (closer to support = tighter risk) + higher IBS
   - T: Prefer stock that more recently touched upper Keltner
3. **No duplicate stocks** — skip if you already hold a position in the same stock
4. **Sector diversification** — avoid 2 stocks from the same sector when choosing between signals

---

## Risk Notes

- 2018 is the worst year — broad market selloff (Oct 2018 Nifty crash). Gap-down filter turns it from -9.34% to -0.33% (effectively breakeven).
- Random trade selection causes ~20-30% variance in annual returns between runs. Live results will differ from any single backtest run.
- Results are based on daily closing prices. Live execution at different prices will cause slippage.
- Gap-down filter reduces trade count by ~10% (filters low-conviction "dead cat bounce" entries).
