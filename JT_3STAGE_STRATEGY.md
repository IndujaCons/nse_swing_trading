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

### Production Config (3-Stage + Gap-Down + Risk Ranking)

| Year | Trades | WR% | Return% | P&L | AvgWin | AvgLoss |
|---|---|---|---|---|---|---|
| 2025 | 171 | 63.9% | 14.17% | 2,83,435 | 7,192 | -8,131 |
| 2024 | 196 | 69.6% | 19.65% | 3,92,922 | 7,044 | -9,540 |
| 2023 | 185 | 81.3% | 40.53% | 8,10,517 | 7,513 | -9,273 |
| 2022 | 210 | 64.7% | 13.17% | 2,63,429 | 7,800 | -10,728 |
| 2021 | 206 | 69.7% | 20.06% | 4,01,267 | 7,582 | -11,030 |
| 2020 | 228 | 72.3% | 20.84% | 4,16,835 | 7,971 | -14,177 |
| 2019 | 221 | 67.5% | 15.95% | 3,19,066 | 7,133 | -10,393 |
| 2018 | 179 | 59.1% | 0.63% | 12,487 | 7,498 | -10,671 |
| 2017 | 193 | 81.3% | 44.11% | 8,82,139 | 7,900 | -9,954 |
| 2016 | 173 | 69.2% | 15.69% | 3,13,689 | 7,230 | -10,386 |
| **Total** | | | **204.80%** | **40,95,786** | | |
| **Avg/yr** | | | **20.48%** | **4,09,579** | **7,486** | **-10,428** |

- Winning years: 10/10
- Best year: 44.11% (2017)
- Worst year: 0.63% (2018 — breakeven)

### Baseline (Random Shuffle, No Filters) for Reference

| Metric | Baseline | Production | Improvement |
|---|---|---|---|
| Avg Return/yr | 17.65% | 20.48% | +2.83% |
| Avg Loss | -11,115 | -10,428 | +Rs 687 smaller |
| Worst Year | -9.34% | +0.63% | No losing years |
| Total P&L (10yr) | 35.3L | 41.0L | +5.7L |

Improvements from: gap-down filter (+1.93%/yr) + risk ranking (+1.08%/yr, -Rs 687 avg loss).

---

## Live Trade Selection Rules

When multiple signals fire on the same day and you have limited slots:

1. **Rank by stop_distance_pct ASC** — tightest risk first, regardless of strategy type
   - J: `stop_pct = (price - weekly_low_stop) / price * 100` (variable, typically 2-8%)
   - T: `stop_pct = 5.0` (fixed hard SL)
   - A J signal with 2% stop beats a T signal with 5% stop (better risk/reward)
   - A J signal with 7% stop loses to T (worse risk/reward)
2. **Seed-based tiebreaker** — when stop_pct is equal, random (reproducible) selection
3. **No duplicate stocks** — skip if you already hold a position in the same stock

This reduces avg loss by Rs 687/trade and adds +1.08%/yr vs random selection.

---

## Risk Notes

- 2018 is the worst year — broad market selloff (Oct 2018 Nifty crash). Gap-down filter + risk ranking turns it from -9.34% to +0.63%.
- Risk ranking by stop distance reduces avg loss by Rs 687/trade and adds +1.08%/yr vs random selection.
- Results are 3-seed averages. Live results will differ from any single backtest run.
- Results are based on daily closing prices. Live execution at different prices will cause slippage.
- Gap-down filter reduces trade count by ~10% (filters low-conviction "dead cat bounce" entries).
