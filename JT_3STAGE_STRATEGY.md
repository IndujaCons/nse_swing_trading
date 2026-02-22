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
2. **+6%**: Sell 1/3 of shares (lock in first profit)
3. **+10%**: Sell 1/3 of shares (lock in second profit)
4. **Indicator exit on remaining 1/3**: Price reaches upper Keltner band (EMA20 + 2x ATR14)

### Why 3-Stage with 6/10 Targets
The original system sold 1/2 at +5% and exited the other 1/2 on indicator. The 3-stage system with wider targets:
- Sells only 1/3 at +6% (keeps more capital in the trade, lets winners run further)
- Adds a +10% exit for 1/3 (captures a much larger profit tier)
- Lets the final 1/3 ride the full indicator exit
- Wider targets (6/10 vs 5/8) add +1.99%/yr and increase avg win by Rs 899/trade (+12%)

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

### Production Config (3-Stage 6/10 + Gap-Down + Risk Ranking)

| Year | Trades | WR% | Return% | P&L | AvgWin | AvgLoss |
|---|---|---|---|---|---|---|
| 2025 | 167 | 61.3% | 13.34% | 2,66,785 | 7,778 | -8,176 |
| 2024 | 189 | 69.7% | 23.54% | 4,70,866 | 7,892 | -9,992 |
| 2023 | 173 | 79.0% | 37.78% | 7,55,691 | 8,193 | -10,013 |
| 2022 | 206 | 63.8% | 16.54% | 3,30,866 | 8,545 | -10,635 |
| 2021 | 194 | 67.8% | 23.68% | 4,73,693 | 8,905 | -11,204 |
| 2020 | 216 | 70.3% | 22.35% | 4,47,001 | 8,914 | -14,157 |
| 2019 | 210 | 65.1% | 16.24% | 3,24,805 | 8,124 | -10,757 |
| 2018 | 175 | 57.5% | 0.55% | 10,974 | 7,988 | -10,675 |
| 2017 | 172 | 79.1% | 45.01% | 9,00,131 | 9,291 | -10,202 |
| 2016 | 158 | 66.3% | 14.88% | 2,97,472 | 8,108 | -10,396 |
| **Total** | | | **213.91%** | **42,78,284** | | |
| **Avg/yr** | | | **21.39%** | **4,27,828** | **8,374** | **-10,621** |

- Winning years: 10/10
- Best year: 45.01% (2017)
- Worst year: 0.55% (2018 — breakeven)

### Baseline (Random Shuffle, No Filters) for Reference

| Metric | Baseline | Production | Improvement |
|---|---|---|---|
| Avg Return/yr | 17.65% | 21.39% | +3.74% |
| Avg Win | 7,486 | 8,374 | +Rs 888 per trade |
| Avg Loss | -11,115 | -10,621 | +Rs 494 smaller |
| Worst Year | -9.34% | +0.55% | No losing years |
| Total P&L (10yr) | 35.3L | 42.8L | +7.5L |

Improvements from: gap-down filter (+1.93%/yr) + risk ranking (+1.08%/yr) + wider T targets 6/10 (+1.99%/yr, +Rs 888 avg win).

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

- 2018 is the worst year — broad market selloff (Oct 2018 Nifty crash). Gap-down filter + risk ranking turns it from -9.34% to +0.55%.
- Risk ranking by stop distance reduces avg loss by Rs 687/trade and adds +1.08%/yr vs random selection.
- Results are 3-seed averages. Live results will differ from any single backtest run.
- Results are based on daily closing prices. Live execution at different prices will cause slippage.
- Gap-down filter reduces trade count by ~10% (filters low-conviction "dead cat bounce" entries).
