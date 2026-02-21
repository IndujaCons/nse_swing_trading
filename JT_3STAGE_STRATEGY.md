# JT 3-Stage Scale-Out Strategy

## Overview

Momentum dip-buying strategy on Nifty 100 stocks combining two signal types (J, T) with a 3-stage partial exit system. Capital: 20L, 2L per trade, 2 entries/day.

---

## Entry Signals

### Strategy J — Weekly Support Bounce
- **Condition**: Close within 0-3% above weekly support level, IBS > 0.5, green candle, CCI(20) > -100
- **Stop**: Below weekly low (support break)
- **Edge**: Defined structural support gives tight risk

### Strategy T — Keltner Channel Pullback
- **Condition**: Price within 1% of EMA(20), stock touched upper Keltner band (EMA20 + 2x ATR14) in last 10 bars, green candle
- **Stop**: 5% hard SL
- **Edge**: Buying a pullback in a confirmed uptrend (was recently at upper band = strong momentum)

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

## Backtest Results — 11 Years (2015-2025), Nifty 100

### 3-Stage (Production)

| Year | Trades | WR% | Return% | P&L | Avg Win | Avg Loss |
|---|---|---|---|---|---|---|
| 2025 | 182 | 65.4% | 11.03% | 2,20,676 | 6,873 | -9,479 |
| 2024 | 259 | 70.3% | 14.59% | 2,91,850 | 6,210 | -10,888 |
| 2023 | 224 | 80.4% | 42.25% | 8,44,971 | 7,055 | -9,659 |
| 2022 | 252 | 71.4% | 25.37% | 5,07,399 | 7,404 | -11,462 |
| 2021 | 297 | 74.1% | 31.73% | 6,34,692 | 6,612 | -10,648 |
| 2020 | 289 | 73.7% | 29.39% | 5,87,813 | 7,323 | -12,788 |
| 2019 | 239 | 69.0% | 7.86% | 1,57,234 | 6,093 | -11,460 |
| 2018 | 242 | 57.9% | -8.80% | -1,75,922 | 6,675 | -10,887 |
| 2017 | 205 | 82.9% | 40.91% | 8,18,151 | 6,851 | -9,901 |
| 2016 | 234 | 70.5% | 16.69% | 3,33,748 | 6,833 | -11,502 |
| 2015 | 255 | 65.5% | 2.15% | 43,004 | 5,968 | -10,837 |
| **Total** | **2,678** | **71.0%** | **213.18%** | **42,63,617** | | |
| **Avg/yr** | | | **19.38%** | **3,87,602** | | |

- Winning years: 10/11
- Best year: 42.25% (2023)
- Worst year: -8.80% (2018)

### Original (1/2-1/2) for Reference

| Year | Trades | WR% | Return% | P&L | Avg Win | Avg Loss |
|---|---|---|---|---|---|---|
| 2025 | 171 | 63.2% | 8.83% | 1,76,610 | 7,075 | -9,325 |
| 2024 | 230 | 66.5% | 10.68% | 2,13,587 | 6,713 | -10,565 |
| 2023 | 207 | 78.7% | 36.98% | 7,39,594 | 7,102 | -9,501 |
| 2022 | 236 | 69.5% | 20.75% | 4,15,073 | 7,468 | -11,244 |
| 2021 | 264 | 70.8% | 26.17% | 5,23,481 | 7,036 | -10,288 |
| 2020 | 256 | 70.3% | 26.01% | 5,20,160 | 8,241 | -12,673 |
| 2019 | 213 | 67.6% | 8.14% | 1,62,704 | 6,384 | -10,965 |
| 2018 | 230 | 55.7% | -9.81% | -1,96,262 | 6,904 | -10,588 |
| 2017 | 188 | 81.4% | 37.01% | 7,40,260 | 7,035 | -9,605 |
| 2016 | 219 | 68.5% | 12.12% | 2,42,447 | 6,737 | -11,132 |
| 2015 | 234 | 60.7% | -3.85% | -76,989 | 6,417 | -10,741 |
| **Total** | **2,448** | **68.3%** | **173.03%** | **34,60,666** | | |
| **Avg/yr** | | | **15.73%** | **3,14,606** | | |

- Winning years: 9/11
- Best year: 37.01% (2017)
- Worst year: -9.81% (2018)

### 3-Stage vs Original Summary

| Metric | Original | 3-Stage | Improvement |
|---|---|---|---|
| Total Return | 173.03% | 213.18% | +40.15% |
| Avg Return/yr | 15.73% | 19.38% | +3.65% |
| Win Rate | 68.3% | 71.0% | +2.7% |
| Total P&L | 34.6L | 42.6L | +8.0L |
| Winning Years | 9/11 | 10/11 | +1 |
| Total Trades | 2,448 | 2,678 | +230 |

3-Stage beat Original in 9 out of 11 individual years.

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

- 2018 is the worst year for both variants — broad market selloff (Oct 2018 Nifty crash). Both systems lost money.
- Random trade selection causes ~20-30% variance in annual returns between runs. Live results will differ from any single backtest run.
- Results are based on daily closing prices. Live execution at different prices will cause slippage.
