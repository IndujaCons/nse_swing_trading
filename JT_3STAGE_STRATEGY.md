# JT 3-Stage Scale-Out Strategy

## Overview

Momentum dip-buying strategy on Nifty 100 stocks combining two signal types (J, T) with a 3-stage partial exit system. Capital: 20L, 2L per trade, 2 entries/day.

---

## Entry Signals

### Strategy J — Weekly Support Bounce
- **Condition**: Close within 0-3% above weekly support level, IBS > 0.5, green candle, CCI(20) > -100, **no gap-down** (open >= prev close)
- **Support calculation**: 26-week rolling min of weekly close, **skipping the last 2 weeks** (current + previous). Uses proven support levels, not recent noise.
- **Stop**: Below weekly low (support break), also skipping last 2 weeks
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

## Signal Ranking — ATR% (Lowest Volatility First)

When multiple signals fire on the same day and you have limited slots (2/day):

### Current Method: Rank by ATR% (ATR14 / Price)
- **Sort**: Lowest `atr_norm` first — prefer calmer, less volatile stocks
- **Logic**: Lower volatility stocks are less likely to gap through stops and more likely to have orderly pullbacks that recover to targets
- **Tiebreaker**: Seed-based random jitter for reproducibility

### Why ATR% over Stop%
Four ranking strategies were tested across 11 years (2015-2025):

| Rank Method | Total PnL | Avg Ret/yr | Positive Yrs | Avg WR | Avg PF |
|---|---|---|---|---|---|
| **B: ATR% (current)** | **+43.88L** | **+19.95%** | **11/11** | **67.1%** | **1.79** |
| A: Stop% (old) | +40.40L | +18.36% | 9/11 | 66.6% | 1.79 |
| C: Trend/Risk | +42.42L | +19.28% | 10/11 | 68.5% | 1.75 |
| D: Composite | +42.48L | +19.31% | 11/11 | 67.8% | 1.82 |

ATR% ranking:
- **+3.48L more** than old Stop% ranking over 11 years
- **Eliminates both losing years** (2015: -2.0% -> +0.6%, 2018: -2.6% -> +5.0%)
- Entry conditions already filter for trend quality, so ranking by volatility avoids double-counting
- Slight weakness in very strong trending markets (2019, 2023) where volatile runners get skipped

### Stop% is still displayed
The UI shows both ATR% (ranking column) and Stop% (SL distance) in the Top Picks panel so you can see the risk profile of each signal.

---

## Backtest Results — 11 Years (2015-2025), Nifty 100

### Production Config (3-Stage 6/10 + Gap-Down + ATR% Ranking)

| Year | Trades | J | T | Win | Loss | WR% | P&L | Return% | AvgWin | AvgLoss | PF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025 | 164 | 97 | 67 | 110 | 54 | 67.1% | +4,64,706 | +23.24% | 8,262 | -8,225 | 2.05 |
| 2024 | 187 | 74 | 113 | 129 | 58 | 69.0% | +5,48,242 | +27.41% | 8,299 | -9,006 | 2.05 |
| 2023 | 159 | 68 | 91 | 125 | 34 | 78.6% | +6,72,189 | +33.61% | 8,017 | -9,703 | 3.04 |
| 2022 | 195 | 103 | 92 | 125 | 70 | 64.1% | +2,49,805 | +12.49% | 8,310 | -11,271 | 1.32 |
| 2021 | 185 | 37 | 148 | 125 | 60 | 67.6% | +5,09,796 | +25.49% | 9,380 | -11,045 | 1.77 |
| 2020 | 206 | 66 | 140 | 143 | 63 | 69.4% | +3,63,722 | +18.19% | 8,547 | -13,626 | 1.42 |
| 2019 | 185 | 99 | 86 | 111 | 74 | 60.0% | +87,619 | +4.38% | 7,795 | -10,508 | 1.11 |
| 2018 | 157 | 95 | 62 | 98 | 59 | 62.4% | +1,01,641 | +5.08% | 7,721 | -11,102 | 1.16 |
| 2017 | 164 | 69 | 95 | 134 | 30 | 81.7% | +9,97,931 | +49.90% | 9,856 | -10,761 | 4.09 |
| 2016 | 167 | 77 | 90 | 114 | 53 | 68.3% | +3,81,051 | +19.05% | 8,208 | -10,465 | 1.69 |
| 2015 | 180 | 106 | 74 | 103 | 77 | 57.2% | +11,496 | +0.57% | 7,923 | -10,450 | 1.01 |
| **Total** | **1,949** | **891** | **1,058** | **1,317** | **632** | | **+43,88,199** | | | | |
| **Avg/yr** | **177** | | | | | **67.8%** | **+3,98,927** | **+19.95%** | **8,393** | **-10,560** | **1.97** |

- **Positive years: 11/11**
- **Best year: +49.90% (2017)**
- **Worst year: +0.57% (2015)**
- **Total P&L: Rs 43.88 Lakhs on 20L capital over 11 years**

### Comparison: Old Stop% Ranking (for reference)

| Year | Old (Stop%) | New (ATR%) | Difference |
|---|---|---|---|
| 2025 | +15.03% | +23.24% | +8.21% |
| 2024 | +17.03% | +27.41% | +10.38% |
| 2023 | +38.51% | +33.61% | -4.90% |
| 2022 | +16.07% | +12.49% | -3.58% |
| 2021 | +20.88% | +25.49% | +4.61% |
| 2020 | +23.46% | +18.19% | -5.27% |
| 2019 | +19.36% | +4.38% | -14.98% |
| 2018 | **-2.68%** | **+5.08%** | +7.76% |
| 2017 | +44.80% | +49.90% | +5.10% |
| 2016 | +13.95% | +19.05% | +5.10% |
| 2015 | **-1.99%** | **+0.57%** | +2.56% |
| **Avg/yr** | **+18.58%** | **+19.95%** | **+1.37%** |

ATR% wins 7/11 years. Where it loses (2019, 2023, 2020), it was underweight on volatile runners in trending markets. But it eliminates both losing years, giving a smoother equity curve.

---

## Filters Tested and Rejected

### EMA20 Rising Filter (for T signals)
- **Idea**: Only take T entries when EMA20 is rising over last N bars (uptrend confirmation)
- **Tested**: 20-bar and 10-bar lookback across 10 years
- **Result**: 10-bar showed +37K on aggregate but **0/10 years positive** on per-year testing, -3.45L total
- **Verdict**: Rejected. T entry conditions already require "was at upper Keltner" which implies uptrend. Adding EMA slope is double-filtering and hurts more than helps.

### Momentum Smoothness Filter (for T signals)
- **Idea**: Only take T entries when `20d_return / 20d_stdev > threshold` (mini Sharpe ratio — prefer smooth momentum)
- **Tested**: Thresholds 0.0, 0.3, 0.5 across 10 years
- **Result**: Only threshold 0.0 showed marginal aggregate improvement (+42K), but per-year: 4/10 wins, -1.03L total
- **Verdict**: Rejected. The signal is not strong enough on per-year validation.

---

## 2015 Loss Analysis (Worst Year)

2015 was the worst year (+0.57% with ATR% ranking, -1.99% with old Stop% ranking). Key findings:

- **August 2015 crash** (Chinese stock market panic): 6 trades exited on Aug 24 alone, losing Rs -1.29L. This single day accounts for most of the year's drag.
- **T SL slippage**: 27 T trades exited beyond -5% hard SL (backtest checks on daily close, not intraday). Extra loss beyond perfect 5% SL: Rs -1.23L. In live trading with real SL orders, this would be tighter.
- **J wide support**: 4 J trades lost >8% due to support being far from entry. ATR% ranking naturally avoids these (volatile stocks rank lower).
- **Re-entries after loss**: Only 5 instances, net Rs -826 — not a material issue.
- **Repeat losers**: TATASTEEL (3 J losses in Jan-Mar), LT (4 losses Jul-Dec), JSWSTEEL (4 losses Jan-Mar) — all declining stocks that kept triggering signals.

---

## Live Signals UI

### Top Picks Panel
- Combined J+T signals ranked by ATR% (lowest volatility first)
- Top 2 signals highlighted with star and green background
- Columns: Rank, Strategy (J/T), Stock, Close, **ATR%** (ranking), **Stop%** (SL distance), Stop Price, Detail, Buy button
- ATR% color: green (<2%), gray (2-3.5%), red (>3.5%)
- Stop% color: green (<5%), gray (=5%), red (>5%)

### Entered Signals (Positions) Panel
- Shows active positions with: Strategy badge (J green / T blue), User, Entry Date, Entry Price
- SL price with percentage (J: support level, T: entry * 0.95)
- T1 target (J: +5%, T: +6%), T2 target (J: +10%, T: +10%)
- Stage label (J: Full pos / Half out; T: Stage 0 / Stage 1 (1/3 out) / Stage 2 (2/3 out))
- Shares: remaining/total

### Historical Date Picker
- Date picker in sidebar to scan signals for any past date (2015 onwards)
- Historical scans hide Buy buttons (view-only)
- "Today" button resets to live cached data

---

## Risk Notes

- 2015 and 2018 are the weakest years. ATR% ranking turns both from negative to positive (2015: +0.57%, 2018: +5.08%).
- ATR% ranking can underperform in strong trending markets where volatile stocks are the big runners (e.g., 2019: +4.38% vs +19.36% with Stop% ranking).
- Results are based on daily closing prices. Live execution at different prices will cause slippage.
- T strategy SL slippage in backtest is an artifact of daily-bar checking. Live trading with actual SL orders will have tighter stops.
- Gap-down filter reduces trade count by ~10% (filters low-conviction "dead cat bounce" entries).
- ~170-200 trades per year (avg 177), roughly evenly split between J and T.
