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

### Production Config (3-Stage 6/10 + Gap-Down + ATR% Ranking + Skip 2wk Support + Capital Check)

| Year | Trades | J | T | Win | Loss | WR% | P&L | Return% | AvgWin | AvgLoss | PF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025 | 153 | 78 | 75 | 111 | 42 | 72.5% | +5,28,787 | +26.44% | 8,313 | -9,379 | 2.34 |
| 2024 | 177 | 60 | 117 | 122 | 55 | 68.9% | +4,03,236 | +20.16% | 7,805 | -9,982 | 1.73 |
| 2023 | 162 | 61 | 101 | 126 | 36 | 77.8% | +7,08,960 | +35.45% | 8,633 | -10,521 | 2.87 |
| 2022 | 188 | 89 | 99 | 119 | 69 | 63.3% | +2,98,775 | +14.94% | 8,883 | -10,990 | 1.39 |
| 2021 | 183 | 40 | 143 | 129 | 54 | 70.5% | +5,60,201 | +28.01% | 9,170 | -11,531 | 1.90 |
| 2020 | 224 | 74 | 150 | 161 | 63 | 71.9% | +5,79,226 | +28.96% | 8,668 | -12,957 | 1.71 |
| 2019 | 183 | 77 | 106 | 113 | 70 | 61.7% | +1,73,010 | +8.65% | 7,958 | -10,375 | 1.24 |
| 2018 | 157 | 93 | 64 | 97 | 60 | 61.8% | +2,03,759 | +10.19% | 8,031 | -9,587 | 1.35 |
| 2017 | 166 | 63 | 103 | 134 | 32 | 80.7% | +9,39,744 | +46.99% | 9,529 | -10,536 | 3.79 |
| 2016 | 161 | 68 | 93 | 113 | 48 | 70.2% | +3,98,957 | +19.95% | 8,217 | -11,033 | 1.75 |
| 2015 | 143 | 70 | 73 | 78 | 65 | 54.5% | -73,513 | -3.68% | 7,986 | -10,715 | 0.89 |
| **Total** | **1,897** | **773** | **1,124** | **1,303** | **594** | | **+47,21,142** | | | | |
| **Avg/yr** | **172** | | | | | **68.7%** | **+4,29,195** | **+21.46%** | **8,522** | **-10,747** | **1.74** |

- **Positive years: 10/11** (2015: -3.68%)
- **Best year: +46.99% (2017)**
- **Worst year: -3.68% (2015)**
- **Total P&L: Rs 47.21 Lakhs on 20L capital over 11 years**

### Why Skip 2 Weeks of Support
J support uses the 26-week rolling min of weekly closes. Including the last 2 weeks (current + previous) means support can be set by very recent price action that hasn't been tested. Shifting by 2 weeks uses only proven support levels — prices the market has bounced from and held for at least 2 weeks.

**Impact over 11 years:**

| Metric | Without skip | With skip 2wk | Diff |
|---|---|---|---|
| Avg return/yr | +19.95% | +21.46% | +1.51% |
| Total PnL | +43.88L | +47.21L | +3.33L |
| Worst year | +0.57% | -3.68% | — |
| Positive years | 11/11 | 10/11 | — |

Wins 8/11 years. The 3 underperformance years (2024, 2021, 2017) are due to butterfly effects on T slot allocation, not structural J issues.

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

## 2015 Loss Analysis

2015 is the only losing year (-2.00% with capital check enforced). Key findings:

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

- 2015 is the only losing year (-3.68%). Capital check prevents entries when losses have exhausted available capital.
- 2018 and 2019 are the weakest positive years (+10.19% and +8.65%).
- ATR% ranking can underperform in strong trending markets where volatile stocks are the big runners (e.g., 2019: +18.57% vs higher with Stop% ranking).
- Results are based on daily closing prices. Live execution at different prices will cause slippage.
- T strategy SL slippage in backtest is an artifact of daily-bar checking. Live trading with actual SL orders will have tighter stops.
- Capital check: entries are skipped if trade cost would exceed available capital (20L + running PnL - deployed). Prevents over-leveraging after losses.
- Gap-down filter reduces trade count by ~10% (filters low-conviction "dead cat bounce" entries).
- ~143-224 trades per year (avg 172), T trades slightly outnumber J.
