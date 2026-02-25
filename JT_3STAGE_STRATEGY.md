# JTR 3-Stage Scale-Out Strategy

## Overview

Momentum dip-buying strategy on Nifty 100 stocks combining three signal types (J, T, R) with a 3-stage partial exit system. Capital: 20L, 2L per trade, 3 entries/day.

---

## Entry Signals

### Strategy J — Weekly Support Bounce
- **Condition**: Close within 0-3% above weekly support level, IBS > 0.5, green candle, CCI(20) > -100, **no gap-down** (open >= prev close)
- **Support calculation**: 26-week rolling min of weekly close, **skipping the last 2 weeks** (current + previous). Uses proven support levels, not recent noise.
- **Stop**: Below weekly low (support break), also skipping last 2 weeks
- **Edge**: Defined structural support gives tight risk

### Strategy T — Keltner Channel Pullback
- **Condition**: Price within 1% of EMA(20), stock touched upper Keltner band (EMA20 + 2x ATR14) in last 10 bars, green candle, **no gap-down** (open >= prev close)
- **Stop**: 5% hard SL (tightens to 3% after first +6% partial exit — see Exit Rules)
- **Edge**: Buying a pullback in a confirmed uptrend (was recently at upper band = strong momentum)

### Strategy R — Bullish RSI Divergence
- **Condition**: Price makes a lower low but RSI(14) makes a higher low (bullish divergence), **RSI(14) < 40** at the divergence point, **min 3-point RSI divergence** between the two swing lows, green candle, **no gap-down** (open >= prev close)
- **Swing Low Detection**: Low[i] is minimum of surrounding window (5 bars left, 3 bars right). Signal confirmed 3 bars after actual low. Min 5 bars separation between the two swing lows.
- **Divergence Window**: Looks back up to 50 bars for two qualifying swing lows
- **Stop**: Structural SL — 1% below the divergence swing low (natural invalidation level). **Max stop distance: 5%** — skip signal if structural stop is >5% away (stale divergence).
- **Dedup**: Skip if J or T already fired for the same stock on the same day
- **Edge**: RSI divergence signals momentum improving despite price weakness — early reversal signal

### Gap-Down Filter (applies to J, T, and R)
- **Rule**: Skip entry if today's Open < yesterday's Close (stock gapped down)
- **Why**: Analysis of top 10 losses in 2018 showed 9/10 had gap-down entries on low volume — classic "dead cat bounce" pattern. Filtering these improves avg return by +2.6%/yr and eliminates losing years.

---

## Exit Rules

### Strategy J Exits
J uses its own exit logic (not the 3-stage system):
1. **Support break**: Full exit if price drops below weekly low stop (with Nifty shield — skipped if Nifty dropped same % or more)
2. **+5%**: Sell 50%
3. **+10%**: Sell remaining
4. **Chandelier exit**: After partial, sell remaining if price < Highest High (since entry) - 3x ATR(14)

### Strategy T Exits (3-Stage)
1. **Hard SL**: 5% below entry — exit all remaining shares
2. **+6%**: Sell 1/3 of shares (lock in first profit)
3. **Tight SL kicks in**: After first +6% exit, SL tightens from 5% to **3%** on remaining shares
4. **+10%**: Sell 1/3 of shares (lock in second profit)
5. **Indicator exit on remaining 1/3**: Price reaches upper Keltner band (EMA20 + 2x ATR14)

### Strategy R Exits (3-Stage, mirrors T)
1. **Structural SL**: 1% below the divergence swing low — exit all remaining shares (natural invalidation)
2. **+6%**: Sell 1/3 of shares (lock in first profit)
3. **Tight SL kicks in**: After first +6% exit, SL tightens to **3%** on remaining shares
4. **+10%**: Sell 1/3 of shares (lock in second profit)
5. **Indicator exit on remaining 1/3**: Price reaches upper Keltner band (EMA20 + 2x ATR14)

### Underwater Exit (applies to J, T, and R)
- **Rule**: If a trade is held for **10+ trading days** and price is still **below entry price**, exit at close
- **Why**: Analysis of 2024-2025 biggest losses showed a common "falling knife" pattern — stocks entering near support but drifting lower for weeks. Cutting these losers after 10 days frees capital for better opportunities.
- **Impact**: +4.52L improvement over 11 years, turned 2015 from a losing year into a winning year (11/11 positive years)
- **Nifty shield NOT used**: Tested shielding underwater exit when Nifty itself was underwater — it neutralized the benefit entirely (-865 vs baseline). Pure discipline of cutting losers works better.

### Why 3% Tight SL After First Target
After Strategy T hits the first +6% target and sells 1/3, the remaining 2/3 position has a real risk: the stock can reverse and hit the original 5% SL, giving back more than the +6% partial profit.

Tested across 11 years (2015-2025):

| SL after 1st exit | Total P&L | Avg/yr | Delta vs 5% |
|---|---|---|---|
| 5% (original) | Rs 47.76L | 21.71% | — |
| 4% | Rs 47.42L | 21.56% | -0.34L |
| **3% (adopted)** | **Rs 48.1L** | **~22%** | **+2.36L** |

3% SL was positive in 9/11 years. It works through faster capital recycling — losing positions are cut sooner, freeing money for winning trades.

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
| Entries per day | 3 |
| Strategies | J + T + R |
| Underwater exit | 10 trading days |
| T tight SL | 3% after first +6% exit |

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

### Current Baseline — JTR (3-Stage 6/10 + Gap-Down + ATR% Ranking + Skip 2wk Support + UW Exit 10d + T Tight SL 3% + R filters: RSI<40, divergence>=3pt, stop cap 5% + 3 entries/day)

| Year | Trades | WR% | Return% | P&L | AvgWin | AvgLoss | PF |
|---|---|---|---|---|---|---|---|
| 2025 | 257 | 47.9% | +21.0% | +4,19,157 | 7,477 | -3,735 | 1.84 |
| 2024 | 244 | 57.8% | +29.2% | +5,83,114 | 7,958 | -5,232 | 2.08 |
| 2023 | 217 | 56.2% | +28.1% | +5,62,018 | 7,982 | -4,334 | 2.36 |
| 2022 | 269 | 59.1% | +26.1% | +5,21,611 | 7,579 | -6,213 | 1.76 |
| 2021 | 250 | 56.4% | +33.0% | +6,59,408 | 8,894 | -5,456 | 2.11 |
| 2020 | 303 | 58.7% | +27.0% | +5,39,255 | 8,384 | -7,625 | 1.57 |
| 2019 | 333 | 58.6% | +37.9% | +7,57,259 | 8,129 | -5,999 | 1.91 |
| 2018 | 248 | 53.2% | +17.0% | +3,39,740 | 7,597 | -5,716 | 1.51 |
| 2017 | 254 | 59.1% | +49.3% | +9,86,825 | 9,534 | -4,262 | 3.23 |
| 2016 | 255 | 51.0% | +20.1% | +4,02,324 | 7,592 | -4,677 | 1.69 |
| 2015 | 234 | 45.7% | +2.9% | +57,063 | 7,740 | -6,072 | 1.07 |
| **Avg/yr** | **260** | **55.1%** | **+26.5%** | **+5,29,798** | **8,079** | **-5,393** | **1.92** |

- **Positive years: 11/11**
- **Best year: +49.3% (2017)**
- **Worst year: +2.9% (2015)**
- **Total P&L: Rs ~58.3 Lakhs on 20L capital over 11 years (291% cumulative)**
- **Avg Win/Loss ratio: 1.5x** (make Rs 8,079 on winners, lose Rs 5,393 on losers)
- **R adds +9.5L over JT-only baseline**, value across most years

### Previous JT-only Baseline (before Strategy R)

| Year | Trades | WR% | Return% | P&L | AvgWin | AvgLoss | PF |
|---|---|---|---|---|---|---|---|
| 2025 | 236 | 48.7% | 19.4% | +3,87,352 | 8,118 | -4,514 | 1.71 |
| 2024 | 231 | 61.5% | 30.9% | +6,18,676 | 8,319 | -6,321 | 2.10 |
| 2023 | 220 | 59.5% | 33.2% | +6,63,231 | 8,313 | -4,784 | 2.56 |
| 2022 | 248 | 58.5% | 22.9% | +4,58,722 | 8,163 | -7,039 | 1.63 |
| 2021 | 239 | 54.0% | 26.9% | +5,37,875 | 9,118 | -5,804 | 1.84 |
| 2020 | 249 | 60.2% | 28.4% | +5,68,858 | 8,790 | -7,573 | 1.76 |
| 2019 | 242 | 50.8% | 13.0% | +2,59,948 | 8,151 | -6,240 | 1.35 |
| 2018 | 213 | 47.9% | 5.3% | +1,05,872 | 7,782 | -6,197 | 1.15 |
| 2017 | 211 | 58.8% | 41.8% | +8,36,657 | 9,496 | -3,917 | 3.45 |
| 2016 | 234 | 50.4% | 18.4% | +3,68,231 | 8,233 | -5,200 | 1.61 |
| 2015 | 222 | 48.6% | 3.9% | +77,925 | 7,744 | -6,653 | 1.10 |
| **Avg/yr** | **231** | **54.4%** | **~22%** | **+4,43,941** | **8,384** | **-5,840** | **1.82** |

- **Total P&L: Rs ~48.8 Lakhs** (JTR adds +8.6L = **+17.6% improvement**)

### Why Skip 2 Weeks of Support
J support uses the 26-week rolling min of weekly closes. Including the last 2 weeks (current + previous) means support can be set by very recent price action that hasn't been tested. Shifting by 2 weeks uses only proven support levels — prices the market has bounced from and held for at least 2 weeks.

---

## Filters/Rules Tested and Rejected

### Earnings Blackout (5 days before quarterly results)
- **Idea**: Block entries during peak results season (Jan 10-Feb 10, Apr 10-May 10, Jul 10-Aug 10, Oct 10-Nov 10)
- **Result**: Blocked ~400 trades across 11 years, lost **-14.23L** vs baseline
- **Verdict**: Rejected. Too blunt — blocks good trades along with bad ones. The strategy already has stop-losses to handle earnings-related drops.

### T Chandelier Exit for Stage 2 (trailing stop on last 1/3)
- **Idea**: After T's first two partial exits, use Chandelier exit (Highest High - 3x ATR14) on the remaining 1/3 instead of waiting for upper Keltner
- **Result**: **-1.02L** vs baseline over 11 years
- **Verdict**: Rejected. Cuts T winners short before they reach upper Keltner. The upper Keltner exit already acts as a natural trailing indicator.

### Nifty Shield for Underwater Exit
- **Idea**: Skip underwater exit if Nifty index itself is also underwater (market-wide fall, not stock-specific)
- **Result**: Neutralized the underwater exit benefit entirely (-865 vs baseline instead of +4.52L)
- **Verdict**: Rejected. A losing position is a losing position regardless of market context. Pure discipline of cutting losers after 10 days works better than making excuses for them.

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

### 4% Tight SL After First T Target
- **Idea**: Tighten SL from 5% to 4% (instead of 3%) after first +6% exit
- **Result**: **-0.34L** vs baseline — actually slightly worse than keeping 5%
- **Verdict**: Rejected. 4% is too close to the noise band. 3% is the sweet spot — tight enough to save capital but not so tight it shakes out recoverable dips.

---

## Code Paths — All Three in Sync

The strategy runs in three places. All three use identical entry/exit logic:

| Feature | Backtest Engine | Live Signals | UI API |
|---|---|---|---|
| File | `data/momentum_backtest.py` | `data/live_signals_engine.py` | `ui/app.py` |
| Gap-down filter | `no_gap_down=True` | Open < prev close → skip | `no_gap_down=True` |
| ATR% ranking | `rank_by_risk=True` | Sort by `atr_pct` | `rank_by_risk=True` |
| Underwater exit | `underwater_exit_days=10` | 10 trading days check | `underwater_exit_days=10` |
| T tight SL | `t_tight_sl=0.03` | 3% after stage >= 1 | `t_tight_sl=0.03` |
| J Nifty shield | SL skip if Nifty fell more | SL skip if Nifty fell more | — |
| J Chandelier | HH - 3x ATR14 after partial | HH - 3x ATR14 after partial | — |
| T 3-stage | 6% / 10% / Keltner | 6% / 10% / Keltner | `three_stage_exit=True` |
| R structural SL | 1% below swing low | 1% below swing low | metadata.r_swing_low_stop |
| R 3-stage | 6% / 10% / Keltner | 6% / 10% / Keltner | `three_stage_exit=True` |
| R dedup | Skip if J/T fired | Skip if J/T fired | — |

---

## Live Signals UI

### Top Picks Panel
- Combined J+T+R signals ranked by ATR% (lowest volatility first)
- Top 2 signals highlighted with star and green background
- Columns: Rank, Strategy (J/T/R), Stock, Close, **ATR%** (ranking), **Stop%** (SL distance), Stop Price, Detail, Buy button
- ATR% color: green (<2%), gray (2-3.5%), red (>3.5%)
- Stop% color: green (<5%), gray (=5%), red (>5%)
- R badge: purple (#ab47bc)

### Entered Signals (Positions) Panel
- Shows active positions with: Strategy badge (J green / T blue / R purple), User, Entry Date, Entry Price
- SL price with percentage (J: support level, T: entry * 0.95 → tightens to 0.97 after first exit, R: structural SL 1% below swing low → tightens to 3% after first exit)
- T1 target (J: +5%, T/R: +6%), T2 target (J/T/R: +10%)
- Stage label (J: Full pos / Half out; T/R: Stage 0 / Stage 1 (1/3 out) / Stage 2 (2/3 out))
- Shares: remaining/total
- Exit signals: UNDERWATER_EXIT after 10 trading days below entry

### Historical Date Picker
- Date picker in sidebar to scan signals for any past date (2015 onwards)
- Historical scans hide Buy buttons (view-only)
- "Today" button resets to live cached data

---

## Risk Notes

- 2015 is the weakest year (+2.9%) — still positive.
- 2018 is second weakest (+17.0%) due to broad market correction.
- ATR% ranking can underperform in strong trending markets where volatile stocks are the big runners.
- Results are based on daily closing prices. Live execution at different prices will cause slippage.
- T strategy SL slippage in backtest is an artifact of daily-bar checking. Live trading with actual SL orders will have tighter stops.
- Capital check: entries are skipped if trade cost would exceed available capital (20L + running PnL - deployed). Prevents over-leveraging after losses.
- Gap-down filter reduces trade count by ~10% (filters low-conviction "dead cat bounce" entries).
- ~217-333 trades per year (avg 260), mix of J, T, and R.
- yfinance data variability: backtest numbers can shift ~0.5-1% between runs due to Yahoo Finance adjusting historical prices for splits/dividends.
