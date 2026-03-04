# JTR Scale-Out Strategy

## Overview

Momentum dip-buying strategy on Nifty 100 stocks combining three signal types (J, T, R) with a 2-stage partial exit system. Capital: 20L, 2L per trade, 3 entries/day.

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
J uses its own exit logic (2-stage):
1. **Support break**: Full exit if price drops below weekly low stop (with Nifty shield — skipped if Nifty dropped same % or more)
2. **+5%**: Sell 50%
3. **+10%**: Sell remaining (or Chandelier exit: Highest High since entry - 3x ATR14)

### Strategy T Exits (2-Stage)
1. **Hard SL**: 5% below entry — exit all remaining shares
2. **+6%**: Sell 1/3 of shares (lock in first profit), SL tightens from 5% to **3%**
3. **Upper Keltner band**: Sell remaining 2/3 (EMA20 + 2x ATR14)

Note: The code contains a +10% stage that can fire before Keltner, but in practice Keltner almost always triggers first (~13% of all exits, +60L P&L — the single biggest profit contributor).

### Strategy R Exits (2-Stage, mirrors T)
1. **Structural SL**: 1% below the divergence swing low — exit all remaining shares (natural invalidation)
2. **+6%**: Sell 1/3 of shares (lock in first profit), SL tightens to **3%**
3. **Upper Keltner band**: Sell remaining 2/3 (EMA20 + 2x ATR14)

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

### Why 2-Stage with +6% Partial
The original system sold 1/2 at +5% and exited the other 1/2 on indicator. The 2-stage system:
- Sells only 1/3 at +6% (keeps more capital in the trade, lets winners run further)
- Lets remaining 2/3 ride to upper Keltner band (natural momentum exhaustion point)
- Wider first target (6% vs 5%) adds +1.99%/yr and increases avg win by Rs 899/trade (+12%)
- Tested true 3-stage (+6%, +10%, chandelier) for R — was -6.14L worse. Keltner exit is the right exit for T and R.

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

### Current Baseline — JTR (2-Stage 6%+Keltner + Gap-Down + ATR% Ranking + Skip 2wk Support + UW Exit 10d + T Tight SL 3% + R filters: RSI<40, divergence>=3pt, stop cap 5% + 3 entries/day)

| Year | Trades | Win | Loss | WR% | AvgWin | AvgLoss | PF | Return% | P&L | Hold |
|---|---|---|---|---|---|---|---|---|---|---|
| 2015 | 223 | 99 | 124 | 44.4% | 7,131 | -6,276 | 0.91 | -3.6% | -0.72L | 16d |
| 2016 | 279 | 143 | 136 | 51.3% | 8,297 | -4,469 | 1.95 | +28.9% | +5.79L | 23d |
| 2017 | 270 | 161 | 109 | 59.6% | 9,179 | -4,240 | 3.20 | +50.8% | +10.16L | 17d |
| 2018 | 306 | 156 | 150 | 51.0% | 7,670 | -4,956 | 1.61 | +22.7% | +4.53L | 18d |
| 2019 | 447 | 305 | 142 | 68.2% | 8,829 | -5,520 | 3.44 | +95.5% | +19.09L | 26d |
| 2020 | 345 | 217 | 128 | 62.9% | 9,120 | -7,582 | 2.04 | +50.4% | +10.08L | 19d |
| 2021 | 274 | 160 | 114 | 58.4% | 9,203 | -5,285 | 2.44 | +43.5% | +8.70L | 16d |
| 2022 | 345 | 234 | 111 | 67.8% | 7,787 | -6,422 | 2.56 | +55.5% | +11.09L | 26d |
| 2023 | 242 | 138 | 104 | 57.0% | 8,159 | -4,569 | 2.37 | +32.5% | +6.51L | 25d |
| 2024 | 267 | 160 | 107 | 59.9% | 7,996 | -5,836 | 2.05 | +32.7% | +6.55L | 18d |
| 2025 | 292 | 156 | 136 | 53.4% | 7,862 | -3,372 | 2.67 | +38.4% | +7.68L | 21d |
| **Avg** | **299** | **175** | **124** | **57.6%** | **8,294** | **-5,321** | **2.29** | **+40.7%** | **+8.13L** | **21d** |
| **Total** | **3,290** | **1,929** | **1,361** | | | | | **+447.3%** | **+89.46L** | |

- **Winning years: 10/11** (only 2015 red at -0.72L)
- **Best year: +95.5% (2019)**
- **Worst year: -3.6% (2015)**
- **Total P&L: Rs 89.5 Lakhs on 20L capital**
- **CAGR: 16.7%**
- **Avg Win/Loss ratio: 1.6x** (make Rs 8,294 on winners, lose Rs 5,321 on losers)
- **Avg PF: 2.29**
- **By strategy**: R: 1,288 trades, 65.4% WR, +49.7L (56% of total) | J: 795 trades, 54.5% WR, +14.1L | T: 1,207 trades, 54.2% WR, +25.6L

### Previous JT-only Baseline (before Strategy R)

| Year | Trades | WR% | Return% | P&L | AvgWin | AvgLoss | PF |
|---|---|---|---|---|---|---|---|
| 2025 | 226 | 47.3% | +19.4% | +3,87,418 | 7,911 | -3,857 | 1.84 |
| 2024 | 235 | 61.7% | +32.8% | +6,55,078 | 8,570 | -6,529 | 2.11 |
| 2023 | 229 | 57.2% | +31.4% | +6,27,162 | 8,181 | -4,537 | 2.41 |
| 2022 | 240 | 58.8% | +25.4% | +5,08,685 | 8,217 | -6,565 | 1.78 |
| 2021 | 253 | 54.5% | +31.5% | +6,29,306 | 9,527 | -5,960 | 1.92 |
| 2020 | 255 | 60.4% | +30.0% | +5,99,890 | 8,701 | -7,328 | 1.81 |
| 2019 | 243 | 48.6% | +7.5% | +1,50,705 | 8,103 | -6,444 | 1.19 |
| 2018 | 221 | 48.4% | +8.4% | +1,67,811 | 8,148 | -6,175 | 1.24 |
| 2017 | 222 | 60.4% | +41.3% | +8,26,326 | 8,978 | -4,281 | 3.19 |
| 2016 | 226 | 47.8% | +14.2% | +2,83,456 | 8,182 | -5,087 | 1.47 |
| 2015 | 202 | 46.5% | +2.3% | +46,779 | 7,876 | -6,422 | 1.07 |
| **Avg/yr** | **232** | **54.0%** | **+22.2%** | **+4,43,874** | **8,445** | **-5,742** | **1.72** |

- **Total P&L: Rs ~48.8 Lakhs** (JTR adds +35.5L = **+72.7% improvement**)

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
| T 2-stage | 6% partial + Keltner | 6% partial + Keltner | `three_stage_exit=True` |
| R structural SL | 1% below swing low | 1% below swing low | metadata.r_swing_low_stop |
| R 2-stage | 6% partial + Keltner | 6% partial + Keltner | `three_stage_exit=True` |
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

- 2015 is the only negative year (-3.6%) — survivorship bias (current Nifty 100 stocks tested on 2015 data).
- 2018 is weakest positive year (+22.7%) due to broad market correction.
- ATR% ranking can underperform in strong trending markets where volatile stocks are the big runners.
- Results are based on daily closing prices. Live execution at different prices will cause slippage.
- T strategy SL slippage in backtest is an artifact of daily-bar checking. Live trading with actual SL orders will have tighter stops.
- Capital check: entries are skipped if trade cost would exceed available capital (20L + running PnL - deployed). Prevents over-leveraging after losses.
- Gap-down filter reduces trade count by ~10% (filters low-conviction "dead cat bounce" entries).
- ~223-447 trades per year (avg 299), mix of J, T, and R.
- yfinance data variability: backtest numbers can shift ~0.5-1% between runs due to Yahoo Finance adjusting historical prices for splits/dividends.
