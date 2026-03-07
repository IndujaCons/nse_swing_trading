# JTR Scale-Out Strategy

## Overview

Momentum dip-buying strategy on Nifty 100 stocks combining three signal types (J, T, R) with a 2-stage partial exit system. Capital: 20L, 2L per trade, 3 entries/day. Signals ranked by strategy priority (R→T→J) then sector momentum then ATR%.

---

## Entry Signals

### Strategy J — Weekly Support Bounce
- **Condition**: Close within 0-3% above weekly support level, IBS > 0.5, green candle, CCI(20) > -100, **no gap-down** (open >= prev close)
- **Support calculation**: 26-week rolling min of weekly close, **skipping the last 2 weeks** (current + previous). Uses proven support levels, not recent noise.
- **Stop**: Below weekly low (support break), also skipping last 2 weeks
- **Edge**: Defined structural support gives tight risk

### Strategy T — Keltner Channel Pullback
- **Condition**: Price within 1% of EMA(20), stock touched upper Keltner band (EMA20 + 2x ATR14) in last 10 bars, green candle, **IBS > 0.5** (close in upper half of bar), **no gap-down** (open >= prev close)
- **Stop**: 5% hard SL (shifts up to 3% below entry after first +6% partial exit — see Exit Rules)
- **Edge**: Buying a pullback in a confirmed uptrend (was recently at upper band = strong momentum)

### Strategy R — Bullish RSI Divergence (Regular + Hidden)
- **Regular Divergence** (reversal): Price makes a lower low but RSI(14) makes a higher low, **RSI(14) < 40** at the divergence point, **min 3-point RSI divergence**
- **Hidden Divergence** (continuation): Price makes a higher low but RSI(14) makes a lower low, **RSI(14) < 60** (relaxed), **min 5-point RSI divergence**, **close > EMA(50)** (uptrend filter)
- **Common Filters**: Green candle, **no gap-down** (open >= prev close)
- **Swing Low Detection**: Low[i] is minimum of surrounding window (5 bars left, 3 bars right). Signal confirmed 3 bars after actual low. Min 5 bars separation between the two swing lows.
- **RSI Zone**: Uses min RSI within ±3 bars of each swing low (price low and RSI low often don't land on same bar — this matches how traders visually read charts)
- **Divergence Window**: Looks back up to 50 bars for two qualifying swing lows
- **Stop**: Structural SL — 1% below the divergence swing low (natural invalidation level). **Regular: stop 0–5%**, **Hidden: stop 2–5%** (min 2% for hidden to avoid too-tight continuation entries).
- **Priority**: Regular divergence checked first; hidden only if regular not found
- **Dedup**: Skip if J or T already fired for the same stock on the same day
- **Edge**: Regular divergence = momentum improving despite price weakness (reversal). Hidden divergence = uptrend continuation despite temporary RSI dip.

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
2. **+6%**: Sell 1/3 of shares (lock in first profit), SL shifts up from 5% to **3% below entry** (entry × 0.97)
3. **Upper Keltner band**: Sell remaining 2/3 (EMA20 + 2x ATR14)

Note: The code contains a +10% stage that can fire before Keltner, but in practice Keltner almost always triggers first (~13% of all exits, +60L P&L — the single biggest profit contributor).

### Strategy R Exits (2-Stage, mirrors T)
1. **Structural SL**: 1% below the divergence swing low — exit all remaining shares (natural invalidation)
2. **+6%**: Sell 1/3 of shares (lock in first profit), SL shifts up to **3% below entry** (entry × 0.97)
3. **Upper Keltner band**: Sell remaining 2/3 (EMA20 + 2x ATR14)

### Underwater Exit (applies to J, T, and R)
- **Rule**: If a trade is held for **10+ trading days** and price is still **below entry price**, exit at close
- **Why**: Analysis of 2024-2025 biggest losses showed a common "falling knife" pattern — stocks entering near support but drifting lower for weeks. Cutting these losers after 10 days frees capital for better opportunities.
- **Impact**: +4.52L improvement over 11 years, turned 2015 from a losing year into a winning year (11/11 positive years)
- **Nifty shield NOT used**: Tested shielding underwater exit when Nifty itself was underwater — it neutralized the benefit entirely (-865 vs baseline). Pure discipline of cutting losers works better.

### Why 3% Tight SL After First Target
After Strategy T hits the first +6% target and sells 1/3, the remaining 2/3 position has a real risk: the stock can reverse and hit the original 5% SL, giving back more than the +6% partial profit. So the SL is shifted up from 5% to 3% below entry price, reducing downside on the remaining position.

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

## Signal Ranking — Strategy Priority + Sector Momentum + ATR%

When multiple signals fire on the same day and you have limited slots (3/day):

### Current Method: R → T → J → Sector Momentum (desc) → ATR% (asc)

Signals are ranked in priority order:
1. **Strategy priority**: R → T → J (R signals always come first)
2. **Sector momentum** (descending): Stocks in sectors with rising relative strength rank higher
3. **ATR%** (ascending): Among equal sector momentum, calmer stocks preferred

### Sector Momentum Score
- Measures whether a sector's RS vs Nifty 50 is accelerating or decelerating
- Formula: `delta_5d × 3 + delta_10d × 2 + delta_20d × 1` (weighted sum of RS changes)
- A sector at -11% RS but +27 momentum = recovering fast → ranks higher than flat +4%
- Uses 17 Nifty sectoral indices (IT, Bank, Pharma, Auto, etc.)

### 11-Year Backtest: Sector Momentum vs ATR-only Ranking

| Metric | ATR Baseline | Sector Momentum | Delta |
|---|---|---|---|
| Net P&L | +77.2L | +77.4L | **+0.2L** |
| CAGR (net) | 15.5% | 15.5% | — |
| Max Drawdown | 14.3% | 10.0% | **-4.3%** |
| Sharpe | 1.45 | 1.90 | +0.45 |
| Calmar | 1.08 | 1.55 | +0.47 |
| Wins | — | **11/11 years** | — |

### Stop% and ATR% are still displayed
The UI shows ATR%, Stop% (SL distance), and Sector column with momentum arrows in the Top Picks panel.

---

## Backtest Results — 11 Years (2015-2025), Nifty 100

### Current — TR Sector Momentum (2-Stage 6%+Keltner + Gap-Down + Sector Momentum Ranking + Skip 2wk Support + UW Exit 10d + T: IBS>0.5 + tight SL 3% + R: regular+hidden divergence, RSI<40/60, div>=3/5pt, EMA50 filter, regular stop 0-5%, hidden stop 2-5%, RSI ±3 bar zone + 3 entries/day)

Charges: Zerodha delivery (₹0 brokerage, STT 0.1% both sides). Tax: STCG 20% on (gross - deductible charges). STT not deductible.

| Year | Trades | Win | Loss | WR% | AvgWin | AvgLoss | PF | Gross | Chg | Tax | Net | NetR% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2015 | 277 | 152 | 125 | 54.9% | +9,360 | -6,141 | 1.85 | +6.6L | 124K | 129K | +4.0L | +20.1% |
| 2016 | 269 | 176 | 93 | 65.4% | +9,579 | -5,591 | 3.24 | +11.7L | 121K | 231K | +8.1L | +40.7% |
| 2017 | 230 | 122 | 108 | 53.0% | +9,874 | -4,151 | 2.69 | +7.6L | 103K | 149K | +5.0L | +25.2% |
| 2018 | 299 | 162 | 137 | 54.2% | +8,307 | -6,163 | 1.59 | +5.0L | 133K | 98K | +2.7L | +13.6% |
| 2019 | 372 | 266 | 106 | 71.5% | +8,999 | -5,239 | 4.31 | +18.4L | 167K | 364K | +13.1L | +65.4% |
| 2020 | 313 | 220 | 93 | 70.3% | +9,919 | -7,365 | 3.19 | +15.0L | 141K | 297K | +10.6L | +53.0% |
| 2021 | 359 | 226 | 133 | 63.0% | +8,663 | -5,757 | 2.56 | +11.9L | 160K | 235K | +8.0L | +39.8% |
| 2022 | 271 | 166 | 105 | 61.3% | +8,429 | -6,874 | 1.94 | +6.8L | 121K | 133K | +4.2L | +21.2% |
| 2023 | 279 | 177 | 102 | 63.4% | +9,136 | -4,676 | 3.39 | +11.4L | 125K | 226K | +7.9L | +39.5% |
| 2024 | 305 | 184 | 121 | 60.3% | +8,971 | -6,313 | 2.16 | +8.9L | 136K | 175K | +5.8L | +28.8% |
| 2025 | 305 | 180 | 125 | 59.0% | +9,532 | -4,454 | 3.08 | +11.6L | 136K | 229K | +7.9L | +39.7% |
| **Total** | **3,279** | **2,031** | **1,248** | **61.9%** | **+9,146** | **-5,694** | **2.61** | **+114.7L** | **1465K** | **2265K** | **+77.4L** | |

- **Winning years: 11/11** (all years profitable, worst +13.6% net)
- **Best year: +65.4% net (2019)**
- **Gross P&L: Rs 114.7L → Net post-tax: Rs 77.4L** (you keep 67 paise per rupee)
- **Net CAGR: 15.5%** after STT, charges, and 20% STCG tax
- **Avg Win/Loss ratio: 1.61x** (make Rs 9,146 on winners, lose Rs 5,694 on losers)
- **Avg PF: 2.61**
- **Risk metrics**: Sharpe 1.90, Calmar 1.55, Max DD 10.0%, R² 0.027
- **By strategy**: R: 2,153 trades, 66% WR, +90.8L (79% of gross) | T: 1,126 trades, 55% WR, +23.9L (21% of gross)

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
| Sector momentum | `rank_by_sector_momentum=True` | Sector RS momentum score | R→T→J → momentum → ATR% |
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
- Combined J+T+R signals ranked by: Strategy priority (R→T→J) → Sector momentum (desc) → ATR% (asc)
- Top 3 signals highlighted with star and green background
- Columns: Rank, Strategy (J/T/R), Stock, Close, **Sector** (with momentum arrow), **ATR%**, **Stop%** (SL distance), Stop Price, Detail, Buy button
- Sector column: green arrow up = rising RS, red arrow down = falling RS
- ATR% color: green (<2%), gray (2-3.5%), red (>3.5%)
- Stop% color: green (<5%), gray (=5%), red (>5%)
- R badge: purple (#ab47bc)
- Sector concentration warning bar when 3+ stocks from same sector
- Buy button shows confirm dialog when 3+ same-sector positions already held

### Sector Momentum Heatmap Panel
- Always visible regardless of strategy selection
- Shows all 17 Nifty sector indices with RS% and momentum direction
- Color-coded pills: green = positive RS, red = negative RS, arrow = momentum direction
- (i) info button explaining how to read the heatmap
- Sorted by momentum score (strongest momentum first)

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

- 2018 is the weakest year (+13.6% net) — choppy market with many false signals.
- 2015 is second weakest (+20.1% net) — survivorship bias (current Nifty 100 stocks tested on 2015 data).
- ATR% ranking can underperform in strong trending markets where volatile stocks are the big runners.
- Results are based on daily closing prices. Live execution at different prices will cause slippage.
- T strategy SL slippage in backtest is an artifact of daily-bar checking. Live trading with actual SL orders will have tighter stops.
- Capital check: entries are skipped if trade cost would exceed available capital (20L + running PnL - deployed). Prevents over-leveraging after losses.
- Gap-down filter reduces trade count by ~10% (filters low-conviction "dead cat bounce" entries).
- ~230-372 trades per year (avg 298), mix of T and R.
- yfinance data variability: backtest numbers can shift ~0.5-1% between runs due to Yahoo Finance adjusting historical prices for splits/dividends.
- yfinance returns adjusted prices (for splits and dividends), so historical prices in explain_trade may differ from actual traded prices on TradingView charts.
