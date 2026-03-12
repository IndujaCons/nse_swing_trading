# R+MW Swing Trading System

## Overview

Momentum swing trading strategy on Nifty 200 stocks combining two primary signal types (R, MW) with staged partial exit systems. Capital: 20L, 2L per trade, 3 entries/day. Signals ranked by strategy priority (R=MW) then sector momentum then ATR%. Nifty crash shield active on R, MW, and J.

Secondary strategies (T, J) are displayed in live signals but ranked below R and MW.

---

## Entry Signals

### Strategy R — Bullish RSI Divergence (PRIMARY)
- **Regular Divergence** (reversal): Price makes a lower low but RSI(14) makes a higher low, **RSI(14) < 35** at the divergence point, **min 3-point RSI divergence**
- **Hidden Divergence** (continuation): Price makes a higher low but RSI(14) makes a lower low, **RSI(14) < 60** (relaxed), **min 5-point RSI divergence**, **close > EMA(50)** (uptrend filter)
- **Common Filters**: Green candle, **no gap-down** (open >= prev close)
- **Swing Low Detection**: Low[i] is minimum of surrounding window (5 bars left, 3 bars right). Signal confirmed 3 bars after actual low. Min 5 bars separation between the two swing lows.
- **RSI Zone**: Uses min RSI within +/-3 bars of each swing low (price low and RSI low often don't land on same bar — this matches how traders visually read charts)
- **Divergence Window**: Looks back up to 50 bars for two qualifying swing lows
- **Stop**: Structural SL — 1% below the divergence swing low. **Regular: stop 0-6%**, **Hidden: stop 2-6%** (min 2% for hidden to avoid too-tight continuation entries).
- **Priority**: Regular divergence checked first; hidden only if regular not found
- **Edge**: Regular divergence = momentum improving despite price weakness (reversal). Hidden divergence = uptrend continuation despite temporary RSI dip.

### Strategy MW — Weekly ADX Momentum (PRIMARY)
- **Signal**: Weekly ADX >= 25 and rising (curr > prev) + DI+ > DI-
- **Timeframe**: Weekly ADX, daily entry
- **Entry**: Green candle, no gap-down
- **Edge**: Catches the moment a trend shifts from emerging to confirmed

### Strategy T — Keltner Channel Pullback (SECONDARY, priority 2)
- **Condition**: Price within 1% of EMA(20), stock touched upper Keltner band (EMA20 + 2x ATR14) in last 10 bars, green candle, **IBS > 0.5**, **no gap-down**
- **Edge**: Buying a pullback in a confirmed uptrend

### Strategy J — Weekly Support Bounce (SECONDARY, priority 3)
- **Condition**: Close within 0-3% above weekly support level, IBS > 0.5, green candle, CCI(20) > -100, **no gap-down**
- **Support calculation**: 26-week rolling min of weekly close, **skipping the last 2 weeks**
- **Edge**: Defined structural support gives tight risk

### Gap-Down Filter (applies to all strategies)
- **Rule**: Skip entry if today's Open < yesterday's Close
- **Why**: Gap-down entries are classic "dead cat bounce" patterns with poor outcomes

---

## Exit Rules

### Strategy R Exits (2-Stage in Portfolio)
1. **Structural SL**: 1% below the divergence swing low — exit all (skipped if Nifty crash shield active)
2. **+8%**: Sell 1/3 of shares, SL shifts to 3% below entry
3. **+15%**: Sell 1/3 of shares
4. **Tight SL (3%)**: After first partial, exit remaining if price <= entry x 0.97 (skipped if Nifty shielded)
5. **Upper Keltner band (daily)**: Sell remaining (EMA20 + 2x ATR14)
6. **Underwater**: 10 trading days below entry — exit all

### Strategy MW Exits
1. **Hard SL**: 6% initial (skipped if Nifty crash shield active)
2. **+6%**: Sell 1/3, SL tightens to 3%
3. **+10%**: Sell 1/3, SL tightens to breakeven
4. **3% SL / Breakeven SL**: After partials (skipped if Nifty shielded)
5. **Weekly upper Keltner band**: Sell remaining (only after first partial)
6. **Underwater**: 25 trading days below entry — exit all

### Strategy T Exits (SECONDARY)
1. **Hard SL**: 5% below entry, shifts to 3% after first partial
2. **+6%**: Sell 1/3
3. **+10%**: Sell 1/3
4. **Upper Keltner band**: Sell remaining
5. **Underwater**: 10 trading days

### Strategy J Exits (SECONDARY)
1. **Support break**: Full exit (skipped if Nifty crash shield active)
2. **+5%**: Sell 50%
3. **+10%**: Sell remaining (or Chandelier: Highest High - 3x ATR14)
4. **Underwater**: 10 trading days

### Nifty Crash Shield (R, MW, J)
- **Rule**: If Nifty 50 has fallen by the same or more % as the stock since entry, skip all stop-losses
- **Logic**: Stock is falling with the market, not on its own weakness
- **Impact**: +1pp net/yr, protects during broad market corrections
- **NOT applied to**: Underwater exits (pure discipline of cutting losers works better), profit exits, Keltner exits

### Underwater Exit
- **Rule**: If a trade is held for N+ trading days and price is still below entry, exit at close
- **Days**: R/T/J = 10 days, MW = 25 days (MW trends take longer to develop)
- **Why**: Cuts "falling knife" positions that drift lower for weeks. Frees capital for better opportunities.

---

## Portfolio Parameters

| Parameter | Value |
|---|---|
| Universe | Nifty 200 |
| Capital | 20 Lakhs |
| Per trade | 2 Lakhs |
| Max positions | 10 (20L / 2L) |
| Entries per day | 3 |
| Primary strategies | R + MW (equal priority) |
| Secondary strategies | T (priority 2), J (priority 3) |
| Nifty crash shield | R, MW, J (not T) |
| R RSI threshold | < 35 (regular), < 60 (hidden) |
| MW ADX threshold | >= 25 |
| Sector momentum | Ranking tiebreaker between R and MW |

---

## Signal Ranking — Strategy Priority + Sector Momentum + ATR%

When multiple signals fire on the same day and you have limited slots (3/day):

1. **Strategy priority**: R = MW (0) > T (2) > J (3)
2. **Sector momentum** (descending): Stocks in rising-RS sectors rank higher
3. **ATR%** (ascending): Among equal momentum, calmer stocks preferred

### Sector Momentum Score
- Measures whether a sector's RS vs Nifty 50 is accelerating or decelerating
- Formula: `delta_5d x 3 + delta_10d x 2 + delta_20d x 1` (weighted sum of RS changes)
- Uses 17 Nifty sectoral indices (IT, Bank, Pharma, Auto, Metal, etc.)

---

## Backtest Results — 11 Years (2015-2025), Nifty 200

### Current: R+MW with Nifty Crash Shield

| Year | Trades | Win | Loss | WR% | PF | Gross | Net | NetR% |
|---|---|---|---|---|---|---|---|---|
| 2015 | 157 | 53 | 104 | 33.8% | 1.26 | +2.7L | +1.4L | +7.1% |
| 2016 | 133 | 55 | 78 | 41.4% | 1.60 | +5.2L | +3.5L | +17.5% |
| 2017 | 143 | 66 | 77 | 46.2% | 2.32 | +8.1L | +5.8L | +28.8% |
| 2018 | 138 | 63 | 75 | 45.7% | 2.26 | +9.1L | +6.6L | +33.0% |
| 2019 | 123 | 54 | 69 | 43.9% | 1.88 | +5.3L | +3.7L | +18.4% |
| 2020 | 177 | 104 | 73 | 58.8% | 2.76 | +17.1L | +12.8L | +64.0% |
| 2021 | 205 | 98 | 107 | 47.8% | 2.03 | +12.4L | +8.9L | +44.6% |
| 2022 | 170 | 80 | 90 | 47.1% | 2.00 | +9.7L | +6.9L | +34.7% |
| 2023 | 157 | 86 | 71 | 54.8% | 3.60 | +14.2L | +10.6L | +52.9% |
| 2024 | 164 | 69 | 95 | 42.1% | 1.70 | +7.1L | +4.9L | +24.6% |
| 2025 | 145 | 60 | 85 | 41.4% | 2.07 | +6.4L | +4.4L | +22.2% |
| **Total** | **1712** | **788** | **924** | **46.0%** | **2.06** | **+97.5L** | **+69.6L** | |

- **Avg net: +31.6%/yr** | **All 11 years profitable**
- **20L -> 89.6L net** (4.5x in 11 years)
- **Sharpe (net): 1.53** | **Sortino: 256** | **Max DD: 0.0%**
- **Expectancy: +5,694/trade (+2.85%)**
- **Payoff Ratio: 1.31x** | Target: >= 1.5x
- **R**: 510 trades, 41% WR, +23.8L (24%)
- **MW**: 1202 trades, 48% WR, +73.7L (76%)

### Charges & Tax
- Gross: +97.5L (100%)
- Charges: 8.6L (8.8%)
- Tax (STCG 20%): 19.3L (19.8%)
- **Net: +69.6L (71.4%)**

---

## Filters/Rules Tested and Rejected

| Filter | Result | Verdict |
|---|---|---|
| Entry: RSI < 70 | Worse | Rejected |
| Entry: 15d return > 10% | Worse | Rejected |
| Entry: Dist from EMA50 | Worse | Rejected |
| Entry: ATR + RSI combos | Worse | Rejected |
| Entry: Volume filter | Worse | Rejected |
| Entry: SMA 200 filter | Worse | Rejected |
| Exit: 2.5x ATR Keltner | Worse | Rejected |
| Exit: T Chandelier for Stage 2 | -1.02L | Rejected |
| Nifty shield for underwater exit | Neutralized benefit | Rejected |
| Earnings blackout | -14.23L | Rejected |
| EMA20 rising filter (T) | 0/10 years positive | Rejected |
| Momentum smoothness (T) | 4/10 years positive | Rejected |
| MW hidden divergence filter | Neutral to negative | Rejected |
| 3WTC strategy | Negative year (2018), single-digit returns | Rejected |
| J+T as primary strategies | Dilutes portfolio, competes for R/MW slots | Deprioritized |

---

## Code Paths

The strategy runs in three places with identical logic:

| Feature | Backtest Engine | Live Signals | UI API |
|---|---|---|---|
| File | `data/momentum_backtest.py` | `data/live_signals_engine.py` | `ui/app.py` |
| Gap-down filter | All strategies | All strategies | All strategies |
| Sector momentum | `rank_by_sector_momentum=True` | Sector RS score | R=MW > T > J -> momentum -> ATR% |
| Nifty crash shield | R, MW, J | R, MW, J | nifty_at_entry in metadata |
| R structural SL | 1% below swing low | 1% below swing low | metadata.r_swing_low_stop |
| MW hard SL | 6% / 3% / breakeven | 6% / 3% / breakeven | Per stage |
| R partials (portfolio) | +8% / +15% | +6% / +10% | 3-stage exit |
| MW partials | +6% / +10% | +6% / +10% | Per stage |
| Underwater | R/T/J: 10d, MW: 25d | R/T/J: 10d, MW: 25d | Per strategy |

---

## Refinement Targets

| Metric | Current | Target | Gap |
|---|---|---|---|
| **Win Rate** | 46.0% | >= 50% | +4pp |
| **Payoff Ratio** | 1.31x | >= 1.5x | +0.19x |
| **Net Return** | 31.6%/yr | >= 24%/yr | Already exceeded |
| **PF** | 2.06 | >= 1.5 | Already exceeded |

Once WR and Payoff targets are achieved: scale to multiple accounts.

---

## Risk Notes

- 2015 is the weakest year (+7.1% net) — survivorship bias (current Nifty 200 stocks tested on 2015 data)
- Results are based on daily closing prices. Live execution at different prices will cause slippage
- yfinance data variability: backtest numbers can shift ~0.5-1% between runs due to Yahoo Finance adjusting historical prices
- ~123-205 trades per year (avg 156), manageable with EOD analysis
- Capital check: entries are skipped if trade cost would exceed available capital
