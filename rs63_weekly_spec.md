# RS63_Weekly — Full Implementation Spec
**Version:** 1.0  
**Status:** Backtest candidate; not yet validated  
**Purpose:** Weekly-cadence variant of RS63 Satellite for live executability at reduced operational touchpoints

---

## 1. Overview

RS63_weekly is a weekly-cadence variant of the RS63 Satellite strategy. It preserves RS63's entry filters, exit rules, and position sizing mechanics, but shifts signal generation and entry execution from daily to weekly cadence. Exits retain intraday responsiveness for the hard stop (X1); the other exit rules (X2–X5) move to weekly evaluation.

**Design goals:**
- Reduce operational load from daily screen time to ~30 min/week
- Eliminate the "25 signals on one day" decision-load problem via cross-sectional Z-score ranking
- Preserve RS63's entry logic and exit discipline
- Match Mom15's ranking methodology for methodological consistency

---

## 2. Data Requirements

| Item | Value |
|------|-------|
| Universe | Nifty 200 PIT constituents |
| Price data | Daily OHLCV for all constituents + Nifty 200 benchmark |
| Data snapshot | `rs63_4015d_pit_20260406.pkl` (frozen for reproducibility, per Invariant 10) |
| Backtest period | 2015-04-09 to 2026-04-02 (11 years) |
| Trading calendar | NSE trading days only |

---

## 3. Pre-Computed Indicators (Daily)

Computed once per stock at the start of the backtest, indexed by trading date:

| Indicator | Formula | Window |
|-----------|---------|--------|
| RS63_raw | `(stock_close / stock_close[t-63]) / (nifty200_close / nifty200_close[t-63]) − 1` | 63 trading days |
| RS63 | 5-day SMA of RS63_raw | 5-day |
| RSI_raw | EWM RSI with alpha = 1/14 | 14-day |
| RSI | 3-day SMA of RSI_raw | 3-day |
| IBS | `(close − low) / (high − low)` | intraday |
| SMA63 | 63-day simple moving average of close | 63-day |
| vol_20d_avg | 20-day simple moving average of volume | 20-day |
| vol_ratio | `today_volume / vol_20d_avg` | daily |

Keep all of these as columns on per-stock DataFrames indexed by date.

---

## 4. Entry Logic

### 4.1 Daily Signal Logging

Every trading day `t` (Mon–Fri), for each stock in the universe, check all of:

| Rule | Condition |
|------|-----------|
| E1 | RS63[t] > 0 |
| E2 | SKIPPED (no historical 1H data; live-only filter) |
| E3 | RSI[t] > 50 |
| E4 | IBS[t] > 0.5 |
| E5 | close[t] > open[t] (green candle) |
| E6 | vol_ratio[t] >= 1.0 |

If all pass, append `(stock, t)` to a weekly log.

### 4.2 Weekly Pool Construction (Friday EOD)

On each Friday `F` (or the last trading day of the week if Friday is a holiday):

```python
# Step 1: Gather all stocks that triggered any day this week (Mon-Fri)
week_triggers = set()
for day in trading_days_in_week(F):
    for stock in stocks_that_triggered_on(day):
        week_triggers.add(stock)

# Step 2: Re-qualify against Friday EOD data
candidate_pool = []
for stock in week_triggers:
    if (stock.RS63[F]      > 0 and
        stock.RSI[F]       > 50 and
        stock.IBS[F]       > 0.5 and
        stock.close[F]     > stock.open[F] and
        stock.vol_ratio[F] >= 1.0):
        candidate_pool.append({
            'ticker':      stock.ticker,
            'rs63':        stock.RS63[F],
            'rsi':         stock.RSI[F],
            'friday_close': stock.close[F],
            'trigger_day': first_trigger_day_this_week(stock),  # diagnostic
        })
```

**Note on trigger_day tracking:** record the earliest day of the week on which the stock triggered (Mon=0, Tue=1, …, Fri=4). Used only for diagnostic output; does not affect ranking.

### 4.3 Ranking (Cross-Sectional Z-Score)

```python
if len(candidate_pool) < 3:
    # Fallback: raw RS63-magnitude ranking
    candidate_pool.sort(key=lambda s: s['rs63'], reverse=True)
else:
    rs63_values = [s['rs63'] for s in candidate_pool]
    rsi_values  = [s['rsi']  for s in candidate_pool]

    mean_rs63, std_rs63 = mean(rs63_values), std(rs63_values)
    mean_rsi,  std_rsi  = mean(rsi_values),  std(rsi_values)

    # Guard against zero std (all candidates have identical value)
    if std_rs63 == 0 or std_rsi == 0:
        candidate_pool.sort(key=lambda s: s['rs63'], reverse=True)
    else:
        for s in candidate_pool:
            z_rs63 = (s['rs63'] - mean_rs63) / std_rs63
            z_rsi  = (s['rsi']  - mean_rsi)  / std_rsi
            s['score'] = 0.5 * z_rs63 + 0.5 * z_rsi

        candidate_pool.sort(key=lambda s: s['score'], reverse=True)
```

`candidate_pool` is now ranked descending; index 0 is the strongest candidate.

### 4.4 Monday Execution

On Monday `M` (the next trading day after Friday `F`):

```python
nav = cash + sum(position.shares * position.price_friday_close
                 for position in open_positions)
slot_size = nav / 7
open_slots = 7 - len(open_positions)

for candidate in candidate_pool[:open_slots]:
    entry_price = candidate.open_price_monday  # Monday's open
    shares = int(slot_size // entry_price)
    if shares <= 0:
        continue
    cost = shares * entry_price
    charges = calc_charges(cost, 'buy')
    if cost + charges > cash:
        break
    cash -= (cost + charges)
    open_positions[candidate.ticker] = {
        'entry_date':    M,
        'entry_price':   entry_price,
        'shares':        shares,
        'rsi_low_count': 0,
        'trigger_day':   candidate['trigger_day'],
    }
```

**Edge cases:**
- If Monday is a holiday, use the next trading day
- If `open_slots == 0`, no entries this week
- If `candidate_pool` is empty, no entries this week (track as "dead week")

---

## 5. Exit Logic (Hybrid Cadence)

### 5.1 X1 — Hard Stop (Daily, Intraday)

Evaluated every trading day for every open position:

```python
if position.shares > 0:
    if today_low <= position.entry_price * 0.92:
        exit_price = position.entry_price * 0.92  # backtest proxy
        close_position(ticker, exit_price, reason='HARD_SL_8%', date=today)
```

Backtest proxy: exit at the stop price (entry × 0.92) on the same day the intraday low touches it. Live: this will be a broker GTT order.

### 5.2 X2–X5 — Weekly Evaluation (Friday EOD → Monday Open)

On each Friday `F`, for every open position, check:

| Rule | Condition |
|------|-----------|
| X2 (RS breakdown) | RS63[F] < 0 |
| X3 (Trend break) | close[F] < SMA63[F] |
| X4 (RSI exhaustion) | RSI[F] < 40 AND RSI[F-1] < 40 AND RSI[F-2] < 40 (3 consecutive days) |
| X5 (Time stop) | `(F - entry_date).days >= 56` AND `(close[F] / entry_price − 1) < 0.03` |

If any of X2–X5 fires, mark the position for exit at Monday's open.

**Execution order:**
1. Check X1 daily — if triggered, exit that day
2. On Friday EOD, check X2–X5 — mark for Monday exit
3. On Monday open: first execute pending exits (from X2–X5), then compute available slots, then execute entries

*Why this order: exits free up slots before entries are evaluated.*

---

## 6. Position Sizing

| Parameter | Value |
|-----------|-------|
| Starting capital | ₹10,00,000 |
| Max positions | 7 |
| Slot size | current_NAV / 7 recomputed at each entry |
| Re-entry cooloff | None |

---

## 7. Transaction Costs (Zerodha Delivery)

| Charge | Rate |
|--------|------|
| STT | 0.1% of turnover (both sides) |
| Exchange txn | 0.00307% |
| SEBI | 0.0001% |
| Stamp duty | 0.015% (buy side only) |
| GST | 18% on exchange + SEBI |
| Round-trip | ~0.22% of position |

Deduct charges from cash at both entry and exit.

---

## 8. Metrics to Report

### 8.1 Core Metrics (match current RS63 baseline report)
- Total return %, CAGR
- Total trades, Win rate, Profit factor
- Avg win, Avg loss
- Avg hold days
- Year-by-year returns
- Exit reason breakdown: `HARD_SL_8%`, `RS63_NEG`, `TREND_BREAK`, `RSI<40_3D`, `TIME_STOP`
- Final NAV

### 8.2 Weekly-Specific Diagnostic Metrics
- Avg candidates per week (pool size)
- Weeks with 0 candidates (dead weeks — fully under-deployed)
- Weeks with >20 candidates (high selection pressure)
- Avg slot utilization — fraction of slots filled on average
- Avg Z-score of entered trades (rank 1 through top-N)
- Avg Z-score gap between top-N and rank-(N+1)

### 8.3 Trigger-Day Analysis (diagnostic only)

Group all entered trades by `trigger_day` (Mon=0 through Fri=4). For each group report:
- Count, Avg P&L %, Win rate

*Purpose: diagnose whether trigger-day-of-week systematically correlates with outcome.*

### 8.4 Exit Timing Analysis
- X1 count (daily, intraday)
- X2–X5 count (weekly, each separately)
- Avg time between Friday-signal and Monday-exit for X2–X5 exits

---

## 9. Backtest Execution Order (Per Trading Day)

```
FOR each trading day t FROM 2015-04-09 TO 2026-04-02:

    # 1. Intraday: check X1 for all open positions
    FOR position IN open_positions:
        IF stock.low[t] <= position.entry_price * 0.92:
            exit at entry_price * 0.92, reason='HARD_SL_8%'

    # 2. EOD: log daily signal triggers for the week
    FOR stock IN nifty200_pit_on_date(t):
        IF all_entry_filters_pass(stock, t):
            week_log[week_of(t)].add(stock)

    # 3. If t is last trading day of week: evaluate weekly exits + build pool
    IF is_last_trading_day_of_week(t):
        FOR position IN open_positions:
            IF x2 OR x3 OR x4 OR x5 triggers:
                mark_for_exit_on_next_monday(position, reason)
        candidate_pool = build_and_rank_weekly_pool(week_log[week_of(t)], t)

    # 4. If t is first trading day of week: execute pending exits then entries
    IF is_first_trading_day_of_week(t):
        FOR position IN marked_for_exit:
            exit at stock.open[t], reason=marked_reason
        nav = cash + sum_of_open_position_values_at_prev_close()
        slot_size = nav / 7
        open_slots = 7 - len(open_positions)
        FOR candidate IN last_week_candidate_pool[:open_slots]:
            enter at stock.open[t]

    # 5. End of day: NAV snapshot
    nav_history[t] = cash + sum_of_open_position_values_at_close(t)
```

---

## 10. Invariants

1. **No lookahead:** signal at time `t` uses only data from `t` and earlier
2. **Execution prices are real market prices:** Monday open for entries; intraday low for X1 stop-out proxy; Monday open for X2–X5 exits
3. **Max positions ≤ 7** at all times
4. **Slot size recomputed at each entry** (NAV/7 compounds with portfolio growth)
5. **Pool re-qualification is mandatory:** never enter a stock on Monday that didn't pass Friday's re-qualification
6. **Weekly-cadence consistency:** signal decisions only on last-trading-day-of-week; execution only on first-trading-day-of-week; X1 is the sole exception
7. **Same data snapshot as RS63 baseline:** `rs63_4015d_pit_20260406.pkl`

---

## 11. Pass/Fail Criteria for Adoption

Against current daily RS63 baseline (CAGR 24.0%, PF 3.08), with ±2.5pp CAGR / ±0.40 PF noise floor:

| Outcome | Verdict |
|---------|---------|
| CAGR ≥ 22.0% AND PF ≥ 2.80 | **Ship it.** Operational simplicity worth small CAGR cost. |
| CAGR 18–22% AND PF 2.3–2.8 | **Borderline.** Dig into diagnostics before deciding. |
| CAGR < 18.0% OR PF < 2.30 | **Reject.** Weekly cadence breaks strategy too much. |

**Secondary checks (for shipping):**
- Avg slot utilization ≥ 70%
- Dead weeks < 30 out of ~575 total weeks
- No single year worse than −18%

---

## 12. Expected Directional Outcomes

| Metric | Expected change |
|--------|----------------|
| CAGR | down 1–4pp (some daily signals lost) |
| PF | neutral to up (Z-score ranking probably cleaner than volume ranking) |
| Win rate | up (better entry selection) |
| Avg hold | up (weekly exits slower than daily except X1) |
| Hard stop count | down (fewer bad marginal entries survive Z-ranking) |
| Trade count | down meaningfully (weekly entries vs. daily) |

---

## 13. What to Do with the Result

1. Backtest runs on frozen snapshot; report all §8 metrics
2. Compare against §11 pass/fail thresholds
3. If **pass** → document as "RS63 Weekly" alternative in memory
4. If **borderline** → inspect trigger-day analysis and dead-week distribution before deciding
5. If **reject** → document failure mode and move on

---

## 14. Implementation Notes

Re-use `rs63_pit_report.py` as the base. Most code is reusable: indicator computation, cache loading, exit rules, transaction costs, NAV tracking.

**Changes needed from daily baseline:**
1. Replace "enter on daily signal" with: aggregate into weekly pool → re-qualify Friday EOD → Z-score rank → enter Monday open
2. Add weekly cadence helpers: `is_last_trading_day_of_week(t)`, `is_first_trading_day_of_week(t)`
3. Add Z-score computation function
4. Add pool re-qualification on Friday EOD
5. Modify exit logic: defer X2–X5 to Monday execution (mark + execute)
6. Add diagnostic metrics to output (§8.2–8.4)

**Script name:** `rs63_weekly_report.py` (separate from daily baseline — do not modify `rs63_pit_report.py`)  
**Output format:** match `rs63_pit_report.py` for easy side-by-side comparison, with diagnostic section appended  
**Data:** load `rs63_4015d_pit_20260406.pkl` same as daily baseline
