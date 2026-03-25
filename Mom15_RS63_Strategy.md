# Mom15 + RS63 Combined Strategy

## Primary Equity Trading System — ₹20L Portfolio
**Prepared for Jayaraman | Induja Consultancy LLP | March 2026**

---

## Portfolio Structure

| | Mom15 (Core) | RS63 (Satellite) |
|---|---|---|
| **Capital** | ₹10L (50%) | ₹10L (50%) |
| **Signal type** | Normalised momentum score (systematic) | RS63 + RSI + price action (systematic) |
| **Hold period** | 2 months (fixed calendar) | 2–8 weeks (signal-driven) |
| **Positions** | 15 equal-weight (~₹67K each) | 7 equal-weight (~₹1.43L each) |
| **Rebalance** | 6× per year (every 2 months) | Daily scan, trade when signals fire |
| **Stop loss** | None — hold until next rebalance | 8% hard stop |
| **Time commitment** | 30 min, 6× per year | 2 min daily (check exits) |
| **Edge source** | Momentum persistence over months | Buying dips in RS-strong stocks |

**Combined PIT 11yr (2015-2025): +24.1%/yr net | 1 negative year**
**Correlation: 0.56** — genuinely different, good diversification

---

## 1. Mom15 — Core Sleeve (₹10L)

### Methodology
NSE Nifty200 Momentum 30 index methodology adapted for concentrated portfolio.

### Scoring
- Momentum Ratio = Price Return / annualised daily volatility
- MR_12 = 12-month return / vol
- MR_6 = 6-month return / vol
- MR_3 = 3-month return / vol
- Z-score all three across Nifty 200 PIT universe
- Weighted Average Z = 30% × Z_12 + 30% × Z_6 + 40% × Z_3
- Normalized Score = (1+Z) if Z≥0, else (1-Z)^−1

### Selection
- Top 15 stocks by Normalized Momentum Score
- **EPS filter**: TTM EPS growth > 0% — exclude stocks with declining trailing-twelve-month earnings
  - TTM = sum of latest 4 quarterly Diluted EPS vs previous 4Q shifted by 1 quarter
  - Stocks without EPS data pass through (not filtered)
- **Beta cap**: ≤ 1.0 (only stocks less volatile than Nifty 50)
- **Buffer**: Existing stock stays if rank ≤ 30, new stock enters if rank ≤ 10

### Rebalance
Every 2 months (6 times/year): 1 Apr, 1 Jun, 1 Aug, 1 Oct, 1 Dec, 1 Feb.
No stop loss, no regime filter — hold until next rebalance.

### Parameters
```
top_n=15, buffer_in=10, buffer_out=30, rebalance_months=2,
beta_cap=1.0, pit_universe=True, eps_filter=0.0,
w12=0.30, w6=0.30, w3=0.40
```

### Performance (11yr PIT, 2015-2025, ₹20L capital, net after charges + 20% STCG)
| Year | Net% | Year | Net% |
|------|------|------|------|
| 2015 | -10.8% | 2021 | +46.8% |
| 2016 | +0.7% | 2022 | +63.0% |
| 2017 | +24.8% | 2023 | +14.2% |
| 2018 | +15.7% | 2024 | +74.1% |
| 2019 | +13.5% | 2025 | +12.5% |
| 2020 | +8.2% | | |

**Avg: ~+23.9%/yr | 1 negative year | ~362 trades**

---

## 2. RS63 — Satellite Sleeve (₹10L)

### Origin
Adapted from Vivek Bajaj's RS55 framework (StockEdge / Learn2Trade). Modified with 63-day lookback (actual 3 calendar months), smoothed indicators, and simplified single-stage exits.

### Entry Conditions (all must be true on same day)
1. **RS63 (5-day avg) > 0** — stock outperforming Nifty 200 over 63 trading days
2. **RSI(14) (3-day avg) > 50** — internal momentum positive
3. **IBS > 0.5** — close in upper half of day's range (buying pressure)
4. **Green candle** — close > open

### Ranking (when multiple stocks qualify)
Rank by **tightest stop distance** (smallest % distance from 20-day low). This naturally selects calm, near-support stocks in uptrends — effectively "buy the dip in momentum stocks."

### Exit Rules (any one triggers, full position)
1. **8% hard stop** — price ≤ entry × 0.92. Non-negotiable.
2. **RSI(14, 3d avg) < 40 for 3 consecutive days** — momentum lost
3. **RS63 (5d avg) < 0** — stock no longer outperforming the market
4. **Time stop** — 8 weeks held + < 3% gain = dead money, exit

### Sizing
- ₹10L capital ÷ 7 positions = ~₹1.43L per stock
- Max 7 concurrent positions

### Key Design Insights
- **No 200 EMA filter** — RS63>0 already confirms relative uptrend. Adding 200 EMA cost -2%/yr.
- **No volume filter** — best entries happen on normal volume (quiet accumulation). Volume filter cost -5%/yr.
- **No staged exits** — breakeven stops after partial exits created zero-P&L trades that cost charges. Single-stage full exit is simpler and better.
- **No beta filter** — tightest-stop ranking naturally selects calmer stocks.
- **No sector filter** — strong stocks often lead their sector. Requiring sector RS63>0 missed early movers.
- **3-day smoothing on RSI, 5-day on RS63** — reduces daily chop, prevents whipsaw entries/exits.
- **RS63 > RS55** — 63 trading days = actual 3 months. More stable than Fibonacci-based 55 days.

### Performance (11yr PIT, 2015-2025, ₹10L capital, net after charges + 20% STCG)
| Year | Net% | Year | Net% |
|------|------|------|------|
| 2015 | -5.8% | 2021 | +82.9% |
| 2016 | +3.3% | 2022 | +12.2% |
| 2017 | +19.0% | 2023 | +9.8% |
| 2018 | +25.7% | 2024 | +63.3% |
| 2019 | +3.1% | 2025 | +11.4% |
| 2020 | +32.1% | | |

**Avg: +23.5%/yr | 1 negative year | 465 trades | Avg hold 59 days**

### Exit Reason Distribution (11yr)
| Exit | Count | Description |
|------|-------|-------------|
| RS63_NEG | 205 | RS63 turned negative — primary exit |
| RSI_BELOW40_3D | 112 | Momentum collapsed |
| HARD_SL_8PCT | 72 | Thesis failed — 8% stop |
| TIME_STOP | 69 | Dead money after 8 weeks |

---

## 3. Combined Performance

| Year | Mom15 | RS63 | Combined | Year | Mom15 | RS63 | Combined |
|------|-------|------|----------|------|-------|------|----------|
| 2015 | -5.9% | -5.8% | -6.0% | 2021 | +37.5% | +82.9% | +60.1% |
| 2016 | +2.6% | +3.3% | +2.8% | 2022 | +57.3% | +12.2% | +34.8% |
| 2017 | +31.1% | +19.0% | +24.7% | 2023 | +16.0% | +9.8% | +12.9% |
| 2018 | +7.2% | +25.7% | +15.3% | 2024 | +74.1% | +63.3% | +68.5% |
| 2019 | +13.4% | +3.1% | +8.2% | 2025 | +29.6% | +11.4% | +20.3% |
| 2020 | +9.6% | +32.1% | +20.8% | | | | |

**Combined: +24.1%/yr net | 1 negative year (2015 -6.0%) | Correlation 0.56**

### Diversification Benefit
- Mom15 dominates in 2022 (+57%), 2024 (+74%), 2025 (+30%)
- RS63 dominates in 2018 (+26%), 2020 (+32%), 2021 (+83%)
- They rarely fail in the same year → smoother equity curve

---

## 4. Overlap Handling

When the same stock appears in both systems:
- **Allowed** — it means two independent methods agree the stock is strong
- Combined cap: max ₹2.1L in any single stock (₹67K Mom15 + ₹1.43L RS63)
- **Independent entries/exits** — a satellite exit doesn't affect the Mom15 position

---

## 5. Daily Workflow

### Mon-Fri (2 minutes at 3:30 PM)
1. Check RS63 held positions: has any hit 8% SL? RSI < 40 for 3 days? RS63 < 0?
2. If exit triggered → sell at close
3. If slot open → check if any new stock qualifies (RS63>0, RSI>50, IBS>0.5, green)
4. If qualifying stock found → buy at close

### 6× per year (30 minutes at rebalance — 1 Apr, 1 Jun, 1 Aug, 1 Oct, 1 Dec, 1 Feb)
1. Run Mom15 scanner (30/30/40 scoring with 12m+6m+3m)
2. Apply EPS filter (TTM > 0%)
3. Apply buffer rules (in ≤ 10, out ≤ 30)
4. Sell stocks dropped, buy new entrants

---

## 6. Risk Management

| Risk | Rule |
|------|------|
| Single stock max | ₹2.1L (Mom15 ₹67K + RS63 ₹1.43L) |
| Mom15 drawdown | No action — 4-month rebalance handles it |
| RS63 per-trade risk | 8% max loss = ₹11,400 per position |
| RS63 dead money | 8-week time stop if < 3% gain |
| Total portfolio | If both sleeves down > 20%, review but don't panic — backtest shows recovery |

---

## 7. Transaction Costs (Zerodha Equity Delivery)

| Component | Rate |
|-----------|------|
| Brokerage | ₹0 (free) |
| STT | 0.1% of total turnover |
| Exchange txn | 0.00307% of total turnover |
| SEBI | 0.0001% of total turnover |
| Stamp duty | 0.015% of buy side |
| GST | 18% of (exchange + SEBI) |
| **STCG Tax** | **20% on annual net profit** (after deductible expenses) |

---

## 8. Backtesting Protocol

- **Universe**: Nifty 200 Point-in-Time constituents (survivorship-bias-free)
- **Period**: 2015-2025 (11 years)
- **Charges**: Full Zerodha statutory charges applied per trade
- **Tax**: 20% STCG on annual net gains (losses netted within year)
- **All results are NET** — after charges and tax

### Code
- Mom15: `MomentumBacktester.run_momentum30_backtest()` in `data/momentum_backtest.py`
- RS63: `MomentumBacktester.run_rs55_backtest()` in `data/momentum_backtest.py`
- Live signals: `LiveSignalEngine.scan_all()` in `data/live_signals_engine.py`
- UI: `ui/templates/index.html` — Mom15 panel with EPS toggle + RS63 panel

---

## 9. Tested & Rejected (RS63 Evolution)

| Variant | Result | Why rejected |
|---------|--------|--------------|
| Original Vivek Bajaj RS55 (5 filters + staged exits) | +2.3%/yr | Staged exits killed returns |
| Without 21 EMA + Supertrend filters | +2.5%/yr | Barely changed |
| + IBS > 0.5 + green candle | +4.7%/yr | Helped, but staged exits still drag |
| Single-stage: 8%SL + RS21<0 3d + RSI<40 3d | +4.0%/yr | RS21 too tight (91% of exits) |
| Remove RS21, keep RSI<40 3d only | +11.9%/yr | Winners ran longer, but 5 neg years |
| + RS55<0 exit | +13.9%/yr | Good balance, 2 neg years |
| **Switch to RS63 (63d lookback)** | **+19.5%/yr** | **Better than 55d** |
| **Remove 200 EMA filter** | **+21.5%/yr** | **RS63>0 sufficient** |
| + Volume > 20d avg | +16.4%/yr | Filtered out quiet accumulation entries |
| + Sector RS63 > 0 filter | +13.6%/yr | Strong stocks lead their sector |
| + Beta < 1.2 | +18.9%/yr | Tightest-stop ranking already selects calm stocks |
| Skip top 2 RS63, pick 3rd | +13.9%/yr | Highest RS = most volatile, more stops |
| 30d positive streak filter | +10.6%/yr | Too many false breakouts on fresh turns |
| RS123 entry + RS63 exit | +17.7%/yr | Too loose entry, more churn |
| RS123 for everything | +14.4%/yr | Too slow, missed rotations |
| Remove time stop | +16.6%/yr | Dead money sits, blocks better entries |
| DD re-rank (10%, 15%) | -1 to -2%/yr vs baseline | Sells into panic at the worst time |

**Key insight**: The system works because the tightest-stop ranking implicitly selects calm, near-support stocks in uptrends — it's a momentum-filtered dip-buying strategy. Every "momentum-style" addition (highest RS, volume breakout, sector filter) makes it worse because it fights this implicit dip-buying logic.
