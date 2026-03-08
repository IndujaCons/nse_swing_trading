# Relative Strength Trading System

## Zerodha Brokerage (Equity Delivery, NSE)

Brokerage: **₹0** (free for equity delivery)

### Statutory Charges Formula

| Charge | Rate | Applied On |
|--------|------|------------|
| STT | 0.1% | Total turnover (buy + sell) |
| Exchange txn | 0.00307% | Total turnover (buy + sell) |
| SEBI | 0.0001% | Total turnover (buy + sell) |
| Stamp duty | 0.015% | Buy side only |
| GST | 18% | Exchange txn + SEBI charges |

### Round-trip Cost Formula

For a buy/sell combo with buy turnover `B` and sell turnover `S`:

```
total_turnover = B + S
stt            = 0.001    * total_turnover
exchange       = 0.0000307 * total_turnover
sebi           = 0.000001  * total_turnover
stamp          = 0.00015   * B
gst            = 0.18      * (exchange + sebi)

total_charges  = stt + exchange + sebi + stamp + gst
```

### Example: ₹2L position (breakeven trade)

Buy ₹2,00,000 + Sell ₹2,00,000 = turnover ₹4,00,000

- STT: ₹400
- Exchange: ₹12.28
- SEBI: ₹0.40
- Stamp: ₹30
- GST: ₹2.28
- **Total: ~₹445 per round trip**
- **As % of position: ~0.22%**

### Verified Against Zerodha Calculator

**Trade 1:** Buy 1000, Sell 1100, Qty 800 (turnover 16.8L)
- Charges: ₹1,862.85 (formula gives ₹1,862.85)

**Trade 2:** Buy 1000, Sell 1100, Qty 200 (turnover 4.2L)
- Charges: ₹465.71 (formula gives ₹465.71)

### Tax

- **STCG (Short-Term Capital Gains):** 20% on net profit
- Holding period < 12 months = short-term

### Expense Deduction for STCG

All charges **except STT** can be claimed as expenses to reduce taxable profit:

```
claimable_expenses = exchange + sebi + stamp + gst   (NOT stt)
taxable_profit     = gross_profit - stt - claimable_expenses - claimable_expenses_as_deduction
                   = gross_profit - total_charges
                   (but STT is NOT deductible, so effectively:)

taxable_profit     = gross_profit - (exchange + sebi + stamp + gst)
stcg_tax           = 0.20 * taxable_profit
net_profit         = gross_profit - total_charges - stcg_tax
```

- **STT** is paid but NOT deductible under Income Tax Act
- **Exchange txn, SEBI, Stamp duty, GST** are all deductible expenses
