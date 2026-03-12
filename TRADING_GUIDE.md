# Swing Trading System — A Complete Guide

A practical guide to understanding swing trading and the R+MW strategy system, written for someone with zero trading background.

---

## Part 1: What is Swing Trading?

### The Basic Idea

Imagine you buy a stock today and sell it after 2-3 weeks when it has gone up. That's swing trading. You're not a day trader (buying and selling within hours) and you're not an investor (holding for years). You sit in the sweet spot — holding positions for **5 to 30 days**, capturing short-term price "swings."

Why swing trading?
- You don't need to watch screens all day
- You make decisions after market hours using end-of-day (EOD) data
- Each trade has a clear entry price, stop-loss, and profit target
- You can do this alongside a full-time job

### Trending vs Rangebound Markets

Markets move in two modes:

**Trending Market** — Prices move consistently in one direction. An uptrend makes higher highs and higher lows, like climbing stairs. Stocks in an uptrend are the best candidates for swing trades because momentum is on your side.

```
        /\    /\     <- Higher highs
       /  \  /  \
      /    \/    \   <- Higher lows
     /
    /                <- Uptrend
```

**Rangebound Market** — Prices bounce between a ceiling (resistance) and a floor (support) without making progress in either direction. Trading here is harder — you can still profit from bounces off support, but gains are smaller and less reliable.

```
    -------- Resistance --------
    /\    /\    /\    /\
   /  \  /  \  /  \  /  \
  /    \/    \/    \/    \
    --------- Support ----------
```

Most of the time, individual stocks are in one mode or the other even if the overall market is mixed. The system has strategies designed for both conditions.

---

## Part 2: What is a Trading Strategy?

A trading strategy is a **set of rules** that tell you:
1. **When to buy** (entry signal)
2. **When to sell for profit** (exit/target)
3. **When to sell to limit losses** (stop-loss)

Without a strategy, you're gambling. With a strategy, you're running a business.

### The Three Numbers That Matter

If you run a strategy on 100 trades, three statistics determine whether you made money:

**Win Rate (WR%)** — What percentage of your trades were profitable?
- 50% win rate means 50 winners out of 100 trades
- You do NOT need 80-90% win rate. Even 46-50% is profitable if your wins are bigger than your losses.

**Payoff Ratio (Avg Win / Avg Loss)** — How much do you make when right vs how much do you lose when wrong?
- If you make Rs 12,500 on winners and lose Rs 9,500 on losers, your ratio is 1.31x
- Target: **1.5x or higher**

**Profit Factor (PF)** = (Total profits from winners) / (Total losses from losers)
- PF > 1.0 means you're profitable
- PF > 1.7 is required to survive Indian charges + 20% STCG tax
- PF > 2.0 is excellent

Here's the key insight: **A 46% win rate with 1.3x payoff ratio is very profitable.** You don't need to be right most of the time — you just need your winners to be bigger than your losers.

Example with 100 trades at Rs 2,00,000 per trade:
```
46 winners x Rs 12,500 avg profit = Rs 5,75,000 gained
54 losers  x Rs  9,500 avg loss   = Rs 5,13,000 lost
                          Net P&L = Rs   62,000 profit (before charges/tax)
```

This is why **cutting losses short** (stop-loss) and **letting winners run** (staged exits) is the foundation of every good strategy.

---

## Part 3: Trading Basics

### Entry — Getting In

A good entry is when you buy a stock at a price where the odds of it going up are higher than going down. You look for **confirmation signals**:
- The stock shows a technical pattern (divergence, momentum crossover)
- The candle is green (close > open, meaning buyers won that day)
- There's no gap-down (today's open is not below yesterday's close, which would signal panic selling)

A bad entry is buying just because "the stock fell a lot." Without a setup, you're catching a falling knife.

### Exit — Getting Out

There are two kinds of exits:

**Profit exit** — The stock hit your target, you sell and book gains.

**Stop-loss exit** — The stock went against you, you sell to limit damage.

Both are equally important. Many beginners hold losers hoping they'll recover ("it'll come back") — this is how small losses become large losses.

### What is a Stop-Loss (SL)?

A stop-loss is a pre-decided price level where you will sell **no matter what**. It protects you from catastrophic losses.

Example:
```
Entry price:     Rs 1,000
Stop-loss:       Rs 950 (5% below entry)
Potential loss:  Rs 50 per share = 5%
```

If the stock drops to Rs 950, you sell immediately. You don't hope, you don't pray, you don't check the news. You exit. The loss is small and manageable.

Without a stop-loss, a 5% dip can turn into a 20-30% loss, and now you need the stock to rise 40% just to break even.

**Golden rule: Always know your stop-loss BEFORE you enter a trade.**

### Nifty Crash Shield

One exception to the stop-loss rule: if the entire Nifty 50 index has fallen by the same or more percentage as your stock since entry, the SL is suppressed. This means the stock isn't falling due to its own weakness — it's falling because the entire market is crashing. In such cases, holding through the crash often leads to recovery.

---

## Part 4: Key Concepts

### Support and Resistance

**Support** is a price level where a stock has repeatedly stopped falling and bounced back up. Think of it as a floor — buyers step in at this price because they consider it good value.

**Resistance** is a price level where a stock has repeatedly stopped rising and pulled back. Think of it as a ceiling — sellers take profits here because they consider it expensive.

```
    Rs 550 -------- Resistance (sellers here) --------
              /\         /\
             /  \       /  \     Stock bounces between
            /    \     /    \    support and resistance
           /      \   /      \
    Rs 480 --------- Support (buyers here) -----------
```

### Keltner Channel

The Keltner Channel is a technical indicator that draws an **envelope** around the price. It consists of:

- **Middle line**: EMA(20) — the 20-day average price
- **Upper band**: EMA(20) + 2 x ATR(14) — the "expensive" zone
- **Lower band**: EMA(20) - 2 x ATR(14) — the "cheap" zone

**ATR(14)** is the Average True Range over 14 days — it measures how much the stock typically moves in a day.

```
    ====================================  <- Upper Keltner (expensive)
         *  *     *
        * ** *   * *    *
    ~~~*~~~~~~~~~*~~*~~*~*~~~~~~~~~  <- EMA(20) (middle)
       *        *    **
      *        *
    ====================================  <- Lower Keltner (cheap)
```

When price reaches the upper Keltner band, the stock is stretched and may pull back — this is where we take profits.

### RSI (Relative Strength Index)

RSI measures how "overbought" or "oversold" a stock is on a scale of 0-100:
- **RSI > 70**: Overbought (may pull back)
- **RSI < 30**: Oversold (may bounce)
- **RSI 30-70**: Neutral zone

### RSI Divergence

This is a powerful reversal signal. It occurs when price and RSI disagree:

**Bullish RSI Divergence** — Price makes a **lower low** (new bottom) but RSI makes a **higher low** (less oversold). This means selling pressure is weakening even though price is still falling. A reversal upward is likely.

```
    Price:  100 --> 90 --> 85     (lower lows -- looks weak)
    RSI:     25 --> 22 --> 28     (higher lows -- actually strengthening!)
                              ^
                    Divergence = buying opportunity
```

The market is telling you: "Yes the price dropped, but the momentum behind the drop is fading." Smart money is quietly accumulating.

### ADX (Average Directional Index)

ADX measures the **strength** of a trend on a scale of 0-100, regardless of direction:
- **ADX < 20**: Weak/no trend
- **ADX 20-25**: Trend emerging
- **ADX >= 25**: Strong trend confirmed
- **ADX > 40**: Very strong trend

ADX is used alongside **DI+ and DI-** (Directional Indicators):
- **DI+ > DI-**: Uptrend (buyers dominating)
- **DI- > DI+**: Downtrend (sellers dominating)

Strategy MW uses weekly ADX crossing above 25 with DI+ > DI- as its entry signal — it catches the moment a trend shifts from "maybe" to "confirmed."

---

## Part 5: Multi-Stage Exits — The Secret Sauce

Most beginners either:
- Sell too early (take small profit, miss the big move)
- Sell too late (hold too long, give back all gains)

The solution is **multi-stage exits** — selling in portions at different levels.

### Example: R Strategy 2-Stage Exit

You buy 90 shares at Rs 1,000. Instead of selling all 90 at one target:

```
Stage 1: Price hits +8% (Rs 1,080)
         -> Sell 30 shares (1/3), lock in Rs 2,400 profit
         -> SL shifts up to 3% below entry (Rs 970)

Stage 2: Price hits +15% (Rs 1,150) OR upper Keltner band
         -> Sell next 30 shares (1/3)
         -> Remaining 30 ride to Keltner exit

Final: Price hits upper Keltner band (Rs 1,200)
       -> Sell remaining 30 shares, lock in Rs 6,000 profit
       -> Total: Rs 12,600 on a Rs 90,000 position (+14%)
```

**Why this works:**
- Stage 1 locks in guaranteed profit early (you can't lose on this trade anymore)
- The tighter stop-loss after Stage 1 protects your remaining position
- Stages 2 and Keltner let the winners run to natural momentum exhaustion
- If the stock reverses after Stage 1, you still made money on the first 1/3

---

## Part 6: The Strategies

The system runs two primary strategies (R and MW) on Nifty 200 stocks, with two secondary strategies (T and J) shown for reference.

---

### Strategy R — Bullish RSI Divergence (PRIMARY)

**What it does:** Buys stocks showing bullish RSI divergence — price is making new lows but momentum is improving. Catches reversals.

**Two types:**
1. **Regular divergence** (reversal): Price makes a lower low but RSI makes a higher low. RSI < 35 required.
2. **Hidden divergence** (continuation): Price makes a higher low but RSI makes a lower low — uptrend intact. RSI < 60, price > EMA(50).

**Entry Conditions — Regular Divergence:**
| Condition | Why |
|-----------|-----|
| Two swing lows where price: lower low, RSI(14): higher low | Bullish divergence confirmed |
| RSI divergence >= 3 points | Meaningful divergence, not noise |
| RSI(14) < 35 at current swing low | Stock is oversold |
| Structural stop 0-6% from entry | Risk is manageable |
| Green candle (close > open) | Reversal candle |
| No gap-down | No continued panic |

**Entry Conditions — Hidden Divergence:**
| Condition | Why |
|-----------|-----|
| Two swing lows where price: higher low, RSI(14): lower low | Hidden bullish divergence |
| RSI divergence >= 5 points | Stronger threshold for continuation |
| RSI(14) < 60 at current swing low | Relaxed (stock in uptrend) |
| Close > EMA(50) | Confirms uptrend |
| Structural stop 2-6% from entry | Min 2% for hidden |
| Green candle, no gap-down | Confirmation |

**Stop-Loss:**
- **Structural stop**: 1% below the most recent swing low
- After first partial (+8%), SL shifts to 3% below entry
- **Nifty crash shield**: Skip SL if Nifty 50 fell same or more since entry

**Exit (Portfolio 2-stage):**
| Stage | Trigger | Action |
|-------|---------|--------|
| 1 | Price reaches entry + 8% | Sell 1/3 of position |
| 2 | Price reaches entry + 15% | Sell 1/3, SL shifts to 3% |
| Final | Upper Keltner band (daily) | Sell remaining |
| Safety | 10 trading days underwater | Exit all remaining |

---

### Strategy MW — Weekly ADX Momentum (PRIMARY)

**What it does:** Buys stocks where weekly ADX just crossed above 25 with DI+ > DI- — confirming a strong new uptrend is beginning.

**Entry Conditions (ALL must be true):**
| Condition | Why |
|-----------|-----|
| Weekly ADX >= 25 and rising (curr > prev) | Trend strength confirmed |
| Weekly DI+ > DI- | Uptrend direction confirmed |
| Green candle (close > open) | Daily confirmation |
| No gap-down | No overnight panic |

**Stop-Loss:**
- **Stage 0**: 6% hard SL
- **After first partial (+6%)**: Tightens to 3%
- **After second partial (+10%)**: Tightens to breakeven
- **Nifty crash shield**: Skip SL if Nifty 50 fell same or more since entry

**Exit (2-stage):**
| Stage | Trigger | Action |
|-------|---------|--------|
| 1 | Price reaches entry + 6% | Sell 1/3 |
| 2 | Price reaches entry + 10% | Sell 1/3 |
| Final | Weekly upper Keltner band | Sell remaining (after P1) |
| Safety | 25 trading days underwater | Exit all remaining |

---

### Strategy T — Keltner Channel Pullback (SECONDARY)

**What it does:** Buys strong stocks that have pulled back from the upper Keltner band to the EMA(20) midline.

**Entry:** Price within 1% of EMA(20), touched upper Keltner in last 10 bars, green candle, IBS > 0.5, no gap-down.

**Exit:** 5% hard SL (3% after first partial), +6% sell 1/3, upper Keltner sell remaining. 10d underwater exit.

---

### Strategy J — Weekly Support Bounce (SECONDARY)

**What it does:** Buys stocks at 26-week support levels showing bounce signs.

**Entry:** Within 0-3% of weekly support, IBS > 0.5, green candle, CCI(20) > -100, no gap-down.

**Exit:** Support break SL (with Nifty shield), +5% sell half, +10% or Chandelier (HH - 3x ATR14) sell rest. 10d underwater exit.

---

## Part 7: How the Strategies Work Together

The system runs R and MW as primary strategies on Nifty 200 stocks. T and J signals are still displayed but ranked lower.

| Strategy | Market Condition | Style | Priority |
|----------|-----------------|-------|----------|
| **R** | Bottoming / reversal | RSI Divergence | 0 (highest) |
| **MW** | Trending / momentum | ADX Breakout | 0 (equal to R) |
| **T** | Trending / pullback | Keltner Pullback | 2 (secondary) |
| **J** | Rangebound / support | Support Bounce | 3 (lowest) |

### Capital Allocation

- **Capital**: Rs 20 Lakhs
- **Per trade**: Rs 2 Lakhs
- **Max positions**: 10 (20L / 2L)
- **Max new entries**: 3 per day

### Signal Ranking

When R and MW both fire on the same day:
1. **Strategy priority**: R = MW (equal) > T > J
2. **Sector momentum** (descending): Stocks in sectors with rising relative strength rank higher
3. **ATR%** (ascending): Among equal momentum, calmer stocks preferred

**Sector momentum** measures whether a sector's RS vs Nifty 50 is accelerating or decelerating. A sector at -11% RS but rising momentum is recovering fast — it ranks higher than a flat +4% sector.

### Risk Management

Every trade has a predefined maximum loss:
- **R**: 1% below swing low (typically 2-6%)
- **MW**: 6% hard stop
- **T**: 5% hard stop
- **J**: Distance to weekly low support

With Rs 2,00,000 per trade and ~6% max stop, the worst-case loss per trade is Rs 12,000.

### Nifty Crash Shield (R, MW, J)

If Nifty 50 has fallen by the same or more % as the stock since entry, stop-losses are suppressed. The stock is falling with the market, not on its own weakness. This prevents unnecessary stop-outs during broad market corrections.

---

## Part 8: Backtest Results

The R+MW system has been backtested over 11 years (2015-2025) on Nifty 200 stocks with Rs 20 lakh capital, Rs 2 lakh per trade, and 3 entries per day. Signals ranked by strategy priority (R=MW) then sector momentum then ATR%.

Charges: Zerodha delivery (Rs 0 brokerage, STT 0.1% both sides, exchange txn 0.00307%, SEBI 0.0001%, stamp duty 0.015% buy side, GST 18%). Tax: STCG 20%.

```
R+MW Portfolio — Nifty 200, 20L, 2L/trade, 3/day, Sector Momentum, Nifty Crash Shield
R: RSI<35 regular + hidden divergence, structural SL (max 6%)
MW: Weekly ADX >= 25, DI+ > DI-, 6% hard SL

Year   Tr    W    L   WR%    PF    AvgW    AvgL  AvgW%  AvgL%  Hold    Gross     Chg     Tax      Net   Ret%  NetR%
2015  157   53  104  33.8  1.26 +12,079  -9,776  +6.0%  -4.9%   21d    +2.7L   76.6K   52.6K    +1.4L +13.5%  +7.1%
2016  133   55   78  41.4  1.60 +12,045 -10,844  +6.0%  -5.4%   23d    +5.2L   65.4K  102.2K    +3.5L +25.9% +17.5%
2017  143   66   77  46.2  2.32 +12,352  -7,848  +6.2%  -3.9%   25d    +8.1L   71.4K  160.2K    +5.8L +40.4% +28.8%
2018  138   63   75  45.7  2.26 +12,849  -9,243  +6.4%  -4.6%   24d    +9.1L   69.9K  180.8K    +6.6L +45.5% +33.0%
2019  123   54   69  43.9  1.88 +11,893  -7,796  +5.9%  -3.9%   24d    +5.3L   60.3K  105.5K    +3.7L +26.7% +18.4%
2020  177  104   73  58.8  2.76 +12,904 -12,947  +6.5%  -6.5%   17d   +17.1L   92.1K  340.8K   +12.8L +85.7% +64.0%
2021  205   98  107  47.8  2.03 +13,109 -10,976  +6.6%  -5.5%   17d   +12.4L  105.5K  246.7K    +8.9L +62.2% +44.6%
2022  170   80   90  47.1  2.00 +12,857 -10,417  +6.4%  -5.2%   18d    +9.7L   85.1K  192.8K    +6.9L +48.6% +34.7%
2023  157   86   71  54.8  3.60 +12,694  -7,694  +6.3%  -3.8%   22d   +14.2L   80.8K  282.7K   +10.6L +71.1% +52.9%
2024  164   69   95  42.1  1.70 +12,800 -10,804  +6.4%  -5.4%   22d    +7.1L   81.6K  141.3K    +4.9L +35.7% +24.6%
2025  145   60   85  41.4  2.07 +12,172  -6,893  +6.1%  -3.4%   22d    +6.4L   70.0K  127.0K    +4.4L +32.1% +22.2%
=============================================================================================================
Total 1712  788  924  46.0  2.06 +12,523  -9,567  +6.3%  -4.8%   21d   +97.5L  858.7K 1932.5K   +69.6L

Avg gross: +44.3%/yr  |  Avg net: +31.6%/yr  |  Neg years: 0
20L -> 117.5L gross -> 89.6L net

Money Flow:
    Gross P&L:         +97.5L (100%)
    Charges:             8.6L (8.8%)
    Tax:                19.3L (19.8%)
    Net:               +69.6L (71.4%)

Risk Ratios:
    Sharpe (net):       1.53  (rf=6%)
    Sortino (net):      256.24
    Calmar:             999.00  (no drawdown)
    Max Drawdown:       0.0%

Trade Quality:
    Expectancy/trade:   +5,694 (+2.85%)
    Payoff Ratio:       1.31x
    Avg holding days:   21

Strategy Breakdown:
    R: 510 trades | +23.8L gross | 41% WR | 242.1K charges
    MW: 1202 trades | +73.7L gross | 48% WR | 616.5K charges
```

### Key Takeaways
- **All 11 years profitable** — worst year +7.1% net (2015)
- **Best year: +64.0% net (2020)**
- **Net return: +31.6%/yr avg** after STT, charges, and 20% STCG tax
- **You keep 71 paise per rupee earned**
- **PF 2.06** — for every Rs 1 lost, you make Rs 2.06
- **MW is the workhorse** — 70% of trades, 76% of gross P&L, 48% WR
- **R is the specialist** — 30% of trades, 24% of gross P&L, 41% WR, catches reversals

---

## Part 9: Charges & Tax (Zerodha Equity Delivery)

### Per-Trade Charges
```
Brokerage:     Rs 0 (free for equity delivery)
STT:           0.1% of total turnover (buy + sell)
Exchange txn:  0.00307% of total turnover
SEBI:          0.0001% of total turnover
Stamp duty:    0.015% of buy side only
GST:           18% of (exchange + SEBI)

Round trip on Rs 2L position: ~Rs 445 (~0.22%)
```

### STCG Tax
- **20% on net profit** (holding < 12 months = short-term)
- STT is **NOT deductible** from capital gains
- Other charges (exchange, SEBI, stamp, GST) **ARE deductible**

### Money Flow Example (11 years)
```
Gross P&L:    +97.5L (100%)
Charges:       -8.6L (8.8%)    <- cost of doing business
Tax:          -19.3L (19.8%)   <- government's share
Net:          +69.6L (71.4%)   <- what you keep
```

---

## Glossary

| Term | Definition |
|------|-----------|
| **ADX** | Average Directional Index — measures trend strength (0-100) |
| **ATR** | Average True Range — measures daily price volatility |
| **Chandelier Exit** | Trailing stop based on highest high minus a multiple of ATR |
| **DI+/DI-** | Directional Indicators — show uptrend (DI+>DI-) or downtrend |
| **EMA** | Exponential Moving Average — weighted moving average favoring recent prices |
| **EOD** | End of Day — analysis done after market close |
| **Gap-down** | When today's open is below yesterday's close |
| **Green candle** | Close is above open (buyers won the day) |
| **IBS** | Internal Bar Strength — where close falls within the day's range |
| **Keltner Channel** | EMA(20) with bands at +-2x ATR(14) |
| **Nifty Crash Shield** | Skip SL if Nifty 50 fell same or more % since entry |
| **Payoff Ratio** | Average Win / Average Loss (target >= 1.5x) |
| **PF** | Profit Factor — total wins divided by total losses |
| **RSI** | Relative Strength Index — momentum oscillator (0-100) |
| **SL** | Stop-Loss — predetermined exit price to limit losses |
| **STCG** | Short-Term Capital Gains — 20% tax on profits held < 12 months |
| **STT** | Securities Transaction Tax — 0.1% on turnover, not tax-deductible |
| **Support** | Price level where buyers have historically stepped in |
| **Swing Low** | A local price bottom confirmed by higher prices on both sides |
| **WR%** | Win Rate — percentage of profitable trades |
