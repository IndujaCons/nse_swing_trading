# JTR Swing Trading System — A Complete Guide

A practical guide to understanding swing trading and the JTR strategy system, written for someone with zero trading background.

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
        /\    /\     ← Higher highs
       /  \  /  \
      /    \/    \   ← Higher lows
     /
    /                ← Uptrend
```

**Rangebound Market** — Prices bounce between a ceiling (resistance) and a floor (support) without making progress in either direction. Trading here is harder — you can still profit from bounces off support, but gains are smaller and less reliable.

```
    -------- Resistance --------
    /\    /\    /\    /\
   /  \  /  \  /  \  /  \
  /    \/    \/    \/    \
    --------- Support ----------
```

Most of the time, individual stocks are in one mode or the other even if the overall market is mixed. The JTR system has strategies designed for both conditions.

---

## Part 2: What is a Trading Strategy?

A trading strategy is a **set of rules** that tell you:
1. **When to buy** (entry signal)
2. **When to sell for profit** (exit/target)
3. **When to sell to limit losses** (stop-loss)

Without a strategy, you're gambling. With a strategy, you're running a business.

### The Two Numbers That Matter

If you run a strategy on 100 trades, two statistics determine whether you made money:

**Win Rate (WR%)** — What percentage of your trades were profitable?
- 55% win rate means 55 winners out of 100 trades
- You do NOT need 80-90% win rate. Even 50-55% is excellent if your wins are bigger than your losses.

**Average Win vs Average Loss** — How much do you make when right vs how much do you lose when wrong?
- If you make Rs 8,000 on winners and lose Rs 5,000 on losers, your ratio is 1.6x
- This is called the **reward-to-risk ratio**

**Profit Factor (PF)** = (Total profits from winners) / (Total losses from losers)
- PF > 1.0 means you're profitable
- PF > 1.5 is good
- PF > 2.0 is excellent

Here's the key insight: **A 50% win rate with 1.6x reward/risk is very profitable.** You don't need to be right most of the time — you just need your winners to be bigger than your losers.

Example with 100 trades at Rs 2,00,000 per trade:
```
55 winners × Rs 8,000 avg profit = Rs 4,40,000 gained
45 losers  × Rs 5,000 avg loss   = Rs 2,25,000 lost
                          Net P&L = Rs 2,15,000 profit
```

This is why **cutting losses short** (stop-loss) and **letting winners run** (staged exits) is the foundation of every good strategy.

---

## Part 3: Trading Basics

### Entry — Getting In

A good entry is when you buy a stock at a price where the odds of it going up are higher than going down. You look for **confirmation signals**:
- The stock is near a support level (floor price that has held before)
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

Why does this matter? Buying near support gives you a natural stop-loss (just below support) and a built-in target (up to resistance). Your risk is small and your reward is large.

### Breakout

A breakout happens when price smashes through resistance and keeps going. The old resistance becomes new support. This is a powerful signal — it means demand is so strong that it overwhelmed all the sellers at that level.

```
                         /
                        /  ← Breakout!
    Resistance → ------/----------
                 /\   /
                /  \ /
    Support  → /----\
              /
```

### Pullback (to EMA)

After a stock has been trending up strongly, it often pauses and drops back a bit before continuing higher. This temporary drop is called a **pullback**.

A common place for pullbacks to stop is at a **moving average** — specifically the **EMA(20)** (20-day Exponential Moving Average). The EMA is a smoothed average of the last 20 days' closing prices, giving more weight to recent prices.

```
                    *
              *    * *     ← Price pulls back
         *   * *  *   *
    *   * * *   **     *   ← Touches EMA(20) = buy opportunity
   * * *                *
    *     ~~~~~~~~~~~~     ← EMA(20) line (smoothed average)
```

Buying on a pullback to the EMA in an uptrend is one of the highest-probability entries in swing trading. You're buying the dip in a strong stock.

### Keltner Channel

The Keltner Channel is a technical indicator that draws an **envelope** around the price. It consists of:

- **Middle line**: EMA(20) — the 20-day average price
- **Upper band**: EMA(20) + 2 x ATR(14) — the "expensive" zone
- **Lower band**: EMA(20) - 2 x ATR(14) — the "cheap" zone

**ATR(14)** is the Average True Range over 14 days — it measures how much the stock typically moves in a day. A stock that moves Rs 30/day has a higher ATR than one that moves Rs 10/day.

```
    ═══════════════════════════════  ← Upper Keltner (expensive)
         *  *     *
        * ** *   * *    *
    ~~~*~~~~~~~~~*~~*~~*~*~~~~~~~~~  ← EMA(20) (middle)
       *        *    **
      *        *
    ═══════════════════════════════  ← Lower Keltner (cheap)
```

When price is at the upper Keltner band, the stock is stretched and may pull back. When it pulls back to the EMA(20) middle line, that's a buying opportunity (Strategy T uses exactly this).

### RSI (Relative Strength Index)

RSI measures how "overbought" or "oversold" a stock is on a scale of 0-100:
- **RSI > 70**: Overbought (may pull back)
- **RSI < 30**: Oversold (may bounce)
- **RSI 30-70**: Neutral zone

RSI is calculated from the ratio of average gains to average losses over a period (typically 14 days).

### RSI Divergence

This is a powerful reversal signal. It occurs when price and RSI disagree:

**Bullish RSI Divergence** — Price makes a **lower low** (new bottom) but RSI makes a **higher low** (less oversold). This means selling pressure is weakening even though price is still falling. A reversal upward is likely.

```
    Price:  100 ──→ 90 ──→ 85     (lower lows — looks weak)
    RSI:     25 ──→ 22 ──→ 28     (higher lows — actually strengthening!)
                              ↑
                    Divergence = buying opportunity
```

The market is telling you: "Yes the price dropped, but the momentum behind the drop is fading." Smart money is quietly accumulating.

### IBS (Internal Bar Strength)

IBS measures where the close falls within the day's range:

```
IBS = (Close - Low) / (High - Low)
```

- IBS close to 1.0 = closed near the day's high (bullish, buyers dominated)
- IBS close to 0.0 = closed near the day's low (bearish, sellers dominated)
- IBS > 0.5 = a bounce sign, used as a confirmation filter

### CCI (Commodity Channel Index)

CCI measures how far the price is from its average. In the JTR system, CCI(20) > -100 is used as a filter — it confirms the stock is not in deep freefall. A CCI below -100 indicates extreme weakness, and buying there is risky.

---

## Part 5: Multi-Stage Exits — The Secret Sauce

Most beginners either:
- Sell too early (take small profit, miss the big move)
- Sell too late (hold too long, give back all gains)

The solution is **multi-stage exits** — selling in portions at different levels.

### Example: 3-Stage Exit

You buy 90 shares at Rs 1,000. Instead of selling all 90 at one target:

```
Stage 1: Price hits +6% (Rs 1,060)
         → Sell 30 shares (1/3), lock in Rs 1,800 profit
         → Tighten stop-loss from 5% to 3%

Stage 2: Price hits +10% (Rs 1,100)
         → Sell 30 shares (1/3), lock in Rs 3,000 profit

Stage 3: Price hits upper Keltner band (Rs 1,150)
         → Sell remaining 30 shares, lock in Rs 4,500 profit
         → Total: Rs 9,300 on a Rs 90,000 position
```

**Why this works:**
- Stage 1 locks in guaranteed profit early (you can't lose on this trade anymore)
- The tighter stop-loss after Stage 1 protects your remaining position
- Stages 2 and 3 let the winners run for maximum gain
- If the stock reverses after Stage 1, you still made money on the first 1/3

Compare with selling everything at +6%: you'd make Rs 5,400 instead of Rs 9,300. Multi-stage exits capture 50-70% more profit on winning trades.

### Tightening the Stop-Loss

After the first partial exit, the stop-loss is brought closer to the entry price. This is called **trailing** or **tightening** the stop:

```
Entry:        Rs 1,000     Stop: Rs 950 (5% away)
After +6%:    Rs 1,060     Stop: Rs 970 (3% below entry)
```

If the stock reverses after hitting +6%, you exit the remaining shares at Rs 970 instead of Rs 950 — reducing your maximum loss on the remaining position. You've already booked profit on the first 1/3, so the overall trade is still a winner even if the rest gets stopped out.

---

## Part 6: The JTR Strategies

The JTR system runs three independent swing trading strategies on Nifty 100 stocks. Each strategy looks for a different setup, but they share the same principles: buy on confirmation, use stop-losses, exit in stages.

---

### Strategy J — Weekly Support Bounce

**What it does:** Buys stocks that have pulled back to their 6-month support level and are showing signs of bouncing.

**The Logic:** If a stock has held a price level for 26 weeks (6 months) and pulls back to that level again, buyers are likely to step in again. You're buying at a price where the market has repeatedly said "this is cheap enough."

**Entry Conditions (ALL must be true):**
| Condition | Why |
|-----------|-----|
| Price within 0-3% above 26-week weekly close support | Stock is near proven support |
| IBS > 0.5 | Close is in upper half of day's range (bounce) |
| Green candle (close > open) | Buyers won the day |
| CCI(20) > -100 | Not in extreme freefall |
| No gap-down (open >= previous close) | No panic selling overnight |

**Stop-Loss:**
- 26-week lowest weekly LOW (this is below the support level, giving the trade room to breathe)
- Chandelier trailing stop after first partial exit: Highest High since entry - 3x ATR(14)
- **Nifty drop shield**: If Nifty 50 index has fallen by the same or more than the stock since entry, the support break stop is suppressed (market-wide selloff, not stock-specific weakness)

**Exit (2-stage scale-out):**
| Stage | Trigger | Action |
|-------|---------|--------|
| 1 | Price reaches entry + 5% | Sell 50% of position |
| 2 | Price reaches entry + 10% | Sell remaining 50% |
| Trail | Chandelier exit (after Stage 1) | Highest high - 3x ATR(14) |
| Safety | 10 trading days underwater | Exit all remaining |

**Example:**
```
HDFCBANK — 26-week support at Rs 1,420
Day's candle: Open 1,415, Low 1,410, High 1,445, Close 1,438
IBS = (1438-1410)/(1445-1410) = 0.80 ✓
Close near support: (1438-1420)/1420 = 1.3% ✓
Green candle ✓, CCI > -100 ✓, No gap-down ✓

→ BUY at Rs 1,438
→ Stop at Rs 1,395 (26-week low)
→ Target 1: Rs 1,510 (+5%) → sell half
→ Target 2: Rs 1,582 (+10%) → sell rest
```

---

### Strategy T — Keltner Channel Pullback

**What it does:** Buys strong stocks that have pulled back from the upper Keltner band to the EMA(20) midline. This is a classic "buy the dip in an uptrend" strategy.

**The Logic:** A stock that recently touched the upper Keltner band is in a strong uptrend. When it pulls back to the EMA(20), the trend is intact but the stock is temporarily "on sale." You're buying a proven winner at a discount.

**Entry Conditions (ALL must be true):**
| Condition | Why |
|-----------|-----|
| Price within 1% of EMA(20) | Stock has pulled back to the midline |
| High touched upper Keltner in last 10 bars | Confirms recent strength |
| Green candle (close > open) | Bounce is starting |
| No gap-down | No overnight panic |

**Stop-Loss:**
- 5% hard stop-loss from entry (initial)
- Tightens to 3% after the first partial exit

**Exit (3-stage scale-out):**
| Stage | Trigger | Action |
|-------|---------|--------|
| 1 | Price reaches entry + 6% | Sell 1/3 of position, tighten SL to 3% |
| 2 | Price reaches entry + 10% | Sell 1/3 of position |
| 3 | Price reaches upper Keltner band | Sell remaining 1/3 |
| Safety | 10 trading days underwater | Exit all remaining |

**Example:**
```
TRENT — Strong uptrend, touched upper Keltner at Rs 5,800 last week
Today pulled back: EMA(20) = Rs 5,450, Close = Rs 5,470
Distance from EMA(20): 0.4% ✓
Was at upper Keltner 6 days ago ✓
Green candle ✓, No gap-down ✓

→ BUY at Rs 5,470
→ Stop at Rs 5,197 (5% hard SL)
→ Target 1: Rs 5,798 (+6%) → sell 1/3, tighten SL to 3%
→ Target 2: Rs 6,017 (+10%) → sell 1/3
→ Target 3: Upper Keltner band → sell remaining 1/3
```

---

### Strategy R — Bullish RSI Divergence

**What it does:** Buys stocks showing bullish RSI divergence — where price is making new lows but momentum (RSI) is actually improving. This catches reversals at the bottom.

**The Logic:** When a stock's price makes a lower low but RSI makes a higher low, it means the selling pressure is exhausting itself. The stock is bottoming out and likely to reverse upward. This is one of the most reliable reversal signals in technical analysis.

**Entry Conditions (ALL must be true):**
| Condition | Why |
|-----------|-----|
| Two swing lows where price: lower low, RSI(14): higher low | Bullish divergence confirmed |
| RSI divergence >= 3 points | Meaningful divergence, not noise |
| RSI(14) < 40 at current swing low | Stock is in oversold zone |
| Structural stop <= 5% from entry | Risk is manageable |
| Green candle (close > open) | Reversal candle |
| No gap-down | No continued panic |

**Stop-Loss:**
- **Structural stop**: 1% below the most recent swing low (this is the "structure" that must hold for the reversal to be valid)
- Tightens to 3% after the first partial exit
- Maximum 5% stop distance (if swing low is too far away, the trade is skipped)

**Exit (3-stage scale-out, same as T):**
| Stage | Trigger | Action |
|-------|---------|--------|
| 1 | Price reaches entry + 6% | Sell 1/3, tighten SL to 3% |
| 2 | Price reaches entry + 10% | Sell 1/3 |
| 3 | Price reaches upper Keltner band | Sell remaining 1/3 |
| Safety | 10 trading days underwater | Exit all remaining |

**Example:**
```
TATAPOWER — Two swing lows:
  Swing Low 1: Rs 376 on Aug 08, RSI = 30.7
  Swing Low 2: Rs 368 on Aug 29, RSI = 34.0

  Price: 376 → 368 (lower low ✓)
  RSI:  30.7 → 34.0 (higher low ✓, divergence = +3.3 pts)
  RSI < 40 ✓

Aug 29 candle is green, no gap-down
Structural stop = 368 × 0.99 = Rs 364.32

→ BUY at Rs 374.15
→ Stop at Rs 364.32 (structural, 2.6% away)
→ Target 1: Rs 396.60 (+6%) → sell 1/3
→ Target 2: Rs 411.57 (+10%) → sell 1/3
→ Target 3: Upper Keltner → sell remaining 1/3
```

---

## Part 7: How the Strategies Work Together

The JTR system runs all three strategies simultaneously on the Nifty 100 universe (top 100 Indian stocks by market cap). This diversification is important:

| Strategy | Market Condition | Style |
|----------|-----------------|-------|
| **J** | Rangebound / support bounce | Mean reversion |
| **T** | Trending / pullback in uptrend | Trend following |
| **R** | Bottoming / reversal | Divergence reversal |

In a trending market, T generates the most signals. In a rangebound market, J shines. When stocks are bottoming after a correction, R catches the reversals. Together, they ensure you always have opportunities regardless of market conditions.

### Capital Allocation

The system allocates a fixed amount per trade (e.g., Rs 2,00,000) regardless of which strategy generated the signal. On any given day, up to 3 new positions can be opened. Signals are ranked by risk (lowest volatility first) so the system prefers calmer stocks over volatile ones.

### Risk Management

Every trade has a predefined maximum loss:
- **J**: Distance to weekly low support (typically 2-5%)
- **T**: 5% hard stop, tightening to 3% after first exit
- **R**: 1% below swing low (typically 2-5%), tightening to 3% after first exit

With Rs 2,00,000 per trade and a 5% max stop, the worst-case loss per trade is Rs 10,000. With a 55% win rate and 1.5x reward/risk ratio, the system generates consistent profits over time.

---

## Part 8: Backtest Results

The JTR system has been backtested over 11 years (2015-2025) on Nifty 100 stocks with Rs 20 lakh capital and Rs 2 lakh per trade:

```
  Year    Tr   Win  Loss    WR%   AvgWin  AvgLoss     PF    Ret%     P&L
  2015   229   104   125  45.4%    7,698   -6,472   0.99   -0.4%  -0.08L
  2016   250   125   125  50.0%    7,773   -4,588   1.69  +19.9%   3.98L
  2017   253   149   104  58.9%    9,557   -4,565   3.00  +47.5%   9.49L
  2018   271   149   122  55.0%    7,733   -5,294   1.78  +25.3%   5.06L
  2019   336   195   141  58.0%    8,136   -5,856   1.92  +38.0%   7.61L
  2020   303   180   123  59.4%    8,353   -7,710   1.59  +27.8%   5.55L
  2021   256   145   111  56.6%    8,999   -5,634   2.09  +34.0%   6.79L
  2022   269   158   111  58.7%    7,669   -6,258   1.74  +25.9%   5.17L
  2023   216   123    93  56.9%    8,039   -4,245   2.50  +29.7%   5.94L
  2024   247   146   101  59.1%    7,944   -5,490   2.09  +30.3%   6.05L
  2025   258   125   133  48.4%    7,699   -3,685   1.96  +23.6%   4.72L

  Avg    263   145   117  55.1%    8,145   -5,436   1.94  +27.4%   5.48L
  Total P&L: Rs 60.3 Lakhs on 20L capital (11 years)
  Winning years: 10/11
```

Key takeaways:
- **10 out of 11 years profitable** (2015 was essentially flat at -0.4%)
- **Average return: +27.4% per year** on effective capital deployed
- **Win rate: 55.1%** — you're wrong 45% of the time, but winners are 1.5x larger than losers
- **Average Profit Factor: 1.94** — for every Rs 1 lost, you make Rs 1.94
- **~263 trades per year** — roughly 1 trade per trading day, very manageable

---

## Glossary

| Term | Definition |
|------|-----------|
| **ATR** | Average True Range — measures daily price volatility |
| **CCI** | Commodity Channel Index — measures distance from average price |
| **Chandelier Exit** | Trailing stop based on highest high minus a multiple of ATR |
| **EMA** | Exponential Moving Average — weighted moving average favoring recent prices |
| **EOD** | End of Day — analysis done after market close |
| **Gap-down** | When today's open is below yesterday's close |
| **Green candle** | Close is above open (buyers won the day) |
| **IBS** | Internal Bar Strength — where close falls within the day's range |
| **Keltner Channel** | EMA(20) with bands at +-2x ATR(14) |
| **PF** | Profit Factor — total wins divided by total losses |
| **RSI** | Relative Strength Index — momentum oscillator (0-100) |
| **SL** | Stop-Loss — predetermined exit price to limit losses |
| **Support** | Price level where buyers have historically stepped in |
| **Swing Low** | A local price bottom confirmed by higher prices on both sides |
| **WR%** | Win Rate — percentage of profitable trades |
