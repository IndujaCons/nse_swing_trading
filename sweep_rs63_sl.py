"""
Sweep #5 — RS63 Hard Stop Loss %
Tests SL values: 5%, 6%, 7%, 8%, 9%, 10%, 12%

Runs Phase 1-3 (data loading + indicators) ONCE, then sweeps Phase 4 only.
"""
import sys
import copy
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, "/Users/jay/Desktop/relative_strength")

import yfinance as yf
from data.momentum_backtest import (
    load_pit_nifty200, get_all_pit_tickers, get_pit_universe,
    TICKER_ALIASES, NIFTY_NEXT50_TICKERS,
    NIFTY_200_NEXT100_TICKERS, _fetch_alias,
)
from data.momentum_engine import NIFTY_50_TICKERS

# ─── Config ───────────────────────────────────────────────────────────────────
CAPITAL = 10_00_000   # 10 lakhs
MAX_POSITIONS = 7
PER_STOCK = CAPITAL // MAX_POSITIONS
PERIOD_DAYS = 365 * 11
PIT_UNIVERSE = True
SL_VALUES = [5, 6, 7, 8, 9, 10, 12]

# ─── Phase 1-3: Load data ONCE ────────────────────────────────────────────────
print("=" * 60)
print("RS63 Hard Stop Loss Sweep")
print("=" * 60)

end_date = datetime.now()
daily_start = end_date - timedelta(days=PERIOD_DAYS + 600)
bt_start_date = (end_date - timedelta(days=PERIOD_DAYS)).date()

# PIT universe
pit_data = None
if PIT_UNIVERSE:
    pit_data = load_pit_nifty200()
    if pit_data is None:
        print("  WARNING: pit_universe=True but nifty200_pit.json not found")

if pit_data is not None:
    all_pit_tickers = get_all_pit_tickers(pit_data)
    tickers = sorted(all_pit_tickers)
    print(f"  PIT universe: {len(tickers)} unique tickers")
else:
    tickers = NIFTY_50_TICKERS + NIFTY_NEXT50_TICKERS + NIFTY_200_NEXT100_TICKERS

# Phase 1: Fetch stock data
print("\n--- Phase 1: Fetching stock data ---")
stock_data = {}
total = len(tickers)
for idx, ticker in enumerate(tickers):
    nse_symbol = f"{ticker}.NS"
    try:
        daily = yf.Ticker(nse_symbol).history(start=daily_start, end=end_date)
    except Exception:
        daily = pd.DataFrame()
    if (daily.empty or len(daily) < 300) and ticker in TICKER_ALIASES:
        daily = _fetch_alias(ticker, daily_start, end_date)
    if not daily.empty and len(daily) >= 300:
        stock_data[ticker] = daily
    if (idx + 1) % 50 == 0:
        print(f"  Loaded {idx + 1}/{total} stocks...")

print(f"  Data loaded: {len(stock_data)} stocks with sufficient history")

# Fetch Nifty 200 benchmark
try:
    bench_data = yf.Ticker("^CNX200").history(start=daily_start, end=end_date)
    print(f"  Nifty 200 benchmark: {len(bench_data)} bars")
except Exception:
    print("  ERROR: Could not fetch Nifty 200 benchmark")
    sys.exit(1)

# Fetch Nifty 50 for beta
nifty50_data = None
try:
    nifty50_data = yf.Ticker("^NSEI").history(start=daily_start, end=end_date)
    print(f"  Nifty 50 loaded: {len(nifty50_data)} bars")
except Exception:
    print("  WARNING: Could not fetch Nifty 50 for beta")

# Phase 2: Pre-compute indicators
print("\n--- Phase 2: Pre-computing indicators ---")
bench_close = bench_data["Close"]


def compute_supertrend(highs, lows, closes, period=10, multiplier=3):
    n = len(closes)
    tr = pd.concat([
        highs - lows,
        (highs - closes.shift(1)).abs(),
        (lows - closes.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    basic_ub = (highs + lows) / 2 + multiplier * atr
    basic_lb = (highs + lows) / 2 - multiplier * atr
    final_ub = np.empty(n)
    final_lb = np.empty(n)
    direction = np.ones(n)
    final_ub[0] = basic_ub.iloc[0]
    final_lb[0] = basic_lb.iloc[0]
    c = closes.values
    bub = basic_ub.values
    blb = basic_lb.values
    for i in range(1, n):
        final_ub[i] = bub[i] if bub[i] < final_ub[i-1] or c[i-1] > final_ub[i-1] else final_ub[i-1]
        final_lb[i] = blb[i] if blb[i] > final_lb[i-1] or c[i-1] < final_lb[i-1] else final_lb[i-1]
        if direction[i-1] == 1:
            direction[i] = -1 if c[i] < final_lb[i] else 1
        else:
            direction[i] = 1 if c[i] > final_ub[i] else -1
    return pd.Series(direction, index=closes.index)


def compute_rsi(closes, period=14):
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


indicators = {}
for ticker, df in stock_data.items():
    closes = df["Close"]
    highs = df["High"]
    lows = df["Low"]
    volumes = df["Volume"]
    bench_aligned = bench_close.reindex(df.index, method="ffill")
    rs_ratio = closes / bench_aligned
    rs123_raw = (rs_ratio / rs_ratio.shift(123) - 1) * 100
    rs123 = rs123_raw.rolling(5, min_periods=1).mean()
    rs55_raw = (rs_ratio / rs_ratio.shift(63) - 1) * 100
    rs55 = rs55_raw.rolling(5, min_periods=1).mean()
    rs21 = (rs_ratio / rs_ratio.shift(21) - 1) * 100
    rs63_positive = (rs55 > 0).astype(int)
    rs63_streak = rs63_positive.copy()
    for k in range(1, len(rs63_streak)):
        if rs63_streak.iloc[k] == 1:
            rs63_streak.iloc[k] = rs63_streak.iloc[k-1] + 1
        else:
            rs63_streak.iloc[k] = 0
    rsi14_raw = compute_rsi(closes, 14)
    rsi14 = rsi14_raw.rolling(3, min_periods=1).mean()
    rsi14_exit = rsi14_raw.rolling(2, min_periods=1).mean()
    ema200 = closes.ewm(span=200, adjust=False).mean()
    ema21 = closes.ewm(span=21, adjust=False).mean()
    st_dir = compute_supertrend(highs, lows, closes, 10, 3)
    high_20d = highs.rolling(20).max().shift(1)
    low_20d = lows.rolling(20).min().shift(1)
    vol_avg20 = volumes.rolling(20).mean()
    beta_series = pd.Series(index=closes.index, dtype=float)
    if nifty50_data is not None:
        n50_aligned = nifty50_data["Close"].reindex(df.index, method="ffill")
        stk_rets = closes.pct_change()
        n50_rets = n50_aligned.pct_change()
        for k in range(252, len(closes)):
            sr = stk_rets.iloc[k-252:k].values
            nr = n50_rets.iloc[k-252:k].values
            mask = ~(np.isnan(sr) | np.isnan(nr))
            if mask.sum() > 100:
                cov = np.cov(sr[mask], nr[mask])
                if cov.shape == (2, 2) and cov[1, 1] > 1e-10:
                    beta_series.iloc[k] = cov[0, 1] / cov[1, 1]
    indicators[ticker] = {
        "rs55": rs55, "rs123": rs123, "rs21": rs21, "rsi14": rsi14,
        "ema200": ema200, "ema21": ema21, "beta": beta_series,
        "supertrend": st_dir,
        "high_20d": high_20d, "low_20d": low_20d,
        "vol_avg20": vol_avg20,
        "rsi14_exit": rsi14_exit,
        "rs63_streak": rs63_streak,
    }

# Phase 3: Build date mapping
print("--- Phase 3: Building date mapping ---")
date_to_iloc = {}
for ticker, df in stock_data.items():
    mapping = {}
    for iloc_idx in range(len(df)):
        dt = df.index[iloc_idx].date()
        mapping[dt] = iloc_idx
    date_to_iloc[ticker] = mapping

all_dates_count = {}
for ticker, df in stock_data.items():
    for d in df.index:
        dt = d.date()
        all_dates_count[dt] = all_dates_count.get(dt, 0) + 1
trading_days = sorted(d for d, c in all_dates_count.items()
                      if c >= 100 and d >= bt_start_date)

print(f"  Trading days: {len(trading_days)} ({trading_days[0]} to {trading_days[-1]})")
print("\nData loading complete. Starting SL sweep...\n")


# ─── Phase 4: Daily loop as a function ────────────────────────────────────────
def run_phase4(sl_pct: float) -> list:
    """Run the daily backtest loop with a given hard stop loss percentage."""
    sl_mult = 1.0 - sl_pct / 100.0   # e.g. 8% SL → 0.92

    portfolio = {}
    trades = []
    cash = float(CAPITAL)

    for day_idx, day in enumerate(trading_days):
        pit_set = get_pit_universe(pit_data, day) if pit_data is not None else None

        # === EXIT PROCESSING ===
        for ticker in list(portfolio.keys()):
            pos = portfolio[ticker]
            idx_map = date_to_iloc.get(ticker, {})
            ci = idx_map.get(day)
            if ci is None:
                continue

            df = stock_data[ticker]
            price = float(df["Close"].iloc[ci])
            ind = indicators[ticker]
            shares = pos["shares"]

            # Exit 1: Hard stop loss
            if price <= pos["entry_price"] * sl_mult:
                pnl = (price - pos["entry_price"]) * shares
                trades.append({
                    "symbol": ticker, "strategy": "RS55",
                    "entry_date": pos["entry_date"], "exit_date": day,
                    "entry_price": pos["entry_price"], "exit_price": price,
                    "shares": shares, "pnl": pnl,
                    "exit_reason": f"HARD_SL_{sl_pct}PCT",
                    "hold_days": (day - pos["entry_date"]).days,
                })
                cash += shares * price
                del portfolio[ticker]
                continue

            # Exit 2: RSI(14, 3d avg) < 40 for 3 consecutive days
            rsi_v = float(ind["rsi14"].iloc[ci]) if ci < len(ind["rsi14"]) and not pd.isna(ind["rsi14"].iloc[ci]) else 50
            if rsi_v < 40:
                pos["rsi_low_count"] = pos.get("rsi_low_count", 0) + 1
            else:
                pos["rsi_low_count"] = 0
            if pos.get("rsi_low_count", 0) >= 3:
                pnl = (price - pos["entry_price"]) * shares
                trades.append({
                    "symbol": ticker, "strategy": "RS55",
                    "entry_date": pos["entry_date"], "exit_date": day,
                    "entry_price": pos["entry_price"], "exit_price": price,
                    "shares": shares, "pnl": pnl,
                    "exit_reason": "RSI_BELOW40_3D",
                    "hold_days": (day - pos["entry_date"]).days,
                })
                cash += shares * price
                del portfolio[ticker]
                continue

            # Exit 3: RS55 (5d avg) < 0
            rs55_v = float(ind["rs55"].iloc[ci]) if ci < len(ind["rs55"]) and not pd.isna(ind["rs55"].iloc[ci]) else 0
            if rs55_v < 0:
                pnl = (price - pos["entry_price"]) * shares
                trades.append({
                    "symbol": ticker, "strategy": "RS55",
                    "entry_date": pos["entry_date"], "exit_date": day,
                    "entry_price": pos["entry_price"], "exit_price": price,
                    "shares": shares, "pnl": pnl,
                    "exit_reason": "RS63_NEG",
                    "hold_days": (day - pos["entry_date"]).days,
                })
                cash += shares * price
                del portfolio[ticker]
                continue

            # Exit 4: Time stop — 8 weeks + <3% gain
            days_held = (day - pos["entry_date"]).days
            gain_pct = (price - pos["entry_price"]) / pos["entry_price"] * 100
            if days_held >= 56 and gain_pct < 3.0:
                pnl = (price - pos["entry_price"]) * shares
                trades.append({
                    "symbol": ticker, "strategy": "RS55",
                    "entry_date": pos["entry_date"], "exit_date": day,
                    "entry_price": pos["entry_price"], "exit_price": price,
                    "shares": shares, "pnl": pnl,
                    "exit_reason": "TIME_STOP",
                    "hold_days": days_held,
                })
                cash += shares * price
                del portfolio[ticker]

        # === ENTRY SIGNALS ===
        if len(portfolio) >= MAX_POSITIONS:
            continue

        signals = []
        for ticker, df in stock_data.items():
            if ticker in portfolio:
                continue
            if pit_set is not None and ticker not in pit_set:
                continue

            idx_map = date_to_iloc.get(ticker, {})
            ci = idx_map.get(day)
            if ci is None or ci < 210:
                continue

            ind = indicators[ticker]
            price = float(df["Close"].iloc[ci])
            open_price = float(df["Open"].iloc[ci])
            volume = float(df["Volume"].iloc[ci])

            rs55_v = float(ind["rs55"].iloc[ci]) if not pd.isna(ind["rs55"].iloc[ci]) else -1
            rsi_v = float(ind["rsi14"].iloc[ci]) if not pd.isna(ind["rsi14"].iloc[ci]) else 0
            l20d = float(ind["low_20d"].iloc[ci]) if not pd.isna(ind["low_20d"].iloc[ci]) else 0

            if rs55_v <= 0 or rsi_v <= 50:
                continue

            high_today = float(df["High"].iloc[ci])
            low_today = float(df["Low"].iloc[ci])
            day_range = high_today - low_today
            ibs = (price - low_today) / day_range if day_range > 0 else 0.5
            if ibs <= 0.5:
                continue
            if price <= open_price:
                continue

            stop_pct = (price - l20d) / price * 100 if price > 0 and l20d > 0 else 8

            signals.append({
                "ticker": ticker,
                "price": price,
                "swing_low": l20d,
                "stop_pct": stop_pct,
                "rs55": rs55_v,
            })

        signals.sort(key=lambda s: s["stop_pct"])
        for sig in signals:
            if len(portfolio) >= MAX_POSITIONS:
                break
            ticker = sig["ticker"]
            price = sig["price"]
            shares = int(PER_STOCK // price)
            if shares <= 0:
                continue
            cost = shares * price
            if cost > cash:
                continue

            portfolio[ticker] = {
                "entry_date": day,
                "entry_price": price,
                "shares": shares,
                "rs21_neg_count": 0,
                "rsi_low_count": 0,
            }
            cash -= cost

    # Close remaining at end
    last_day = trading_days[-1]
    for ticker in list(portfolio.keys()):
        pos = portfolio[ticker]
        idx_map = date_to_iloc.get(ticker, {})
        ci = idx_map.get(last_day)
        if ci is not None:
            exit_price = float(stock_data[ticker]["Close"].iloc[ci])
        else:
            exit_price = pos["entry_price"]
        shares = pos["shares"]
        pnl = (exit_price - pos["entry_price"]) * shares
        trades.append({
            "symbol": ticker, "strategy": "RS55",
            "entry_date": pos["entry_date"], "exit_date": last_day,
            "entry_price": pos["entry_price"], "exit_price": exit_price,
            "shares": shares, "pnl": pnl,
            "exit_reason": "BACKTEST_END",
            "hold_days": (last_day - pos["entry_date"]).days,
        })

    return trades


# ─── Metrics calculation ───────────────────────────────────────────────────────
def compute_charges(entry_price, exit_price, shares):
    """Zerodha equity delivery charges."""
    B = entry_price * shares
    S = exit_price * shares
    total_turnover = B + S
    stt = 0.001 * total_turnover
    exchange = 0.0000307 * total_turnover
    sebi = 0.000001 * total_turnover
    stamp = 0.00015 * B
    gst = 0.18 * (exchange + sebi)
    total_charges = stt + exchange + sebi + stamp + gst
    # STT not deductible
    deductible = exchange + sebi + stamp + gst
    return total_charges, deductible


def compute_metrics(trades, capital=CAPITAL):
    """Compute WR, PF, net %/yr, neg years, year-by-year returns."""
    if not trades:
        return {}

    rows = []
    for t in trades:
        gross_pnl = t["pnl"]
        total_charges, deductible = compute_charges(
            t["entry_price"], t["exit_price"], t["shares"]
        )
        taxable_profit = gross_pnl - deductible
        stcg_tax = 0.20 * taxable_profit if taxable_profit > 0 else 0
        net_profit = gross_pnl - total_charges - stcg_tax
        year = t["exit_date"].year if hasattr(t["exit_date"], "year") else int(str(t["exit_date"])[:4])
        rows.append({
            "year": year,
            "gross_pnl": gross_pnl,
            "net_profit": net_profit,
            "win": 1 if gross_pnl > 0 else 0,
        })

    df = pd.DataFrame(rows)

    wr = df["win"].mean() * 100
    gross_win = df[df["gross_pnl"] > 0]["gross_pnl"].sum()
    gross_loss = df[df["gross_pnl"] < 0]["gross_pnl"].abs().sum()
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    yearly = df.groupby("year")["net_profit"].sum()
    yearly_pct = (yearly / capital * 100).round(1)
    neg_years = (yearly_pct < 0).sum()

    # Average net %/yr (simple mean of yearly returns)
    net_pct_yr = yearly_pct.mean()

    return {
        "trades": len(df),
        "wr": round(wr, 1),
        "pf": round(pf, 2),
        "net_pct_yr": round(net_pct_yr, 1),
        "neg_years": int(neg_years),
        "yearly": yearly_pct.to_dict(),
    }


# ─── Run sweep ────────────────────────────────────────────────────────────────
results = {}
for sl_pct in SL_VALUES:
    print(f"Running SL={sl_pct}%...", flush=True)
    trades = run_phase4(sl_pct)
    metrics = compute_metrics(trades)
    results[sl_pct] = metrics
    print(f"  SL={sl_pct}%: {metrics['trades']} trades, WR={metrics['wr']}%, "
          f"PF={metrics['pf']}, Net%/yr={metrics['net_pct_yr']}, NegYrs={metrics['neg_years']}")

# ─── Results table ────────────────────────────────────────────────────────────
all_years = sorted(set(
    y for m in results.values() for y in m.get("yearly", {}).keys()
))

print("\n" + "=" * 80)
print("RS63 HARD STOP LOSS SWEEP RESULTS")
print("=" * 80)

# Header
header = f"{'SL%':>5} | {'Trades':>6} | {'WR%':>5} | {'PF':>5} | {'Net%/yr':>7} | {'NegYrs':>6} | " + " | ".join(f"{y}" for y in all_years)
print(header)
print("-" * len(header))

for sl_pct in SL_VALUES:
    m = results[sl_pct]
    year_cols = " | ".join(f"{m['yearly'].get(y, 0.0):>6.1f}" for y in all_years)
    marker = " <-- FROZEN" if sl_pct == 8 else ""
    print(f"{sl_pct:>5} | {m['trades']:>6} | {m['wr']:>5.1f} | {m['pf']:>5.2f} | {m['net_pct_yr']:>7.1f} | {m['neg_years']:>6} | {year_cols}{marker}")

print("=" * 80)
print(f"\nCapital: ₹{CAPITAL:,} | Max positions: {MAX_POSITIONS} | PIT: {PIT_UNIVERSE}")
print(f"Backtest period: {trading_days[0]} to {trading_days[-1]}")
