"""
RS63 Kalman filter experiment — compares baseline RS63 vs Kalman slope
as entry/exit signal on the Vivek Bajaj RS63 strategy.

Baseline  : rs55 (5d smoothed RS63) > 0 for entry, < 0 for exit
Kalman    : kalman_slope > threshold for entry, < 0 for exit

Run: python3 data/rs63_kalman_test.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from data.etf_core_backtest import kalman_rs_slope
from data.momentum_backtest import (
    load_pit_nifty200, get_all_pit_tickers, get_pit_universe, TICKER_ALIASES
)

# ── Config ────────────────────────────────────────────────────────────────────
PERIOD_DAYS   = 365 * 11
CAPITAL       = 10 * 100_000
PER_STOCK     = CAPITAL // 7
MAX_POSITIONS = 7
END_DATE      = datetime.now()
DAILY_START   = END_DATE - timedelta(days=PERIOD_DAYS + 600)
BT_START      = (END_DATE - timedelta(days=PERIOD_DAYS)).date()

# Kalman params to test (q1, q2, threshold)
KALMAN_CONFIGS = [
    (0.001, 0.0001, 0.0),
    (0.01,  0.001,  0.0),
    (0.01,  0.001,  0.00005),
    (0.01,  0.001,  0.0001),
    (0.05,  0.0001, 0.0),
    (0.1,   0.0001, 0.0),
]

# ── Data fetch ────────────────────────────────────────────────────────────────

def fetch_data():
    pit_data = load_pit_nifty200()
    tickers = sorted(get_all_pit_tickers(pit_data)) if pit_data else []
    print(f"PIT universe: {len(tickers)} tickers")

    print("Fetching benchmark (^CNX200)...", end=" ", flush=True)
    bench_raw = yf.Ticker("^CNX200").history(start=DAILY_START, end=END_DATE)
    bench_close = bench_raw["Close"]
    print(f"{len(bench_close)} bars")

    stock_data = {}
    for i, ticker in enumerate(tickers):
        try:
            df = yf.Ticker(f"{ticker}.NS").history(start=DAILY_START, end=END_DATE)
        except Exception:
            df = pd.DataFrame()
        if (df.empty or len(df) < 300) and ticker in TICKER_ALIASES:
            alias = TICKER_ALIASES[ticker]
            try:
                df2 = yf.Ticker(f"{alias}.NS").history(start=DAILY_START, end=END_DATE)
                if len(df2) >= 300:
                    df = df2
            except Exception:
                pass
        if not df.empty and len(df) >= 300:
            stock_data[ticker] = df
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(tickers)} loaded...", flush=True)

    print(f"Loaded {len(stock_data)} stocks")
    return stock_data, bench_close, pit_data

# ── Indicators ────────────────────────────────────────────────────────────────

def compute_rsi(closes, period=14):
    delta = closes.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))

def build_indicators(stock_data, bench_close, kalman_configs):
    """Build indicator dicts including Kalman slopes for each config."""
    print("Computing indicators...", flush=True)
    inds = {}  # ticker -> dict of series

    for ticker, df in stock_data.items():
        closes  = df["Close"]
        opens   = df["Open"]
        highs   = df["High"]
        lows    = df["Low"]
        volumes = df["Volume"]

        bench_aligned = bench_close.reindex(df.index, method="ffill")

        # RS63 (rs55 in original code — 5d smoothed 63d relative return)
        rs_ratio = closes / bench_aligned
        rs63_raw = (rs_ratio / rs_ratio.shift(63) - 1) * 100
        rs63     = rs63_raw.rolling(5, min_periods=1).mean()

        # RSI(14) — 3d avg
        rsi14_raw = compute_rsi(closes, 14)
        rsi14     = rsi14_raw.rolling(3, min_periods=1).mean()

        # Price reference
        high_20d = highs.rolling(20).max().shift(1)
        low_20d  = lows.rolling(20).min().shift(1)
        vol_avg  = volumes.rolling(20).mean()

        entry = {
            "closes": closes, "opens": opens,
            "highs": highs, "lows": lows,
            "rs63": rs63,
            "rsi14": rsi14,
            "high_20d": high_20d, "low_20d": low_20d,
            "vol_avg": vol_avg,
        }

        # Kalman slopes for each config
        log_rs = np.log(closes.values / bench_aligned.values)
        valid  = ~np.isnan(log_rs)
        for (q1, q2, _) in kalman_configs:
            key = f"kal_{q1}_{q2}"
            if key not in entry:
                if valid.sum() > 10:
                    ks = kalman_rs_slope(log_rs[valid], q1=q1, q2=q2, r=1.0)
                    full = np.full(len(log_rs), np.nan)
                    full[valid] = ks
                    entry[key] = pd.Series(full, index=closes.index)
                else:
                    entry[key] = pd.Series(np.nan, index=closes.index)

        inds[ticker] = entry

    print(f"Indicators done for {len(inds)} stocks")
    return inds

# ── Backtest engine ───────────────────────────────────────────────────────────

def run_one(stock_data, inds, bench_close, pit_data,
            use_kalman=False, q1=0.01, q2=0.001, kalman_threshold=0.0,
            label="Baseline"):

    kal_key = f"kal_{q1}_{q2}" if use_kalman else None

    # Date index
    date_to_iloc = {}
    for ticker, df in stock_data.items():
        date_to_iloc[ticker] = {df.index[i].date(): i for i in range(len(df))}

    all_dates_count = {}
    for ticker, df in stock_data.items():
        for d in df.index:
            dt = d.date()
            all_dates_count[dt] = all_dates_count.get(dt, 0) + 1
    trading_days = sorted(d for d, c in all_dates_count.items()
                          if c >= 100 and d >= BT_START)

    portfolio = {}
    trades    = []
    cash      = float(CAPITAL)

    for day_idx, day in enumerate(trading_days):
        pit_set = get_pit_universe(pit_data, day) if pit_data else None

        # ── EXITS ──
        for ticker in list(portfolio.keys()):
            pos = portfolio[ticker]
            ci  = date_to_iloc.get(ticker, {}).get(day)
            if ci is None:
                continue
            ind   = inds[ticker]
            price = float(ind["closes"].iloc[ci])
            shares = pos["shares"]

            # Hard SL 8%
            if price <= pos["entry_price"] * 0.92:
                pnl = (price - pos["entry_price"]) * shares
                trades.append({"day": day, "pnl": pnl, "reason": "SL"})
                cash += shares * price
                del portfolio[ticker]
                continue

            # RSI < 40 for 3 days
            rsi_v = float(ind["rsi14"].iloc[ci]) if not pd.isna(ind["rsi14"].iloc[ci]) else 50
            pos["rsi_low"] = (pos.get("rsi_low", 0) + 1) if rsi_v < 40 else 0
            if pos["rsi_low"] >= 3:
                pnl = (price - pos["entry_price"]) * shares
                trades.append({"day": day, "pnl": pnl, "reason": "RSI40"})
                cash += shares * price
                del portfolio[ticker]
                continue

            # RS signal < 0 (RS63 or Kalman)
            if use_kalman:
                sig_s = ind.get(kal_key)
                sig_v = float(sig_s.iloc[ci]) if sig_s is not None and not pd.isna(sig_s.iloc[ci]) else 0
            else:
                sig_v = float(ind["rs63"].iloc[ci]) if not pd.isna(ind["rs63"].iloc[ci]) else 0
            if sig_v < 0:
                pnl = (price - pos["entry_price"]) * shares
                trades.append({"day": day, "pnl": pnl, "reason": "RS_NEG"})
                cash += shares * price
                del portfolio[ticker]
                continue

            # Time stop: 8 weeks + <3% gain
            days_held = (day - pos["entry_date"]).days
            gain_pct  = (price - pos["entry_price"]) / pos["entry_price"] * 100
            if days_held >= 56 and gain_pct < 3.0:
                pnl = (price - pos["entry_price"]) * shares
                trades.append({"day": day, "pnl": pnl, "reason": "TIME"})
                cash += shares * price
                del portfolio[ticker]

        # ── ENTRIES ──
        if len(portfolio) >= MAX_POSITIONS:
            continue

        signals = []
        for ticker, df in stock_data.items():
            if ticker in portfolio:
                continue
            if pit_set is not None and ticker not in pit_set:
                continue
            ci = date_to_iloc.get(ticker, {}).get(day)
            if ci is None or ci < 210:
                continue

            ind   = inds[ticker]
            price = float(ind["closes"].iloc[ci])
            open_ = float(ind["opens"].iloc[ci])
            high  = float(ind["highs"].iloc[ci])
            low   = float(ind["lows"].iloc[ci])

            # RSI > 50
            rsi_v = float(ind["rsi14"].iloc[ci]) if not pd.isna(ind["rsi14"].iloc[ci]) else 0
            if rsi_v <= 50:
                continue

            # RS signal > threshold
            if use_kalman:
                sig_s = ind.get(kal_key)
                sig_v = float(sig_s.iloc[ci]) if sig_s is not None and not pd.isna(sig_s.iloc[ci]) else -1
            else:
                sig_v = float(ind["rs63"].iloc[ci]) if not pd.isna(ind["rs63"].iloc[ci]) else -1
            if sig_v <= kalman_threshold:
                continue

            # IBS > 0.5 + green candle
            day_range = high - low
            ibs = (price - low) / day_range if day_range > 0 else 0.5
            if ibs <= 0.5 or price <= open_:
                continue

            l20 = float(ind["low_20d"].iloc[ci]) if not pd.isna(ind["low_20d"].iloc[ci]) else 0
            stop_pct = (price - l20) / price * 100 if price > 0 and l20 > 0 else 8
            signals.append({"ticker": ticker, "price": price, "stop_pct": stop_pct})

        signals.sort(key=lambda s: s["stop_pct"])
        for sig in signals:
            if len(portfolio) >= MAX_POSITIONS:
                break
            price  = sig["price"]
            shares = int(PER_STOCK // price)
            if shares <= 0 or shares * price > cash:
                continue
            portfolio[sig["ticker"]] = {
                "entry_date": day, "entry_price": price,
                "shares": shares, "rsi_low": 0,
            }
            cash -= shares * price

    # Close open positions at last price
    last_day = trading_days[-1]
    for ticker, pos in portfolio.items():
        ci = date_to_iloc.get(ticker, {}).get(last_day)
        exit_price = float(inds[ticker]["closes"].iloc[ci]) if ci else pos["entry_price"]
        pnl = (exit_price - pos["entry_price"]) * pos["shares"]
        trades.append({"day": last_day, "pnl": pnl, "reason": "OPEN"})
        cash += pos["shares"] * exit_price

    return trades, trading_days

# ── Summary helpers ───────────────────────────────────────────────────────────

def yearly_stats(trades, trading_days):
    """Compute per-year net P&L as % of initial capital."""
    year_pnl = {}
    for t in trades:
        yr = t["day"].year
        year_pnl[yr] = year_pnl.get(yr, 0) + t["pnl"]
    return year_pnl

def summary(trades):
    closed = [t for t in trades if t["reason"] != "OPEN"]
    wins   = [t for t in closed if t["pnl"] > 0]
    total  = sum(t["pnl"] for t in trades)
    wr     = len(wins) / len(closed) * 100 if closed else 0
    net_pct = total / CAPITAL * 100
    return len(closed), wr, net_pct

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    stock_data, bench_close, pit_data = fetch_data()
    inds = build_indicators(stock_data, bench_close, KALMAN_CONFIGS)

    all_year_data = {}

    # Baseline
    print("\nRunning RS63 BASELINE...", flush=True)
    trades_b, tdays = run_one(stock_data, inds, bench_close, pit_data,
                               use_kalman=False, label="RS63 Baseline")
    n_b, wr_b, net_b = summary(trades_b)
    all_year_data["RS63"] = yearly_stats(trades_b, tdays)

    # Kalman configs
    kalman_results = {}
    for q1, q2, thr in KALMAN_CONFIGS:
        lbl = f"Kal q1={q1} q2={q2} thr={thr}"
        print(f"Running {lbl}...", flush=True)
        trades_k, _ = run_one(stock_data, inds, bench_close, pit_data,
                               use_kalman=True, q1=q1, q2=q2,
                               kalman_threshold=thr, label=lbl)
        n_k, wr_k, net_k = summary(trades_k)
        kalman_results[lbl] = (n_k, wr_k, net_k)
        all_year_data[lbl] = yearly_stats(trades_k, tdays)

    # ── Print per-year comparison ──
    all_years = sorted({yr for yd in all_year_data.values() for yr in yd})
    configs   = ["RS63"] + [f"Kal q1={q1} q2={q2} thr={thr}" for q1, q2, thr in KALMAN_CONFIGS]

    # Header
    hdr = f"  {'Year':6s}  {'RS63':>8s}"
    for q1, q2, thr in KALMAN_CONFIGS:
        hdr += f"  {'K'+str(q2):>8s}"
    print("\n" + "="*100)
    print("  RS63 STRATEGY — KALMAN COMPARISON (per year, % return on 10L capital)")
    print("="*100)
    print(hdr)
    sep = f"  {'─'*6}  {'─'*8}" + "  " + "  ".join(["─"*8]*len(KALMAN_CONFIGS))
    print(sep)

    for yr in all_years:
        b_pnl = all_year_data["RS63"].get(yr, 0)
        b_pct = b_pnl / CAPITAL * 100
        row = f"  {yr}  {b_pct:>+7.1f}%"
        for q1, q2, thr in KALMAN_CONFIGS:
            lbl  = f"Kal q1={q1} q2={q2} thr={thr}"
            k_pnl = all_year_data[lbl].get(yr, 0)
            k_pct = k_pnl / CAPITAL * 100
            delta = k_pct - b_pct
            row += f"  {k_pct:>+7.1f}%"
        print(row)

    print(sep)
    # Totals row
    tot_b = sum(all_year_data["RS63"].values()) / CAPITAL * 100
    row = f"  {'Total':6s}  {tot_b:>+7.1f}%"
    for q1, q2, thr in KALMAN_CONFIGS:
        lbl = f"Kal q1={q1} q2={q2} thr={thr}"
        tot_k = sum(all_year_data[lbl].values()) / CAPITAL * 100
        row += f"  {tot_k:>+7.1f}%"
    print(row)

    # Trades/WR row
    row_t = f"  {'Trades':6s}  {n_b:>8d}"
    row_w = f"  {'WR':6s}  {wr_b:>7.1f}%"
    for lbl, (n_k, wr_k, net_k) in kalman_results.items():
        row_t += f"  {n_k:>8d}"
        row_w += f"  {wr_k:>7.1f}%"
    print(sep)
    print(row_t)
    print(row_w)

    print(f"\n  Config labels:")
    print(f"  RS63 = smoothed 63d relative return > 0")
    for i, (q1, q2, thr) in enumerate(KALMAN_CONFIGS):
        print(f"  K{q2} (col {i+1}) = Kalman q1={q1} q2={q2} threshold={thr}")
