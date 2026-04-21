"""
RS63 Entry Threshold Sweep
Tests rs63_min_threshold in [0, 2, 5, 8, 10] (percent).

Data is loaded once; only the Phase 4 daily loop is re-run per threshold.
"""

import sys
import os
import copy
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

# Add project root to path
sys.path.insert(0, "/Users/jay/Desktop/relative_strength")

from data.momentum_backtest import (
    MomentumBacktester,
    load_pit_nifty200,
    get_all_pit_tickers,
    get_pit_universe,
    TICKER_ALIASES,
    NIFTY_NEXT50_TICKERS,
    NIFTY_200_NEXT100_TICKERS,
    _fetch_alias,
)

CAPITAL = 10_00_000  # 10 lakhs
MAX_POSITIONS = 7
PER_STOCK = CAPITAL // MAX_POSITIONS
PERIOD_DAYS = 365 * 11

# ─────────────────────────────────────────────────────────────────────────────
# CHARGES (Zerodha equity delivery)
# ─────────────────────────────────────────────────────────────────────────────

def compute_net_profit(gross_pnl, entry_price, exit_price, shares):
    B = entry_price * shares
    S = exit_price * shares
    total_turnover = B + S
    stt      = 0.001     * total_turnover
    exchange = 0.0000307 * total_turnover
    sebi     = 0.000001  * total_turnover
    stamp    = 0.00015   * B
    gst      = 0.18      * (exchange + sebi)
    total_charges = stt + exchange + sebi + stamp + gst

    # STT not deductible; rest are
    deductible = exchange + sebi + stamp + gst
    taxable_profit = gross_pnl - deductible
    stcg_tax = 0.20 * taxable_profit if taxable_profit > 0 else 0.0
    net = gross_pnl - total_charges - stcg_tax
    return net, total_charges

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(trades, label):
    if not trades:
        print(f"\n{label}: No trades")
        return

    df = pd.DataFrame(trades)
    # Attach net_profit per trade
    nets = []
    for _, row in df.iterrows():
        net, _ = compute_net_profit(row["pnl"], row["entry_price"], row["exit_price"], row["shares"])
        nets.append(net)
    df["net_profit"] = nets

    # Overall stats
    n = len(df)
    wr = (df["pnl"] > 0).mean() * 100
    gross_win = df[df["pnl"] > 0]["pnl"].sum()
    gross_loss = df[df["pnl"] < 0]["pnl"].abs().sum()
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    # Year-by-year net returns
    df["exit_year"] = pd.to_datetime(df["exit_date"].astype(str)).dt.year
    yearly = df.groupby("exit_year")["net_profit"].sum()
    yearly_pct = (yearly / CAPITAL * 100).round(1)

    net_per_yr = yearly_pct.mean()
    neg_years = (yearly_pct < 0).sum()

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Trades: {n}  |  WR: {wr:.1f}%  |  PF: {pf:.2f}")
    print(f"  Avg Net %/yr: {net_per_yr:.1f}%  |  Neg years: {neg_years}")
    print(f"\n  Year-by-year net returns:")
    for yr, pct in yearly_pct.items():
        marker = " <--" if pct < 0 else ""
        print(f"    {yr}: {pct:+.1f}%{marker}")

    return {
        "label": label,
        "trades": n,
        "wr": round(wr, 1),
        "pf": round(pf, 2),
        "net_per_yr": round(net_per_yr, 1),
        "neg_years": int(neg_years),
        "yearly": yearly_pct.to_dict(),
    }

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4: DAILY LOOP (parameterised by rs63_min_threshold)
# ─────────────────────────────────────────────────────────────────────────────

def run_loop(stock_data, indicators, date_to_iloc, trading_days, pit_data,
             rs63_min_threshold=0.0):
    """Re-run the Phase 4 loop with a different RS63 entry threshold."""

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

            # Exit 1: 8% hard stop
            if price <= pos["entry_price"] * 0.92:
                pnl = (price - pos["entry_price"]) * shares
                trades.append({
                    "symbol": ticker, "strategy": "RS55",
                    "entry_date": pos["entry_date"], "exit_date": day,
                    "entry_price": pos["entry_price"], "exit_price": price,
                    "shares": shares, "pnl": pnl,
                    "exit_reason": "HARD_SL_8PCT",
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

            # Exit 3: RS55 (5d avg) < 0 — stock lost relative strength
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

            # Get all indicator values
            rs55_v = float(ind["rs55"].iloc[ci]) if not pd.isna(ind["rs55"].iloc[ci]) else -1
            rsi_v  = float(ind["rsi14"].iloc[ci]) if not pd.isna(ind["rsi14"].iloc[ci]) else 0
            l20d   = float(ind["low_20d"].iloc[ci]) if not pd.isna(ind["low_20d"].iloc[ci]) else 0

            # SWEEP PARAMETER: RS63 entry threshold
            if rs55_v <= rs63_min_threshold or rsi_v <= 50:
                continue

            # IBS > 0.5 (close in upper half of day's range) + green candle
            high_today = float(df["High"].iloc[ci])
            low_today  = float(df["Low"].iloc[ci])
            day_range  = high_today - low_today
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

        # Rank by tightest stop (lowest risk), allocate
        signals.sort(key=lambda s: s["stop_pct"])
        for sig in signals:
            if len(portfolio) >= MAX_POSITIONS:
                break
            ticker = sig["ticker"]
            price  = sig["price"]
            shares = int(PER_STOCK // price)
            if shares <= 0:
                continue
            cost = shares * price
            if cost > cash:
                continue

            portfolio[ticker] = {
                "entry_date":    day,
                "entry_price":   price,
                "shares":        shares,
                "rs21_neg_count": 0,
                "rsi_low_count":  0,
            }
            cash -= cost

        if (day_idx + 1) % 500 == 0:
            print(f"    [threshold={rs63_min_threshold}] Day {day_idx+1}/{len(trading_days)}, "
                  f"positions: {len(portfolio)}, trades: {len(trades)}")

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

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    end_date   = datetime.now()
    daily_start = end_date - timedelta(days=PERIOD_DAYS + 600)
    bt_start_date = (end_date - timedelta(days=PERIOD_DAYS)).date()

    # ── Phase 1: PIT universe ──────────────────────────────────────────────
    pit_data = load_pit_nifty200()
    if pit_data is None:
        print("ERROR: nifty200_pit.json not found")
        sys.exit(1)

    all_pit_tickers = get_all_pit_tickers(pit_data)
    tickers = sorted(all_pit_tickers)
    print(f"PIT universe: {len(tickers)} unique tickers")

    # ── Phase 1: Fetch stock data ──────────────────────────────────────────
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
            print(f"  Loaded {idx+1}/{total} stocks...")

    print(f"Data loaded: {len(stock_data)} stocks with sufficient history")

    # ── Benchmark ─────────────────────────────────────────────────────────
    bench_data = yf.Ticker("^CNX200").history(start=daily_start, end=end_date)
    print(f"Nifty 200 benchmark: {len(bench_data)} bars")
    bench_close = bench_data["Close"]

    # Nifty 50 for beta
    try:
        nifty50_data = yf.Ticker("^NSEI").history(start=daily_start, end=end_date)
        print(f"Nifty 50 loaded: {len(nifty50_data)} bars")
    except Exception:
        nifty50_data = None

    # ── Phase 2: Pre-compute indicators ───────────────────────────────────
    print("Pre-computing indicators...")

    def compute_rsi(closes, period=14):
        delta = closes.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        return 100 - (100 / (1 + rs))

    indicators = {}
    for ticker, df in stock_data.items():
        closes  = df["Close"]
        highs   = df["High"]
        lows    = df["Low"]
        volumes = df["Volume"]

        bench_aligned = bench_close.reindex(df.index, method="ffill")

        rs_ratio  = closes / bench_aligned
        rs55_raw  = (rs_ratio / rs_ratio.shift(63) - 1) * 100
        rs55      = rs55_raw.rolling(5, min_periods=1).mean()
        rs123_raw = (rs_ratio / rs_ratio.shift(123) - 1) * 100
        rs123     = rs123_raw.rolling(5, min_periods=1).mean()
        rs21      = (rs_ratio / rs_ratio.shift(21) - 1) * 100

        rsi14_raw  = compute_rsi(closes, 14)
        rsi14      = rsi14_raw.rolling(3, min_periods=1).mean()
        rsi14_exit = rsi14_raw.rolling(2, min_periods=1).mean()

        ema200 = closes.ewm(span=200, adjust=False).mean()
        ema21  = closes.ewm(span=21,  adjust=False).mean()

        high_20d  = highs.rolling(20).max().shift(1)
        low_20d   = lows.rolling(20).min().shift(1)
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
            "rs55": rs55, "rs123": rs123, "rs21": rs21,
            "rsi14": rsi14, "rsi14_exit": rsi14_exit,
            "ema200": ema200, "ema21": ema21,
            "beta": beta_series,
            "high_20d": high_20d, "low_20d": low_20d,
            "vol_avg20": vol_avg20,
        }

    print("Indicators computed.")

    # ── Phase 3: Build date mapping ────────────────────────────────────────
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
    print(f"Trading days: {len(trading_days)} ({trading_days[0]} to {trading_days[-1]})")

    # ── Phase 4: Sweep ────────────────────────────────────────────────────
    thresholds = [0, 2, 5, 8, 10]
    results = []

    for thresh in thresholds:
        print(f"\n--- Running threshold = {thresh}% ---")
        trades = run_loop(
            stock_data, indicators, date_to_iloc, trading_days, pit_data,
            rs63_min_threshold=float(thresh),
        )
        print(f"  Completed: {len(trades)} trades")
        r = compute_metrics(trades, label=f"RS63 entry threshold = {thresh}%")
        if r:
            results.append(r)

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Threshold':>12} {'Trades':>8} {'WR%':>7} {'PF':>6} {'Net%/yr':>9} {'NegYrs':>8}")
    print("-"*70)
    for r in results:
        print(f"{r['label'].split('=')[1].strip():>12} {r['trades']:>8} {r['wr']:>7.1f} "
              f"{r['pf']:>6.2f} {r['net_per_yr']:>9.1f} {r['neg_years']:>8}")

    # Year-by-year comparison
    print("\n\nYEAR-BY-YEAR COMPARISON (net %/yr)")
    all_years = sorted(set(yr for r in results for yr in r["yearly"]))
    header = f"{'Year':>6}" + "".join(f"{'thr='+str(r['label'].split('=')[1].strip()):>12}" for r in results)
    print(header)
    print("-" * (6 + 12 * len(results)))
    for yr in all_years:
        row = f"{yr:>6}"
        for r in results:
            val = r["yearly"].get(yr, float("nan"))
            row += f"{val:>+12.1f}" if not np.isnan(val) else f"{'N/A':>12}"
        print(row)


if __name__ == "__main__":
    main()
