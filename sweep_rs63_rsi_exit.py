"""
Sweep #6 — RS63 RSI Exit Threshold × Consecutive Days Grid
3×3: RSI threshold (35, 40, 45) × consecutive days (2, 3, 4)
"""

import sys

# Set up path so imports work correctly
sys.path.insert(0, "/Users/jay/Desktop/relative_strength")

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Import helpers from the project
from data.momentum_backtest import (
    load_pit_nifty200,
    get_all_pit_tickers,
    get_pit_universe,
    TICKER_ALIASES,
    NIFTY_NEXT50_TICKERS,
    NIFTY_200_NEXT100_TICKERS,
    _fetch_alias,
)
from data.momentum_engine import NIFTY_50_TICKERS


def compute_rsi(closes, period=14):
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


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
        if bub[i] < final_ub[i-1] or c[i-1] > final_ub[i-1]:
            final_ub[i] = bub[i]
        else:
            final_ub[i] = final_ub[i-1]
        if blb[i] > final_lb[i-1] or c[i-1] < final_lb[i-1]:
            final_lb[i] = blb[i]
        else:
            final_lb[i] = final_lb[i-1]
        if direction[i-1] == 1:
            direction[i] = -1 if c[i] < final_lb[i] else 1
        else:
            direction[i] = 1 if c[i] > final_ub[i] else -1

    return pd.Series(direction, index=closes.index)


def compute_charges(entry_price, exit_price, shares):
    """Zerodha equity delivery charges per trade."""
    B = entry_price * shares
    S = exit_price * shares
    total_turnover = B + S
    stt = 0.001 * total_turnover
    exchange = 0.0000307 * total_turnover
    sebi = 0.000001 * total_turnover
    stamp = 0.00015 * B
    gst = 0.18 * (exchange + sebi)
    total_charges = stt + exchange + sebi + stamp + gst
    return total_charges, exchange + sebi + stamp + gst  # (total, deductible)


def compute_net_profit(pnl, entry_price, exit_price, shares):
    """Net profit after charges and STCG tax."""
    total_charges, deductible = compute_charges(entry_price, exit_price, shares)
    taxable = pnl - deductible
    stcg = 0.20 * taxable if taxable > 0 else 0.0
    net = pnl - total_charges - stcg
    return net


def run_phase4(
    stock_data, indicators, date_to_iloc, trading_days, bt_start_date,
    pit_data, capital_lakhs, max_positions,
    rsi_exit_threshold, rsi_exit_days
):
    """Phase 4 daily backtest loop with parameterized RSI exit."""
    TOTAL_CAPITAL = capital_lakhs * 100_000
    PER_STOCK = TOTAL_CAPITAL // max_positions

    portfolio = {}
    trades = []
    cash = float(TOTAL_CAPITAL)

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
                    "symbol": ticker,
                    "entry_date": pos["entry_date"], "exit_date": day,
                    "entry_price": pos["entry_price"], "exit_price": price,
                    "shares": shares, "pnl": pnl,
                    "exit_reason": "HARD_SL_8PCT",
                    "hold_days": (day - pos["entry_date"]).days,
                })
                cash += shares * price
                del portfolio[ticker]
                continue

            # Exit 2: RSI(14, 3d avg) < threshold for N consecutive days (PARAMETERIZED)
            rsi_v = float(ind["rsi14"].iloc[ci]) if ci < len(ind["rsi14"]) and not pd.isna(ind["rsi14"].iloc[ci]) else 50
            if rsi_v < rsi_exit_threshold:
                pos["rsi_low_count"] = pos.get("rsi_low_count", 0) + 1
            else:
                pos["rsi_low_count"] = 0
            if pos.get("rsi_low_count", 0) >= rsi_exit_days:
                pnl = (price - pos["entry_price"]) * shares
                trades.append({
                    "symbol": ticker,
                    "entry_date": pos["entry_date"], "exit_date": day,
                    "entry_price": pos["entry_price"], "exit_price": price,
                    "shares": shares, "pnl": pnl,
                    "exit_reason": f"RSI_BELOW{int(rsi_exit_threshold)}_{rsi_exit_days}D",
                    "hold_days": (day - pos["entry_date"]).days,
                })
                cash += shares * price
                del portfolio[ticker]
                continue

            # Exit 3: RS63 (5d avg) < 0
            rs55_v = float(ind["rs55"].iloc[ci]) if ci < len(ind["rs55"]) and not pd.isna(ind["rs55"].iloc[ci]) else 0
            if rs55_v < 0:
                pnl = (price - pos["entry_price"]) * shares
                trades.append({
                    "symbol": ticker,
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
                    "symbol": ticker,
                    "entry_date": pos["entry_date"], "exit_date": day,
                    "entry_price": pos["entry_price"], "exit_price": price,
                    "shares": shares, "pnl": pnl,
                    "exit_reason": "TIME_STOP",
                    "hold_days": days_held,
                })
                cash += shares * price
                del portfolio[ticker]

        # === ENTRY SIGNALS ===
        if len(portfolio) >= max_positions:
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

            rs55_v = float(ind["rs55"].iloc[ci]) if not pd.isna(ind["rs55"].iloc[ci]) else -1
            rsi_v = float(ind["rsi14"].iloc[ci]) if not pd.isna(ind["rsi14"].iloc[ci]) else 0

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

            l20d = float(ind["low_20d"].iloc[ci]) if not pd.isna(ind["low_20d"].iloc[ci]) else 0
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
            if len(portfolio) >= max_positions:
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
            "symbol": ticker,
            "entry_date": pos["entry_date"], "exit_date": last_day,
            "entry_price": pos["entry_price"], "exit_price": exit_price,
            "shares": shares, "pnl": pnl,
            "exit_reason": "BACKTEST_END",
            "hold_days": (last_day - pos["entry_date"]).days,
        })

    return trades


def compute_metrics(trades, capital_lakhs=10):
    """Compute summary metrics from trades list."""
    if not trades:
        return {"trades": 0, "wr": 0, "pf": 0, "net_pct_yr": 0, "neg_years": 0, "yearly": {}}

    CAPITAL = capital_lakhs * 100_000
    df = pd.DataFrame(trades)
    df["exit_year"] = df["exit_date"].apply(lambda d: d.year)

    # Compute net profit per trade
    df["net_profit"] = df.apply(
        lambda r: compute_net_profit(r["pnl"], r["entry_price"], r["exit_price"], r["shares"]),
        axis=1
    )

    n = len(df)
    wr = (df["pnl"] > 0).mean() * 100
    gross_win = df[df["pnl"] > 0]["pnl"].sum()
    gross_loss = df[df["pnl"] < 0]["pnl"].abs().sum()
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    # Year-by-year net returns
    yearly_net = df.groupby("exit_year")["net_profit"].sum()
    yearly_pct = (yearly_net / CAPITAL * 100).round(1)

    neg_years = (yearly_pct < 0).sum()

    # Annualized: total net profit over all years / number of years / capital
    years_in_bt = (df["exit_date"].max().year - df["exit_date"].min().year + 1)
    avg_net_yr = yearly_pct.mean()  # simple average of annual returns

    return {
        "trades": n,
        "wr": round(wr, 1),
        "pf": round(pf, 2),
        "net_pct_yr": round(avg_net_yr, 1),
        "neg_years": int(neg_years),
        "yearly": yearly_pct.to_dict(),
    }


def load_shared_data(period_days=365*11, capital_lakhs=10, max_positions=7, pit_universe=True):
    """Load Phases 1-3 once: data, indicators, date mapping."""

    pit_data = None
    if pit_universe:
        pit_data = load_pit_nifty200()
        if pit_data is None:
            print("  WARNING: pit_universe=True but nifty200_pit.json not found")

    if pit_data is not None:
        all_pit_tickers = get_all_pit_tickers(pit_data)
        tickers = sorted(all_pit_tickers)
        print(f"  PIT universe: {len(tickers)} unique tickers")
    else:
        tickers = NIFTY_50_TICKERS + NIFTY_NEXT50_TICKERS + NIFTY_200_NEXT100_TICKERS

    end_date = datetime.now()
    daily_start = end_date - timedelta(days=period_days + 600)
    bt_start_date = (end_date - timedelta(days=period_days)).date()

    # --- Phase 1: Fetch data ---
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
        return None

    bench_close = bench_data["Close"]

    # --- Phase 2: Pre-compute indicators ---
    print("  Pre-computing indicators...")
    indicators = {}
    for ticker, df in stock_data.items():
        closes = df["Close"]
        highs = df["High"]
        lows = df["Low"]
        volumes = df["Volume"]

        bench_aligned = bench_close.reindex(df.index, method="ffill")

        rs_ratio = closes / bench_aligned
        rs55_raw = (rs_ratio / rs_ratio.shift(63) - 1) * 100
        rs55 = rs55_raw.rolling(5, min_periods=1).mean()
        rs123_raw = (rs_ratio / rs_ratio.shift(123) - 1) * 100
        rs123 = rs123_raw.rolling(5, min_periods=1).mean()
        rs21 = (rs_ratio / rs_ratio.shift(21) - 1) * 100

        rsi14_raw = compute_rsi(closes, 14)
        rsi14 = rsi14_raw.rolling(3, min_periods=1).mean()  # 3d smoothed for exit

        ema200 = closes.ewm(span=200, adjust=False).mean()
        ema21 = closes.ewm(span=21, adjust=False).mean()

        high_20d = highs.rolling(20).max().shift(1)
        low_20d = lows.rolling(20).min().shift(1)
        vol_avg20 = volumes.rolling(20).mean()

        indicators[ticker] = {
            "rs55": rs55,
            "rs123": rs123,
            "rs21": rs21,
            "rsi14": rsi14,
            "ema200": ema200,
            "ema21": ema21,
            "high_20d": high_20d,
            "low_20d": low_20d,
            "vol_avg20": vol_avg20,
        }

    # --- Phase 3: Build date mapping ---
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

    return {
        "stock_data": stock_data,
        "indicators": indicators,
        "date_to_iloc": date_to_iloc,
        "trading_days": trading_days,
        "bt_start_date": bt_start_date,
        "pit_data": pit_data,
    }


def main():
    CAPITAL_LAKHS = 10
    MAX_POSITIONS = 7
    PERIOD_DAYS = 365 * 11

    rsi_thresholds = [35, 40, 45]
    rsi_days_list = [2, 3, 4]

    print("=" * 60)
    print("RS63 Sweep #6 — RSI Exit Threshold x Consecutive Days")
    print("=" * 60)
    print("\nLoading shared data (Phases 1-3)...")

    shared = load_shared_data(
        period_days=PERIOD_DAYS,
        capital_lakhs=CAPITAL_LAKHS,
        max_positions=MAX_POSITIONS,
        pit_universe=True,
    )
    if shared is None:
        print("ERROR: Could not load data.")
        return

    print("\nRunning 3x3 grid (9 combinations)...\n")

    results = {}
    for rsi_thresh in rsi_thresholds:
        for rsi_days in rsi_days_list:
            label = f"RSI<{rsi_thresh} x {rsi_days}d"
            print(f"  Running: {label} ...")
            trades = run_phase4(
                stock_data=shared["stock_data"],
                indicators=shared["indicators"],
                date_to_iloc=shared["date_to_iloc"],
                trading_days=shared["trading_days"],
                bt_start_date=shared["bt_start_date"],
                pit_data=shared["pit_data"],
                capital_lakhs=CAPITAL_LAKHS,
                max_positions=MAX_POSITIONS,
                rsi_exit_threshold=rsi_thresh,
                rsi_exit_days=rsi_days,
            )
            m = compute_metrics(trades, CAPITAL_LAKHS)
            results[(rsi_thresh, rsi_days)] = m
            print(f"    => {m['trades']} tr | {m['wr']}% WR | PF {m['pf']} | {m['net_pct_yr']}%/yr | {m['neg_years']} neg yr")

    # --- Print 3x3 grid table ---
    print("\n" + "=" * 70)
    print("RESULTS GRID (Avg Net %/yr)")
    print("=" * 70)
    header = f"{'':12}" + "".join(f"  {d}d consec" for d in rsi_days_list)
    print(header)
    print("-" * 70)
    for rsi_thresh in rsi_thresholds:
        row = f"RSI < {rsi_thresh:2d}   "
        for rsi_days in rsi_days_list:
            m = results[(rsi_thresh, rsi_days)]
            row += f"  {m['net_pct_yr']:>7.1f}%  "
        print(row)

    print("\n" + "=" * 70)
    print("TRADES COUNT GRID")
    print("=" * 70)
    print(header)
    print("-" * 70)
    for rsi_thresh in rsi_thresholds:
        row = f"RSI < {rsi_thresh:2d}   "
        for rsi_days in rsi_days_list:
            m = results[(rsi_thresh, rsi_days)]
            row += f"  {m['trades']:>7d}tr  "
        print(row)

    print("\n" + "=" * 70)
    print("WIN RATE GRID")
    print("=" * 70)
    print(header)
    print("-" * 70)
    for rsi_thresh in rsi_thresholds:
        row = f"RSI < {rsi_thresh:2d}   "
        for rsi_days in rsi_days_list:
            m = results[(rsi_thresh, rsi_days)]
            row += f"  {m['wr']:>6.1f}%   "
        print(row)

    print("\n" + "=" * 70)
    print("PROFIT FACTOR GRID")
    print("=" * 70)
    print(header)
    print("-" * 70)
    for rsi_thresh in rsi_thresholds:
        row = f"RSI < {rsi_thresh:2d}   "
        for rsi_days in rsi_days_list:
            m = results[(rsi_thresh, rsi_days)]
            row += f"  {m['pf']:>7.2f}   "
        print(row)

    print("\n" + "=" * 70)
    print("NEGATIVE YEARS GRID")
    print("=" * 70)
    print(header)
    print("-" * 70)
    for rsi_thresh in rsi_thresholds:
        row = f"RSI < {rsi_thresh:2d}   "
        for rsi_days in rsi_days_list:
            m = results[(rsi_thresh, rsi_days)]
            row += f"  {m['neg_years']:>5d} neg  "
        print(row)

    # --- Find best combo ---
    best_key = max(results.keys(), key=lambda k: results[k]["net_pct_yr"])
    best = results[best_key]
    print(f"\n{'=' * 70}")
    print(f"BEST COMBO: RSI < {best_key[0]}, {best_key[1]} consecutive days")
    print(f"  Trades: {best['trades']} | WR: {best['wr']}% | PF: {best['pf']} | Net/yr: {best['net_pct_yr']}% | Neg years: {best['neg_years']}")
    print(f"\nYear-by-year for best combo (RSI<{best_key[0]}, {best_key[1]}d):")
    print(f"{'Year':<6} {'Net%':>8}")
    print("-" * 16)
    for yr in sorted(best["yearly"].keys()):
        pct = best["yearly"][yr]
        marker = " <-- NEG" if pct < 0 else ""
        print(f"{yr:<6} {pct:>7.1f}%{marker}")

    # Also print baseline (40, 3) year-by-year for reference
    base_key = (40, 3)
    base = results[base_key]
    print(f"\nBaseline (RSI<40, 3d) year-by-year:")
    print(f"{'Year':<6} {'Net%':>8}")
    print("-" * 16)
    for yr in sorted(base["yearly"].keys()):
        pct = base["yearly"][yr]
        marker = " <-- NEG" if pct < 0 else ""
        print(f"{yr:<6} {pct:>7.1f}%{marker}")

    print(f"\nBaseline summary: {base['trades']} tr | {base['wr']}% WR | PF {base['pf']} | {base['net_pct_yr']}%/yr | {base['neg_years']} neg yr")


if __name__ == "__main__":
    main()
