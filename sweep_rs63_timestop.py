"""
Sweep #7 — RS63 Time Stop: weeks x gain threshold grid
4x4 grid: weeks=[6,8,10,12], gain_threshold=[1%,2%,3%,5%]
Runs Phases 1-3 once, then Phase 4 for each combo.
"""
import sys
sys.path.insert(0, "/Users/jay/Desktop/relative_strength")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Import helpers from momentum_backtest and momentum_engine
from data.momentum_backtest import (
    MomentumBacktester,
    load_pit_nifty200,
    get_all_pit_tickers,
    get_pit_universe,
    NIFTY_NEXT50_TICKERS,
    NIFTY_200_NEXT100_TICKERS,
    TICKER_ALIASES,
    _fetch_alias,
)
from data.momentum_engine import NIFTY_50_TICKERS

# ── Config ───────────────────────────────────────────────────────────────────
CAPITAL_LAKHS = 10
MAX_POSITIONS = 7
PERIOD_DAYS = 365 * 11
PIT = True

WEEKS_GRID = [6, 8, 10, 12]
GAIN_GRID  = [1.0, 2.0, 3.0, 5.0]

BASELINE_DAYS = 56   # 8 weeks
BASELINE_GAIN = 3.0


# ── Zerodha charges (equity delivery) ────────────────────────────────────────
def compute_charges(entry_price, exit_price, shares):
    B = entry_price * shares
    S = exit_price  * shares
    total_turnover = B + S
    stt      = 0.001     * total_turnover
    exchange = 0.0000307 * total_turnover
    sebi     = 0.000001  * total_turnover
    stamp    = 0.00015   * B
    gst      = 0.18      * (exchange + sebi)
    total    = stt + exchange + sebi + stamp + gst
    deductible = exchange + sebi + stamp + gst   # STT not deductible
    return total, deductible


def net_profit_trade(pnl, entry_price, exit_price, shares):
    total_chg, deductible = compute_charges(entry_price, exit_price, shares)
    taxable = pnl - deductible
    stcg    = 0.20 * taxable if taxable > 0 else 0.0
    return pnl - total_chg - stcg


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(trades, capital):
    if not trades:
        return dict(n=0, wr=0, pf=0, net_yr=0, neg_yrs=0, yr_returns={})

    df = pd.DataFrame(trades)

    wins   = df[df["pnl"] > 0]["pnl"].sum()
    losses = df[df["pnl"] < 0]["pnl"].abs().sum()
    pf     = wins / losses if losses > 0 else float("inf")
    wr     = (df["pnl"] > 0).mean() * 100

    df["net_pnl"] = df.apply(
        lambda r: net_profit_trade(r["pnl"], r["entry_price"], r["exit_price"], r["shares"]),
        axis=1
    )

    df["exit_year"] = pd.to_datetime(df["exit_date"].astype(str)).dt.year
    yr_net = df.groupby("exit_year")["net_pnl"].sum()
    yr_ret = (yr_net / capital * 100).round(1)

    all_years = list(range(2015, 2026))
    yr_returns = {y: float(yr_ret.get(y, 0.0)) for y in all_years}

    net_pct_yr = float(yr_ret.mean())
    neg_yrs    = int((yr_ret < 0).sum())

    return dict(
        n=len(df), wr=round(wr, 1), pf=round(pf, 2),
        net_yr=round(net_pct_yr, 1), neg_yrs=neg_yrs,
        yr_returns=yr_returns
    )


# ── Phase 4: daily loop (parameterised) ──────────────────────────────────────
def run_phase4(stock_data, indicators, date_to_iloc, trading_days,
               pit_data, capital, max_positions,
               time_stop_days, time_stop_min_gain):
    """Phase 4 daily backtest loop with parameterised time-stop."""
    PER_STOCK = capital // max_positions

    portfolio = {}
    trades    = []
    cash      = float(capital)

    for day_idx, day in enumerate(trading_days):
        pit_set = get_pit_universe(pit_data, day) if pit_data is not None else None

        # === EXIT PROCESSING ===
        for ticker in list(portfolio.keys()):
            pos     = portfolio[ticker]
            idx_map = date_to_iloc.get(ticker, {})
            ci      = idx_map.get(day)
            if ci is None:
                continue

            df     = stock_data[ticker]
            price  = float(df["Close"].iloc[ci])
            ind    = indicators[ticker]
            shares = pos["shares"]

            # Exit 1: 8% hard stop
            if price <= pos["entry_price"] * 0.92:
                pnl = (price - pos["entry_price"]) * shares
                trades.append({
                    "symbol": ticker, "entry_date": pos["entry_date"], "exit_date": day,
                    "entry_price": pos["entry_price"], "exit_price": price,
                    "shares": shares, "pnl": pnl, "exit_reason": "HARD_SL_8PCT",
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
                    "symbol": ticker, "entry_date": pos["entry_date"], "exit_date": day,
                    "entry_price": pos["entry_price"], "exit_price": price,
                    "shares": shares, "pnl": pnl, "exit_reason": "RSI_BELOW40_3D",
                })
                cash += shares * price
                del portfolio[ticker]
                continue

            # Exit 3: RS63 (5d avg) < 0
            rs55_v = float(ind["rs55"].iloc[ci]) if ci < len(ind["rs55"]) and not pd.isna(ind["rs55"].iloc[ci]) else 0
            if rs55_v < 0:
                pnl = (price - pos["entry_price"]) * shares
                trades.append({
                    "symbol": ticker, "entry_date": pos["entry_date"], "exit_date": day,
                    "entry_price": pos["entry_price"], "exit_price": price,
                    "shares": shares, "pnl": pnl, "exit_reason": "RS63_NEG",
                })
                cash += shares * price
                del portfolio[ticker]
                continue

            # Exit 4: Time stop — parameterised
            days_held = (day - pos["entry_date"]).days
            gain_pct  = (price - pos["entry_price"]) / pos["entry_price"] * 100
            if days_held >= time_stop_days and gain_pct < time_stop_min_gain:
                pnl = (price - pos["entry_price"]) * shares
                trades.append({
                    "symbol": ticker, "entry_date": pos["entry_date"], "exit_date": day,
                    "entry_price": pos["entry_price"], "exit_price": price,
                    "shares": shares, "pnl": pnl, "exit_reason": "TIME_STOP",
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
            ci      = idx_map.get(day)
            if ci is None or ci < 210:
                continue

            ind        = indicators[ticker]
            price      = float(df["Close"].iloc[ci])
            open_price = float(df["Open"].iloc[ci])

            rs55_v = float(ind["rs55"].iloc[ci]) if not pd.isna(ind["rs55"].iloc[ci]) else -1
            rsi_v  = float(ind["rsi14"].iloc[ci]) if not pd.isna(ind["rsi14"].iloc[ci]) else 0

            if rs55_v <= 0 or rsi_v <= 50:
                continue

            high_today = float(df["High"].iloc[ci])
            low_today  = float(df["Low"].iloc[ci])
            day_range  = high_today - low_today
            ibs = (price - low_today) / day_range if day_range > 0 else 0.5
            if ibs <= 0.5:
                continue
            if price <= open_price:
                continue

            l20d     = float(ind["low_20d"].iloc[ci]) if not pd.isna(ind["low_20d"].iloc[ci]) else 0
            stop_pct = (price - l20d) / price * 100 if price > 0 and l20d > 0 else 8

            signals.append({"ticker": ticker, "price": price, "stop_pct": stop_pct})

        signals.sort(key=lambda s: s["stop_pct"])
        for sig in signals:
            if len(portfolio) >= max_positions:
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
                "entry_date": day, "entry_price": price, "shares": shares,
                "rsi_low_count": 0,
            }
            cash -= cost

    # Close remaining at end
    last_day = trading_days[-1]
    for ticker in list(portfolio.keys()):
        pos     = portfolio[ticker]
        idx_map = date_to_iloc.get(ticker, {})
        ci      = idx_map.get(last_day)
        exit_price = float(stock_data[ticker]["Close"].iloc[ci]) if ci is not None else pos["entry_price"]
        shares = pos["shares"]
        pnl    = (exit_price - pos["entry_price"]) * shares
        trades.append({
            "symbol": ticker, "entry_date": pos["entry_date"], "exit_date": last_day,
            "entry_price": pos["entry_price"], "exit_price": exit_price,
            "shares": shares, "pnl": pnl, "exit_reason": "BACKTEST_END",
        })

    return trades


# ── Print helpers ─────────────────────────────────────────────────────────────
def print_grid(results, field, title, fmt_fn):
    col_w = 12
    print()
    print(title)
    print("-" * 70)
    header = "weeks\\gain"
    line = header.rjust(10)
    for g in GAIN_GRID:
        line += (str(g) + "%").rjust(col_w)
    print(line)
    print("-" * 70)
    for weeks in WEEKS_GRID:
        row = (str(weeks) + "w").rjust(10)
        for gain in GAIN_GRID:
            m   = results[(weeks, gain)]
            tag = "*" if (weeks == BASELINE_DAYS // 7 and gain == BASELINE_GAIN) else ""
            val = fmt_fn(m[field]) + tag
            row += val.rjust(col_w)
        print(row)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    TOTAL_CAPITAL = CAPITAL_LAKHS * 100_000
    end_date      = datetime.now()
    daily_start   = end_date - timedelta(days=PERIOD_DAYS + 600)
    bt_start_date = (end_date - timedelta(days=PERIOD_DAYS)).date()

    # Phase 1: Load data
    print("=" * 60)
    print("PHASE 1: Loading data...")

    pit_data = load_pit_nifty200() if PIT else None
    if pit_data is not None:
        all_pit_tickers = get_all_pit_tickers(pit_data)
        tickers = sorted(all_pit_tickers)
        print(f"  PIT universe: {len(tickers)} unique tickers")
    else:
        tickers = NIFTY_50_TICKERS + NIFTY_NEXT50_TICKERS + NIFTY_200_NEXT100_TICKERS

    stock_data = {}
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
            print(f"  Loaded {idx+1}/{len(tickers)} stocks...")

    print(f"  Data loaded: {len(stock_data)} stocks")

    try:
        bench_data = yf.Ticker("^CNX200").history(start=daily_start, end=end_date)
        print(f"  Nifty 200 benchmark: {len(bench_data)} bars")
    except Exception:
        print("ERROR: Cannot fetch Nifty 200 benchmark")
        return

    bench_close = bench_data["Close"]

    # Phase 2: Pre-compute indicators
    print("PHASE 2: Pre-computing indicators...")

    def compute_rsi(closes, period=14):
        delta    = closes.diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        return 100 - (100 / (1 + rs))

    indicators = {}
    for ticker, df in stock_data.items():
        closes = df["Close"]
        lows   = df["Low"]

        bench_aligned = bench_close.reindex(df.index, method="ffill")
        rs_ratio = closes / bench_aligned
        rs55_raw = (rs_ratio / rs_ratio.shift(63) - 1) * 100
        rs55     = rs55_raw.rolling(5, min_periods=1).mean()

        rsi14_raw = compute_rsi(closes, 14)
        rsi14     = rsi14_raw.rolling(3, min_periods=1).mean()

        low_20d = lows.rolling(20).min().shift(1)

        indicators[ticker] = {"rs55": rs55, "rsi14": rsi14, "low_20d": low_20d}

    # Phase 3: Build date mapping
    print("PHASE 3: Building date mapping...")
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

    # Phase 4: Sweep
    print()
    print("=" * 60)
    print("PHASE 4: Running 4x4 sweep...")
    print("=" * 60)

    results = {}
    total = len(WEEKS_GRID) * len(GAIN_GRID)
    done  = 0
    for weeks in WEEKS_GRID:
        for gain in GAIN_GRID:
            days_param = weeks * 7
            done += 1
            print(f"  [{done}/{total}] weeks={weeks} ({days_param}d), gain<{gain}%...", end=" ", flush=True)
            trades = run_phase4(
                stock_data, indicators, date_to_iloc, trading_days,
                pit_data, TOTAL_CAPITAL, MAX_POSITIONS,
                time_stop_days=days_param,
                time_stop_min_gain=gain,
            )
            m = compute_metrics(trades, TOTAL_CAPITAL)
            results[(weeks, gain)] = m
            print(f"n={m['n']} WR={m['wr']}% PF={m['pf']} net/yr={m['net_yr']}% neg={m['neg_yrs']}")

    # Print grids
    print()
    print("=" * 70)
    print("SWEEP RESULTS (* = baseline 8w/3%)")
    print_grid(results, "net_yr",  "Net %/yr",     lambda v: str(v) + "%")
    print_grid(results, "n",       "Trade count",  str)
    print_grid(results, "wr",      "Win Rate %",   lambda v: str(v) + "%")
    print_grid(results, "pf",      "Profit Factor",str)
    print_grid(results, "neg_yrs", "Neg years",    str)

    # Best combo
    best_key = max(results, key=lambda k: results[k]["net_yr"])
    best_m   = results[best_key]
    baseline = results[(BASELINE_DAYS // 7, BASELINE_GAIN)]

    print()
    print("=" * 70)
    print("BEST COMBO: weeks={}, gain<{}%".format(best_key[0], best_key[1]))
    print("  n={}, WR={}%, PF={}, net/yr={}%, neg={}".format(
        best_m["n"], best_m["wr"], best_m["pf"], best_m["net_yr"], best_m["neg_yrs"]))
    print("BASELINE : weeks=8, gain<3%")
    print("  n={}, WR={}%, PF={}, net/yr={}%, neg={}".format(
        baseline["n"], baseline["wr"], baseline["pf"], baseline["net_yr"], baseline["neg_yrs"]))

    # Year-by-year comparison: best vs baseline
    print()
    print("=" * 70)
    print("YEAR-BY-YEAR: Best ({}w/<{}%) vs Baseline (8w/<3%)".format(best_key[0], best_key[1]))
    print("=" * 70)
    print("{:<6} {:>10} {:>10} {:>10}".format("Year", "Best", "Baseline", "Delta"))
    print("-" * 40)
    for yr in range(2015, 2026):
        b  = best_m["yr_returns"].get(yr, 0.0)
        bl = baseline["yr_returns"].get(yr, 0.0)
        marker = " <-- NEG" if b < 0 else ""
        print("{:<6} {:>10} {:>10} {:>10}{}".format(
            yr,
            str(round(b, 1)) + "%",
            str(round(bl, 1)) + "%",
            ("+" if b - bl >= 0 else "") + str(round(b - bl, 1)) + "%",
            marker
        ))

    avg_best = sum(best_m["yr_returns"].values()) / len(best_m["yr_returns"])
    avg_base = sum(baseline["yr_returns"].values()) / len(baseline["yr_returns"])
    print("-" * 40)
    print("{:<6} {:>10} {:>10} {:>10}".format(
        "Avg",
        str(round(avg_best, 1)) + "%",
        str(round(avg_base, 1)) + "%",
        ("+" if avg_best - avg_base >= 0 else "") + str(round(avg_best - avg_base, 1)) + "%"
    ))

    # Also print year-by-year for baseline
    print()
    print("=" * 70)
    print("YEAR-BY-YEAR: Baseline (8w/<3%)")
    print("=" * 70)
    for yr in range(2015, 2026):
        bl = baseline["yr_returns"].get(yr, 0.0)
        marker = " <-- NEG" if bl < 0 else ""
        print("{:<6} {:>10}{}".format(yr, str(round(bl, 1)) + "%", marker))
    print("-" * 20)
    print("{:<6} {:>10}".format("Avg", str(round(avg_base, 1)) + "%"))


if __name__ == "__main__":
    main()
