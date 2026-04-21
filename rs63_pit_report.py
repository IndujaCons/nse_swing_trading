#!/usr/bin/env python3
"""
RS63 PIT Backtest — Vivek Bajaj RS Strategy (Event-Driven Daily)
=================================================================
Capital  : ₹10L, max 7 positions (~₹1.43L each), fixed slot size
Universe : Nifty 200 PIT (historically correct constituents)
Entry    : RS63(5d avg) > 0  AND  RSI(14, 3d avg) > 50
           AND  IBS > 0.5  AND  green candle
Exit     : 8% hard SL  OR  RSI(14,3d avg) < 40 for 3 days
           OR  RS63(5d avg) < 0  OR  8-week time stop + <3% gain
Ranking  : Volume ratio (today / 20d avg) — highest first

Usage:
    python3 rs63_pit_report.py           # first run downloads data (~5-10 min)
    python3 rs63_pit_report.py --refresh # force re-download
    python3 rs63_pit_report.py > rs63_report.md
"""

import os, sys, json, pickle, warnings, argparse, time
from datetime import date, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
START_DATE    = date(2015, 4, 9)   # exact match: Apr 6 2026 − 4015 days
CAPITAL       = 10_00_000        # ₹10L (baseline)
MAX_POSITIONS = 7                # baseline: 7 positions, fixed slot
WARMUP_DAYS   = 450

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, 'data', 'cache', 'rs63_pit_report.pkl')
PIT_FILE   = os.path.join(BASE_DIR, 'nse_const', 'nifty200_pit.json')

# ── CHARGES ───────────────────────────────────────────────────────────────────
def calc_charges(buy_val, sell_val):
    total = buy_val + sell_val
    stt   = 0.001     * total
    exch  = 0.0000307 * total
    sebi  = 0.000001  * total
    stamp = 0.00015   * buy_val
    gst   = 0.18      * (exch + sebi)
    return stt + exch + sebi + stamp + gst

# ── TICKER ALIASES ────────────────────────────────────────────────────────────
ALIASES = {
    "ETERNAL":    "ZOMATO.NS",   "ZOMATO":     "ETERNAL.NS",
    "ALBK":       "INDIANB.NS",  "ANDHRABANK": "UNIONBANK.NS",
    "CROMPGREAV": "CGPOWER.NS",  "GMRINFRA":   "GMRAIRPORT.NS",
    "HDFC":       "HDFCBANK.NS", "IDFC":       "IDFCFIRSTB.NS",
    "IDFCBANK":   "IDFCFIRSTB.NS","JUBILANT":  "JUBLPHARMA.NS",
    "KPIT":       "BSOFT.NS",    "LTI":        "LTIM.NS",
    "LTM":        "LTIM.NS",     "MINDTREE":   "LTIM.NS",
    "MAX":        "MFSL.NS",     "NIITTECH":   "COFORGE.NS",
    "ORIENTBANK": "PNB.NS",      "RNAM":       "NAM-INDIA.NS",
    "STRTECH":    "STLTECH.NS",  "SYNDIBANK":  "CANBK.NS",
    "WELSPUNIND": "WELSPUNLIV.NS",
    "ALSTOMT&D":  "532309.BO",   "AMARAJABAT": "500008.BO",
    "DHFL":       "511072.BO",   "GSKCONS":    "500676.BO",
    "IBULHSGFIN": "535789.BO",   "MCDOWELL-N": "UNITDSPR.BO",
    "PEL":        "500302.BO",   "TMPV":       "TMPV.BO",
}

# ── PIT UNIVERSE ──────────────────────────────────────────────────────────────
def load_pit():
    with open(PIT_FILE) as f:
        raw = json.load(f)
    return sorted(
        [(pd.Timestamp(k).date(), set(v)) for k, v in raw.items()],
        key=lambda x: x[0]
    )

def get_pit_universe(pit_data, day):
    universe = set()
    for eff_date, symbols in pit_data:
        if eff_date <= day:
            universe = symbols
    return universe

def get_all_pit_tickers(pit_data):
    all_t = set()
    for _, s in pit_data:
        all_t |= s
    return all_t

# ── DATA ──────────────────────────────────────────────────────────────────────
def fetch_ticker(ticker, start, end):
    sym = ALIASES.get(ticker, f"{ticker}.NS")
    try:
        df = yf.Ticker(sym).history(start=start, end=end)
        if df.empty or len(df) < 200:
            if ".BO" not in sym:
                df2 = yf.Ticker(f"{ticker}.NS").history(start=start, end=end)
                if len(df2) > len(df):
                    df = df2
    except Exception:
        df = pd.DataFrame()
    if not df.empty:
        df.index = df.index.tz_localize(None) if df.index.tzinfo else df.index
    return df

def fetch_with_retry(symbol, start, end, label=""):
    for attempt in range(3):
        try:
            df = yf.Ticker(symbol).history(start=start, end=end)
            if not df.empty:
                df.index = df.index.tz_localize(None) if df.index.tzinfo else df.index
                return df
        except Exception as e:
            print(f"  {label} attempt {attempt+1} failed: {e}")
        time.sleep(2)
    return pd.DataFrame()

def load_or_fetch(tickers, fetch_start, fetch_end, refresh=False):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    if not refresh and os.path.exists(CACHE_FILE):
        print(f"Loading cached data from {os.path.basename(CACHE_FILE)}...")
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    print(f"Fetching {len(tickers)} tickers (this takes ~5-10 min)...")
    stock_data = {}
    for i, ticker in enumerate(sorted(tickers)):
        df = fetch_ticker(ticker, fetch_start, fetch_end)
        if not df.empty and len(df) >= 200:
            stock_data[ticker] = df
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(tickers)} done, {len(stock_data)} loaded...")
    print(f"  Done. {len(stock_data)} stocks loaded.")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(stock_data, f)
    print(f"  Cached to {CACHE_FILE}")
    return stock_data

# ── INDICATORS ────────────────────────────────────────────────────────────────
def compute_rsi(closes, period=14):
    delta    = closes.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return 100 - (100 / (1 + avg_gain / avg_loss.replace(0, 1e-10)))

def build_indicators(stock_data, bench_close):
    print("  Pre-computing indicators...")
    # Ensure bench index is tz-naive for reindex compatibility
    if bench_close.index.tzinfo is not None:
        bench_close = bench_close.copy()
        bench_close.index = bench_close.index.tz_localize(None)
    indicators = {}
    for ticker, df in stock_data.items():
        # Ensure stock index is also tz-naive
        if df.index.tzinfo is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        closes  = df["Close"]
        highs   = df["High"]
        lows    = df["Low"]
        volumes = df["Volume"]

        bench_aligned = bench_close.reindex(df.index, method="ffill")
        rs_ratio  = closes / bench_aligned
        rs63_raw  = (rs_ratio / rs_ratio.shift(63) - 1) * 100
        rs63      = rs63_raw.rolling(5, min_periods=1).mean()   # smoothed RS63

        rsi14_raw  = compute_rsi(closes, 14)
        rsi14_3d   = rsi14_raw.rolling(3, min_periods=1).mean() # entry
        rsi14_2d   = rsi14_raw.rolling(2, min_periods=1).mean() # exit

        vol_avg20  = volumes.rolling(20).mean()
        low_5d     = lows.rolling(5).min().shift(1)
        low_20d    = lows.rolling(20).min().shift(1)
        sma20      = closes.rolling(20).mean().shift(1)

        indicators[ticker] = {
            "rs63":      rs63,
            "rsi14_3d":  rsi14_3d,
            "rsi14_2d":  rsi14_2d,
            "vol_avg20": vol_avg20,
            "low_5d":    low_5d,
            "low_20d":   low_20d,
            "sma20":     sma20,
        }
    print(f"  Indicators built for {len(indicators)} stocks")
    return indicators

# ── FORMATTING ────────────────────────────────────────────────────────────────
def inr(v):
    return f"₹{v:,.0f}"

def pct(v):
    return f"{'+' if v >= 0 else ''}{v:.1f}%"

def print_table(headers, rows, col_widths):
    sep = "  ".join("─" * w for w in col_widths)
    hdr = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(f"  {hdr}")
    print(f"  {sep}")
    for row in rows:
        print("  " + "  ".join(str(c).ljust(w) for c, w in zip(row, col_widths)))

# ── MAIN BACKTEST ─────────────────────────────────────────────────────────────
def run(refresh=False, end_date=None, max_daily_entries=None):
    print("Loading PIT universe...")
    pit_data    = load_pit()
    all_tickers = get_all_pit_tickers(pit_data)
    print(f"  {len(all_tickers)} unique PIT tickers")

    fetch_start = START_DATE - timedelta(days=WARMUP_DAYS)
    fetch_end   = date.today()

    # Try to load from the rs55 shared cache (4-tuple format) first
    RS55_CACHE = os.path.join(BASE_DIR, 'data', 'cache', 'rs63_4015d_pit_20260406.pkl')
    if not refresh and os.path.exists(RS55_CACHE):
        print(f"Loading shared rs55 cache: {os.path.basename(RS55_CACHE)}...")
        with open(RS55_CACHE, 'rb') as f:
            stock_data, bench_raw, _, _ = pickle.load(f)
        bench_raw.index = bench_raw.index.tz_localize(None) if bench_raw.index.tzinfo else bench_raw.index
        print(f"  {len(stock_data)} stocks, {len(bench_raw)} bench bars")
    else:
        stock_data = load_or_fetch(all_tickers, fetch_start, fetch_end, refresh)
        print("Fetching Nifty 200 benchmark...")
        bench_raw = fetch_with_retry("^CNX200", fetch_start, fetch_end, "Nifty200")
        if bench_raw.empty:
            print("  ERROR: Could not fetch Nifty 200 benchmark — aborting")
            return
        print(f"  {len(bench_raw)} bars")

    bench_close = bench_raw["Close"]

    # Build date → iloc maps
    date_to_iloc = {}
    for ticker, df in stock_data.items():
        date_to_iloc[ticker] = {df.index[i].date(): i for i in range(len(df))}

    # All trading days
    day_counts = {}
    for df in stock_data.values():
        for d in df.index:
            dt = d.date()
            day_counts[dt] = day_counts.get(dt, 0) + 1
    trading_days = sorted(d for d, c in day_counts.items()
                          if c >= 50 and d >= START_DATE
                          and (end_date is None or d <= end_date))
    print(f"  Trading days: {len(trading_days)} ({trading_days[0]} → {trading_days[-1]})")

    indicators = build_indicators(stock_data, bench_close)

    # ── Portfolio state ───────────────────────────────────────────────────────
    cash          = float(CAPITAL)
    portfolio     = {}   # ticker → {entry_date, entry_price, shares, rsi_low_count}
    all_trades    = []
    total_charges = 0.0

    # Per-year tracking
    year_entries  = {}   # year → list of entry rows
    year_exits    = {}   # year → list of exit rows
    monthly_nav   = []   # month-end NAV snapshots for portfolio correlation analysis

    print()
    print("=" * 76)
    print("  RS63 PIT BACKTEST  |  ₹10L Capital  |  Max 7 Positions  |  NAV/7 Compounding")
    print("=" * 76)

    current_year = None

    for day_idx, day in enumerate(trading_days):
        yr = day.year
        pit_set = get_pit_universe(pit_data, day)

        # Print year header on year change
        if yr != current_year:
            if current_year is not None:
                _print_year_summary(current_year, year_entries, year_exits,
                                    portfolio, stock_data, date_to_iloc, cash, all_trades)
            current_year = yr
            year_entries[yr] = []
            year_exits[yr]   = []

        # ── EXIT PROCESSING ───────────────────────────────────────────────────
        for ticker in list(portfolio.keys()):
            pos    = portfolio[ticker]
            idx_map = date_to_iloc.get(ticker, {})
            ci     = idx_map.get(day)
            if ci is None:
                continue

            df     = stock_data[ticker]
            price  = float(df["Close"].iloc[ci])
            ind    = indicators[ticker]
            shares = pos["shares"]

            exited     = False
            exit_price = price
            exit_reason = ""

            # Exit 1: 8% hard stop loss
            if price <= pos["entry_price"] * 0.92:
                exit_reason = "HARD_SL_8%"
                exited = True

            # Exit 2: RSI(14, 3d avg) < 40 for 3 consecutive days
            if not exited:
                rsi_v = float(ind["rsi14_3d"].iloc[ci]) if not pd.isna(ind["rsi14_3d"].iloc[ci]) else 50
                if rsi_v < 40:
                    pos["rsi_low_count"] = pos.get("rsi_low_count", 0) + 1
                else:
                    pos["rsi_low_count"] = 0
                if pos["rsi_low_count"] >= 3:
                    exit_reason = "RSI<40_3D"
                    exited = True

            # Exit 3: RS63(5d avg) < 0
            if not exited:
                rs63_v = float(ind["rs63"].iloc[ci]) if not pd.isna(ind["rs63"].iloc[ci]) else 0
                if rs63_v < 0:
                    exit_reason = "RS63_NEG"
                    exited = True

            # Exit 4: Time stop — 8 weeks + <3% gain
            if not exited:
                days_held = (day - pos["entry_date"]).days
                gain_pct  = (price / pos["entry_price"] - 1) * 100
                if days_held >= 56 and gain_pct < 3.0:
                    exit_reason = "TIME_STOP"
                    exited = True

            if exited:
                gross_pnl = (exit_price - pos["entry_price"]) * shares
                buy_val   = pos["entry_price"] * shares
                sell_val  = exit_price * shares
                chg       = calc_charges(buy_val, sell_val)
                net_pnl   = gross_pnl - chg
                hold_days = (day - pos["entry_date"]).days
                cash     += sell_val
                total_charges += chg

                trade = {
                    "ticker": ticker, "entry_date": pos["entry_date"], "exit_date": day,
                    "entry_price": pos["entry_price"], "exit_price": exit_price,
                    "shares": shares, "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl, "charges": chg,
                    "hold_days": hold_days, "exit_reason": exit_reason,
                }
                all_trades.append(trade)
                year_exits.setdefault(yr, []).append(trade)
                del portfolio[ticker]

        # ── ENTRY SIGNALS ─────────────────────────────────────────────────────
        if len(portfolio) >= MAX_POSITIONS:
            continue

        # NAV/7 slot sizing — grows with profits
        nav = cash
        for t, pos in portfolio.items():
            idx_map = date_to_iloc.get(t, {})
            ci_t = idx_map.get(day)
            if ci_t is not None:
                nav += pos["shares"] * float(stock_data[t]["Close"].iloc[ci_t])
            else:
                nav += pos["shares"] * pos["entry_price"]
        per_slot = nav / MAX_POSITIONS

        signals = []
        for ticker, df in stock_data.items():
            if ticker in portfolio:
                continue
            if ticker not in pit_set:
                continue
            idx_map = date_to_iloc.get(ticker, {})
            ci = idx_map.get(day)
            if ci is None or ci < 210:
                continue

            ind   = indicators[ticker]
            price = float(df["Close"].iloc[ci])
            open_ = float(df["Open"].iloc[ci])
            high  = float(df["High"].iloc[ci])
            low   = float(df["Low"].iloc[ci])
            vol   = float(df["Volume"].iloc[ci])

            rs63_v = float(ind["rs63"].iloc[ci]) if not pd.isna(ind["rs63"].iloc[ci]) else -1
            rsi_v  = float(ind["rsi14_3d"].iloc[ci]) if not pd.isna(ind["rsi14_3d"].iloc[ci]) else 0
            va     = float(ind["vol_avg20"].iloc[ci]) if not pd.isna(ind["vol_avg20"].iloc[ci]) else 0
            l20d   = float(ind["low_20d"].iloc[ci]) if not pd.isna(ind["low_20d"].iloc[ci]) else 0

            if rs63_v <= 0:
                continue
            if rsi_v <= 50:
                continue

            day_range = high - low
            ibs = (price - low) / day_range if day_range > 0 else 0
            if ibs <= 0.5:
                continue
            if price <= open_:  # not a green candle
                continue

            stop_pct  = (price - l20d) / price * 100 if price > 0 and l20d > 0 else 8
            vol_ratio = vol / va if va > 0 else 0

            l5d  = float(ind["low_5d"].iloc[ci])  if not pd.isna(ind["low_5d"].iloc[ci])  else 0
            s20  = float(ind["sma20"].iloc[ci])   if not pd.isna(ind["sma20"].iloc[ci])   else price

            signals.append({
                "ticker":      ticker,
                "price":       price,
                "rs63":        rs63_v,
                "rsi":         rsi_v,
                "vol_ratio":   vol_ratio,
                "stop_pct":    stop_pct,                              # active ranking
                "stop_sma20":  (price - s20)  / price if price > 0 else 8,   # future use
                "low_rising":  (l5d - l20d)   / price if price > 0 and l5d > 0 and l20d > 0 else -1,  # future use
            })

        # Rank by tightest stop (close − 20d_low) / close ascending
        signals.sort(key=lambda s: s["stop_pct"])
        if max_daily_entries is not None:
            signals = signals[:max_daily_entries]
        for sig in signals:
            if len(portfolio) >= MAX_POSITIONS:
                break
            ticker = sig["ticker"]
            price  = sig["price"]
            shares = int(per_slot // price)
            if shares <= 0:
                continue
            cost = price * shares
            if cost > cash:
                continue
            chg   = calc_charges(cost, 0)
            cash -= (cost + chg)
            total_charges += chg
            portfolio[ticker] = {
                "entry_date":    day,
                "entry_price":   price,
                "shares":        shares,
                "rsi_low_count": 0,
            }
            year_entries.setdefault(yr, []).append({
                "ticker": ticker, "date": day,
                "price":  price,  "shares": shares,
                "rs63":   sig["rs63"], "rsi": sig["rsi"],
                "vol_ratio": sig["vol_ratio"],
            })

        # Month-end NAV snapshot
        next_day = trading_days[day_idx + 1] if day_idx + 1 < len(trading_days) else None
        if next_day is None or next_day.month != day.month:
            nav_snap = cash
            for t, pos in portfolio.items():
                ci_t = date_to_iloc.get(t, {}).get(day)
                if ci_t is not None:
                    nav_snap += pos["shares"] * float(stock_data[t]["Close"].iloc[ci_t])
                else:
                    nav_snap += pos["shares"] * pos["entry_price"]
            monthly_nav.append({"date": day, "nav": nav_snap})

    # Print final year
    if current_year is not None:
        _print_year_summary(current_year, year_entries, year_exits,
                            portfolio, stock_data, date_to_iloc, cash, all_trades)

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    last_day = trading_days[-1]
    final_value = cash
    for t, pos in portfolio.items():
        idx_map = date_to_iloc.get(t, {})
        last_ci = max(idx_map.values()) if idx_map else None
        lp = float(stock_data[t]["Close"].iloc[last_ci]) if last_ci else pos["entry_price"]
        final_value += lp * pos["shares"]

    years      = (trading_days[-1] - trading_days[0]).days / 365.25
    cagr       = ((final_value / CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else 0
    total_ret  = (final_value / CAPITAL - 1) * 100
    winners    = [t for t in all_trades if t["gross_pnl"] > 0]
    losers     = [t for t in all_trades if t["gross_pnl"] <= 0]
    avg_hold   = sum(t["hold_days"] for t in all_trades) / len(all_trades) if all_trades else 0
    avg_win    = sum(t["gross_pnl"] for t in winners) / len(winners) if winners else 0
    avg_loss   = sum(t["gross_pnl"] for t in losers) / len(losers) if losers else 0
    pf         = abs(sum(t["gross_pnl"] for t in winners) /
                     sum(t["gross_pnl"] for t in losers)) if losers else float("inf")

    reason_counts = {}
    for t in all_trades:
        r = t["exit_reason"]
        reason_counts[r] = reason_counts.get(r, 0) + 1

    print()
    print("=" * 76)
    print("  FINAL SUMMARY")
    print("=" * 76)
    print(f"  Period        : {trading_days[0]} → {trading_days[-1]}  ({years:.1f} years)")
    print(f"  Starting Cap  : {inr(CAPITAL)}")
    print(f"  Final Value   : {inr(final_value)}")
    print(f"  Total Return  : {pct(total_ret)}")
    print(f"  CAGR          : {pct(cagr)}")
    print()
    print(f"  Closed Trades : {len(all_trades)}  |  Open: {len(portfolio)}")
    print(f"  Win Rate      : {len(winners)/len(all_trades)*100:.1f}%  ({len(winners)}W / {len(losers)}L)")
    print(f"  Profit Factor : {pf:.2f}")
    print(f"  Avg Win       : {inr(avg_win)}  |  Avg Loss: {inr(avg_loss)}")
    print(f"  Avg Hold      : {avg_hold:.0f} days")
    print(f"  Total Charges : {inr(total_charges)}")
    print()
    print("  EXIT REASONS:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"    {reason:<20} {count:>4} trades")

    print()
    print("  YEAR-BY-YEAR (% of initial capital):")
    for yr in sorted(year_exits.keys()):
        pnl_y = sum(t["net_pnl"] for t in year_exits.get(yr, []))
        ret_y = pnl_y / CAPITAL * 100
        bar   = ("█" * min(int(abs(ret_y)), 50)) if ret_y > 0 else ("░" * min(int(abs(ret_y)), 50))
        sign  = "+" if ret_y >= 0 else "-"
        print(f"  {yr}  {sign}{abs(ret_y):5.1f}%  {bar}")
    print()

    # Export month-end NAV series for portfolio correlation analysis
    if monthly_nav:
        nav_csv = os.path.join(BASE_DIR, "rs63_monthly.csv")
        pd.DataFrame(monthly_nav).to_csv(nav_csv, index=False)
        print(f"  Monthly NAV exported → {os.path.basename(nav_csv)} ({len(monthly_nav)} rows)")


def _print_year_summary(yr, year_entries, year_exits, portfolio,
                        stock_data, date_to_iloc, cash, all_trades):
    entries = year_entries.get(yr, [])
    exits   = year_exits.get(yr, [])

    print()
    print("=" * 76)
    print(f"  YEAR {yr}")
    print("=" * 76)

    # Exits
    print(f"\n  EXITS ({len(exits)})")
    if exits:
        exit_rows = []
        for t in exits:
            pnl_pct = (t["exit_price"] / t["entry_price"] - 1) * 100
            exit_rows.append((
                t["ticker"],
                t["entry_date"].strftime("%d-%b-%y"),
                t["exit_date"].strftime("%d-%b-%y"),
                f"{t['entry_price']:,.1f}",
                f"{t['exit_price']:,.1f}",
                t["shares"],
                inr(t["gross_pnl"]),
                pct(pnl_pct),
                f"{t['hold_days']}d",
                t["exit_reason"],
            ))
        exit_rows.sort(key=lambda r: float(r[7].replace('+','').replace('%','')), reverse=True)
        print_table(
            ["Ticker","Entry","Exit","Entry₹","Exit₹","Qty","Gross P&L","P&L%","Hold","Reason"],
            exit_rows,
            [10, 10, 10, 10, 10, 5, 12, 8, 6, 14],
        )
    else:
        print("    —")

    # Entries
    print(f"\n  ENTRIES ({len(entries)})")
    if entries:
        entry_rows = [(
            e["ticker"],
            e["date"].strftime("%d-%b-%y"),
            f"{e['rs63']:+.1f}%",
            f"{e['rsi']:.1f}",
            f"{e['vol_ratio']:.2f}x",
            f"{e['price']:,.1f}",
            e["shares"],
            inr(e["price"] * e["shares"]),
        ) for e in entries]
        print_table(
            ["Ticker","Date","RS63","RSI","VolRatio","Entry₹","Qty","Capital"],
            entry_rows,
            [10, 10, 8, 6, 9, 10, 5, 12],
        )
    else:
        print("    —")

    # Year P&L
    closed_pnl   = sum(t["net_pnl"] for t in exits)
    yr_ret       = closed_pnl / CAPITAL * 100
    total_closed = sum(t["net_pnl"] for t in all_trades)
    print(f"\n  {yr} Net P&L: {inr(closed_pnl)} ({pct(yr_ret)} on initial capital)  |  "
          f"Cumulative: {inr(total_closed)}  |  Open: {len(portfolio)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RS63 PIT Backtest Report")
    parser.add_argument("--refresh", action="store_true", help="Re-download price data")
    parser.add_argument("--end-date", type=str, default=None,
                        help="Truncate backtest to this end date YYYY-MM-DD (for noise measurement)")
    parser.add_argument("--max-daily-entries", type=int, default=None,
                        help="Cap new entries per day (default: unlimited)")
    args = parser.parse_args()
    end_dt = date.fromisoformat(args.end_date) if args.end_date else None
    run(refresh=args.refresh, end_date=end_dt, max_daily_entries=args.max_daily_entries)
