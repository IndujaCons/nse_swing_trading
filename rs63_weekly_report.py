#!/usr/bin/env python3
"""
RS63 Weekly PIT Backtest — Weekly-Cadence Variant
==================================================
Capital  : ₹10L, max 7 positions, NAV/7 compounding
Universe : Nifty 200 PIT (historically correct constituents)
Entry    : Daily triggers (E1–E6) aggregated Mon–Fri
           → re-qualify Friday EOD → Z-score rank → execute Monday open
Exit     : X1 (8% hard SL) daily intraday
           X2 (RS63<0), X3 (close<SMA63), X4 (RSI<40 3d), X5 (time stop)
           checked Friday EOD → executed Monday open

Usage:
    python3 rs63_weekly_report.py
    python3 rs63_weekly_report.py > rs63_weekly_result.md
"""

import os, sys, json, pickle, warnings, argparse, time
from datetime import date, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
START_DATE    = date(2015, 4, 9)
CAPITAL       = 10_00_000
MAX_POSITIONS = 7
WARMUP_DAYS   = 450

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, 'data', 'cache', 'rs63_weekly_report.pkl')
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
    if bench_close.index.tzinfo is not None:
        bench_close = bench_close.copy()
        bench_close.index = bench_close.index.tz_localize(None)
    indicators = {}
    for ticker, df in stock_data.items():
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
        rs63      = rs63_raw.rolling(5, min_periods=1).mean()

        rsi14_raw  = compute_rsi(closes, 14)
        rsi14_3d   = rsi14_raw.rolling(3, min_periods=1).mean()

        sma63      = closes.rolling(63, min_periods=1).mean()
        vol_avg20  = volumes.rolling(20).mean()

        indicators[ticker] = {
            "rs63":      rs63,
            "rsi14_3d":  rsi14_3d,
            "vol_avg20": vol_avg20,
            "sma63":     sma63,
        }
    print(f"  Indicators built for {len(indicators)} stocks")
    return indicators

# ── WEEKLY CADENCE HELPERS ────────────────────────────────────────────────────
def is_last_trading_day_of_week(day, trading_days, day_idx):
    if day_idx + 1 >= len(trading_days):
        return True
    # different ISO week → this is the last day of current week
    cur_iso = day.isocalendar()[:2]
    nxt_iso = trading_days[day_idx + 1].isocalendar()[:2]
    return cur_iso != nxt_iso

def is_first_trading_day_of_week(day, trading_days, day_idx):
    if day_idx == 0:
        return True
    cur_iso = day.isocalendar()[:2]
    prv_iso = trading_days[day_idx - 1].isocalendar()[:2]
    return cur_iso != prv_iso

# ── Z-SCORE RANKING ───────────────────────────────────────────────────────────
def zscore_rank(candidates):
    if len(candidates) < 3:
        candidates.sort(key=lambda s: s["rs63"], reverse=True)
        for s in candidates:
            s["score"] = s["rs63"]
        return candidates
    rs_vals  = [s["rs63"] for s in candidates]
    rsi_vals = [s["rsi"]  for s in candidates]
    mu_rs,  sd_rs  = np.mean(rs_vals),  np.std(rs_vals)
    mu_rsi, sd_rsi = np.mean(rsi_vals), np.std(rsi_vals)
    if sd_rs == 0 or sd_rsi == 0:
        candidates.sort(key=lambda s: s["rs63"], reverse=True)
        for s in candidates:
            s["score"] = s["rs63"]
        return candidates
    for s in candidates:
        z_rs  = (s["rs63"] - mu_rs)  / sd_rs
        z_rsi = (s["rsi"]  - mu_rsi) / sd_rsi
        s["score"] = 0.5 * z_rs + 0.5 * z_rsi
    candidates.sort(key=lambda s: s["score"], reverse=True)
    return candidates

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
def run(refresh=False):
    print("Loading PIT universe...")
    pit_data    = load_pit()
    all_tickers = get_all_pit_tickers(pit_data)
    print(f"  {len(all_tickers)} unique PIT tickers")

    fetch_start = START_DATE - timedelta(days=WARMUP_DAYS)
    fetch_end   = date.today()

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

    date_to_iloc = {}
    for ticker, df in stock_data.items():
        date_to_iloc[ticker] = {df.index[i].date(): i for i in range(len(df))}

    day_counts = {}
    for df in stock_data.values():
        for d in df.index:
            dt = d.date()
            day_counts[dt] = day_counts.get(dt, 0) + 1
    trading_days = sorted(d for d, c in day_counts.items() if c >= 50 and d >= START_DATE)
    print(f"  Trading days: {len(trading_days)} ({trading_days[0]} → {trading_days[-1]})")

    indicators = build_indicators(stock_data, bench_close)

    # ── Portfolio state ───────────────────────────────────────────────────────
    cash          = float(CAPITAL)
    portfolio     = {}
    all_trades    = []
    total_charges = 0.0

    year_entries  = {}
    year_exits    = {}
    monthly_nav   = []

    # Weekly state
    weekly_log     = {}   # (iso_year, iso_week) → {ticker: first_weekday (0–4)}
    candidate_pool = []   # built Friday EOD, consumed Monday open
    pending_exits  = {}   # ticker → reason, set Friday EOD, executed Monday open

    # Diagnostics
    weekly_pool_sizes  = []   # pool size each Friday
    weekly_slots_used  = []   # entries executed each Monday
    weekly_pos_count   = []   # portfolio size after Monday entries (for true utilization)
    trigger_day_pnl    = {0: [], 1: [], 2: [], 3: [], 4: []}  # pnl_pct per trigger day
    deferred_exit_info = []   # {reason, friday_date, monday_date}
    trade_ranks        = {}   # ticker+entry_date → rank in candidate_pool (1-based)

    print()
    print("=" * 76)
    print("  RS63 WEEKLY PIT BACKTEST  |  ₹10L Capital  |  Max 7 Positions  |  NAV/7 Compounding")
    print("=" * 76)

    current_year = None

    for day_idx, day in enumerate(trading_days):
        yr      = day.year
        pit_set = get_pit_universe(pit_data, day)

        if yr != current_year:
            if current_year is not None:
                _print_year_summary(current_year, year_entries, year_exits,
                                    portfolio, stock_data, date_to_iloc, cash, all_trades)
            current_year = yr
            year_entries[yr] = []
            year_exits[yr]   = []

        # ── STEP 1: X1 daily hard stop ────────────────────────────────────────
        for ticker in list(portfolio.keys()):
            pos     = portfolio[ticker]
            idx_map = date_to_iloc.get(ticker, {})
            ci      = idx_map.get(day)
            if ci is None:
                continue
            df  = stock_data[ticker]
            low = float(df["Low"].iloc[ci])
            if low <= pos["entry_price"] * 0.92:
                exit_price = pos["entry_price"] * 0.92
                shares     = pos["shares"]
                buy_val    = pos["entry_price"] * shares
                sell_val   = exit_price * shares
                chg        = calc_charges(buy_val, sell_val)
                gross_pnl  = (exit_price - pos["entry_price"]) * shares
                net_pnl    = gross_pnl - chg
                hold_days  = (day - pos["entry_date"]).days
                cash      += sell_val
                total_charges += chg
                trade = {
                    "ticker": ticker, "entry_date": pos["entry_date"], "exit_date": day,
                    "entry_price": pos["entry_price"], "exit_price": exit_price,
                    "shares": shares, "gross_pnl": gross_pnl, "net_pnl": net_pnl,
                    "charges": chg, "hold_days": hold_days, "exit_reason": "HARD_SL_8%",
                    "trigger_day": pos.get("trigger_day", -1),
                    "entry_rank": pos.get("entry_rank", -1),
                }
                all_trades.append(trade)
                year_exits.setdefault(yr, []).append(trade)
                # record for trigger-day analysis
                pnl_pct = (exit_price / pos["entry_price"] - 1) * 100
                td = pos.get("trigger_day", -1)
                if 0 <= td <= 4:
                    trigger_day_pnl[td].append(pnl_pct)
                # remove from pending_exits if marked there too
                pending_exits.pop(ticker, None)
                del portfolio[ticker]

        # ── STEP 2: Accumulate RSI low count daily (for X4) ───────────────────
        for ticker, pos in portfolio.items():
            ci = date_to_iloc.get(ticker, {}).get(day)
            if ci is None:
                continue
            ind   = indicators[ticker]
            rsi_v = float(ind["rsi14_3d"].iloc[ci]) if not pd.isna(ind["rsi14_3d"].iloc[ci]) else 50
            if rsi_v < 40:
                pos["rsi_low_count"] = pos.get("rsi_low_count", 0) + 1
            else:
                pos["rsi_low_count"] = 0

        # ── STEP 3: Log daily signal triggers ─────────────────────────────────
        week_key = (day.isocalendar()[0], day.isocalendar()[1])
        weekday  = day.weekday()

        for ticker, df in stock_data.items():
            if ticker in portfolio or ticker not in pit_set:
                continue
            idx_map = date_to_iloc.get(ticker, {})
            ci = idx_map.get(day)
            if ci is None or ci < 210:
                continue
            ind    = indicators[ticker]
            price  = float(df["Close"].iloc[ci])
            open_  = float(df["Open"].iloc[ci])
            high   = float(df["High"].iloc[ci])
            low    = float(df["Low"].iloc[ci])
            vol    = float(df["Volume"].iloc[ci])

            rs63_v = float(ind["rs63"].iloc[ci])     if not pd.isna(ind["rs63"].iloc[ci])     else -1
            rsi_v  = float(ind["rsi14_3d"].iloc[ci]) if not pd.isna(ind["rsi14_3d"].iloc[ci]) else 0
            va     = float(ind["vol_avg20"].iloc[ci]) if not pd.isna(ind["vol_avg20"].iloc[ci]) else 0
            dr     = high - low
            ibs    = (price - low) / dr if dr > 0 else 0
            vol_ratio = vol / va if va > 0 else 0

            if (rs63_v > 0 and rsi_v > 50 and ibs > 0.5
                    and price > open_ and vol_ratio >= 1.0):
                if week_key not in weekly_log:
                    weekly_log[week_key] = {}
                if ticker not in weekly_log[week_key]:
                    weekly_log[week_key][ticker] = weekday

        # ── STEP 4: Friday EOD — X2–X5 exits + build candidate pool ──────────
        if is_last_trading_day_of_week(day, trading_days, day_idx):
            pending_exits = {}

            for ticker in list(portfolio.keys()):
                pos = portfolio[ticker]
                ci  = date_to_iloc.get(ticker, {}).get(day)
                if ci is None:
                    continue
                ind   = indicators[ticker]
                price = float(stock_data[ticker]["Close"].iloc[ci])

                rs63_v = float(ind["rs63"].iloc[ci]) if not pd.isna(ind["rs63"].iloc[ci]) else 0

                if rs63_v < 0:
                    pending_exits[ticker] = "RS63_NEG"
                    deferred_exit_info.append({"reason": "RS63_NEG", "friday": day})
                    continue
                if pos.get("rsi_low_count", 0) >= 3:
                    pending_exits[ticker] = "RSI<40_3D"
                    deferred_exit_info.append({"reason": "RSI<40_3D", "friday": day})
                    continue
                days_held = (day - pos["entry_date"]).days
                gain_pct  = (price / pos["entry_price"] - 1) * 100
                if days_held >= 56 and gain_pct < 3.0:
                    pending_exits[ticker] = "TIME_STOP"
                    deferred_exit_info.append({"reason": "TIME_STOP", "friday": day})

            # Build & rank candidate pool
            candidates = []
            for ticker, trigger_day in weekly_log.get(week_key, {}).items():
                if ticker in portfolio:
                    continue
                ci = date_to_iloc.get(ticker, {}).get(day)
                if ci is None:
                    continue
                ind    = indicators[ticker]
                price  = float(stock_data[ticker]["Close"].iloc[ci])
                open_  = float(stock_data[ticker]["Open"].iloc[ci])
                high   = float(stock_data[ticker]["High"].iloc[ci])
                low    = float(stock_data[ticker]["Low"].iloc[ci])
                vol    = float(stock_data[ticker]["Volume"].iloc[ci])
                rs63_v = float(ind["rs63"].iloc[ci])     if not pd.isna(ind["rs63"].iloc[ci])     else -1
                rsi_v  = float(ind["rsi14_3d"].iloc[ci]) if not pd.isna(ind["rsi14_3d"].iloc[ci]) else 0
                va     = float(ind["vol_avg20"].iloc[ci]) if not pd.isna(ind["vol_avg20"].iloc[ci]) else 0
                dr     = high - low
                ibs    = (price - low) / dr if dr > 0 else 0
                vol_ratio = vol / va if va > 0 else 0
                if (rs63_v > 0 and rsi_v > 50 and ibs > 0.5
                        and price > open_ and vol_ratio >= 1.0):
                    candidates.append({
                        "ticker":      ticker,
                        "rs63":        rs63_v,
                        "rsi":         rsi_v,
                        "friday_close": price,
                        "trigger_day": trigger_day,
                    })

            candidate_pool = zscore_rank(candidates)
            weekly_pool_sizes.append(len(candidate_pool))

        # ── STEP 5: Monday open — execute pending exits then entries ──────────
        if is_first_trading_day_of_week(day, trading_days, day_idx):
            entries_this_week = 0

            # Execute deferred X2–X5 exits
            for ticker, reason in list(pending_exits.items()):
                if ticker not in portfolio:
                    continue
                ci = date_to_iloc.get(ticker, {}).get(day)
                if ci is None:
                    continue
                pos        = portfolio[ticker]
                exit_price = float(stock_data[ticker]["Open"].iloc[ci])
                shares     = pos["shares"]
                buy_val    = pos["entry_price"] * shares
                sell_val   = exit_price * shares
                chg        = calc_charges(buy_val, sell_val)
                gross_pnl  = (exit_price - pos["entry_price"]) * shares
                net_pnl    = gross_pnl - chg
                hold_days  = (day - pos["entry_date"]).days
                cash      += sell_val
                total_charges += chg
                trade = {
                    "ticker": ticker, "entry_date": pos["entry_date"], "exit_date": day,
                    "entry_price": pos["entry_price"], "exit_price": exit_price,
                    "shares": shares, "gross_pnl": gross_pnl, "net_pnl": net_pnl,
                    "charges": chg, "hold_days": hold_days, "exit_reason": reason,
                    "trigger_day": pos.get("trigger_day", -1),
                    "entry_rank": pos.get("entry_rank", -1),
                }
                all_trades.append(trade)
                year_exits.setdefault(yr, []).append(trade)
                pnl_pct = (exit_price / pos["entry_price"] - 1) * 100
                td = pos.get("trigger_day", -1)
                if 0 <= td <= 4:
                    trigger_day_pnl[td].append(pnl_pct)
                # stamp Monday date on the deferred_exit_info entry
                for info in reversed(deferred_exit_info):
                    if info.get("reason") == reason and "monday" not in info:
                        info["monday"] = day
                        break
                del portfolio[ticker]

            # Recompute NAV/7 after exits
            nav = cash
            for t, pos in portfolio.items():
                ci_t = date_to_iloc.get(t, {}).get(day)
                if ci_t is not None:
                    nav += pos["shares"] * float(stock_data[t]["Close"].iloc[ci_t])
                else:
                    nav += pos["shares"] * pos["entry_price"]
            per_slot   = nav / MAX_POSITIONS
            open_slots = MAX_POSITIONS - len(portfolio)

            for rank_idx, candidate in enumerate(candidate_pool[:open_slots]):
                if len(portfolio) >= MAX_POSITIONS:
                    break
                ticker = candidate["ticker"]
                if ticker in portfolio:
                    continue
                ci = date_to_iloc.get(ticker, {}).get(day)
                if ci is None:
                    continue
                entry_price = float(stock_data[ticker]["Open"].iloc[ci])
                shares = int(per_slot // entry_price)
                if shares <= 0:
                    continue
                cost = shares * entry_price
                chg  = calc_charges(cost, 0)
                if cost + chg > cash:
                    break
                cash -= (cost + chg)
                total_charges += chg
                entry_rank = rank_idx + 1  # 1-based
                portfolio[ticker] = {
                    "entry_date":    day,
                    "entry_price":   entry_price,
                    "shares":        shares,
                    "rsi_low_count": 0,
                    "trigger_day":   candidate["trigger_day"],
                    "entry_rank":    entry_rank,
                }
                trade_ranks[f"{ticker}_{day}"] = entry_rank
                year_entries.setdefault(yr, []).append({
                    "ticker":      ticker,
                    "date":        day,
                    "price":       entry_price,
                    "shares":      shares,
                    "rs63":        candidate["rs63"],
                    "rsi":         candidate["rsi"],
                    "score":       candidate.get("score", candidate["rs63"]),
                    "trigger_day": candidate["trigger_day"],
                    "rank":        entry_rank,
                })
                entries_this_week += 1

            weekly_slots_used.append(entries_this_week)
            weekly_pos_count.append(len(portfolio))
            candidate_pool = []

        # ── STEP 6: Month-end NAV snapshot ────────────────────────────────────
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

    if current_year is not None:
        _print_year_summary(current_year, year_entries, year_exits,
                            portfolio, stock_data, date_to_iloc, cash, all_trades)

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    last_day    = trading_days[-1]
    final_value = cash
    for t, pos in portfolio.items():
        idx_map = date_to_iloc.get(t, {})
        last_ci = max(idx_map.values()) if idx_map else None
        lp = float(stock_data[t]["Close"].iloc[last_ci]) if last_ci else pos["entry_price"]
        final_value += lp * pos["shares"]

    years     = (trading_days[-1] - trading_days[0]).days / 365.25
    cagr      = ((final_value / CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else 0
    total_ret = (final_value / CAPITAL - 1) * 100
    winners   = [t for t in all_trades if t["gross_pnl"] > 0]
    losers    = [t for t in all_trades if t["gross_pnl"] <= 0]
    avg_hold  = sum(t["hold_days"] for t in all_trades) / len(all_trades) if all_trades else 0
    avg_win   = sum(t["gross_pnl"] for t in winners) / len(winners) if winners else 0
    avg_loss  = sum(t["gross_pnl"] for t in losers)  / len(losers)  if losers  else 0
    pf        = abs(sum(t["gross_pnl"] for t in winners) /
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
    print(f"  Win Rate      : {len(winners)/len(all_trades)*100:.1f}%  ({len(winners)}W / {len(losers)}L)" if all_trades else "  No closed trades")
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

    # ── WEEKLY DIAGNOSTICS ────────────────────────────────────────────────────
    total_weeks = len(weekly_pool_sizes)
    dead_weeks  = sum(1 for s in weekly_pool_sizes if s == 0)
    hot_weeks   = sum(1 for s in weekly_pool_sizes if s > 20)
    avg_pool    = np.mean(weekly_pool_sizes) if weekly_pool_sizes else 0

    # True slot utilization: avg(open_positions / MAX_POSITIONS) across all Monday snapshots
    avg_utilization = (np.mean(weekly_pos_count) / MAX_POSITIONS * 100
                       if weekly_pos_count else 0)

    print()
    print("=" * 76)
    print("  WEEKLY DIAGNOSTICS")
    print("=" * 76)
    print(f"  Total weeks            : {total_weeks}")
    if total_weeks:
        print(f"  Dead weeks (0 cand)    : {dead_weeks}  ({dead_weeks/total_weeks*100:.1f}%)")
    print(f"  Weeks >20 candidates   : {hot_weeks}")
    print(f"  Avg candidates/week    : {avg_pool:.1f}")
    print(f"  Avg slot utilization   : {avg_utilization:.1f}%  "
          f"(avg open positions after Monday = {np.mean(weekly_pos_count):.1f} / {MAX_POSITIONS})"
          if weekly_pos_count else "")

    # Slot fill distribution
    print()
    print("  SLOT FILL DISTRIBUTION (open positions after Monday entries):")
    print(f"  {'Slots':>5}  {'Weeks':>6}  {'Pct':>6}")
    print(f"  {'─'*5}  {'─'*6}  {'─'*6}")
    for n in range(MAX_POSITIONS + 1):
        cnt = sum(1 for x in weekly_pos_count if x == n)
        if cnt > 0 or n <= 2:
            pct_w = cnt / len(weekly_pos_count) * 100 if weekly_pos_count else 0
            print(f"  {n:>5}  {cnt:>6}  {pct_w:>5.1f}%")

    print()
    print("  TRIGGER-DAY ANALYSIS:")
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    print(f"  {'Day':<5}  {'Count':>5}  {'Win%':>6}  {'AvgP&L%':>8}")
    print(f"  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*8}")
    for d in range(5):
        pnls = trigger_day_pnl[d]
        if not pnls:
            print(f"  {day_names[d]:<5}  {'0':>5}  {'—':>6}  {'—':>8}")
            continue
        wins  = sum(1 for p in pnls if p > 0)
        wr    = wins / len(pnls) * 100
        avg_p = np.mean(pnls)
        print(f"  {day_names[d]:<5}  {len(pnls):>5}  {wr:>5.1f}%  {avg_p:>+7.1f}%")

    # Win rate by exit reason
    print()
    print("  WIN RATE BY EXIT REASON:")
    print(f"  {'Reason':<20}  {'Count':>5}  {'Win%':>6}  {'AvgP&L%':>8}")
    print(f"  {'─'*20}  {'─'*5}  {'─'*6}  {'─'*8}")
    for reason in sorted(reason_counts.keys(), key=lambda r: -reason_counts[r]):
        grp   = [t for t in all_trades if t["exit_reason"] == reason]
        wins  = sum(1 for t in grp if t["gross_pnl"] > 0)
        wr    = wins / len(grp) * 100 if grp else 0
        pnls  = [(t["exit_price"] / t["entry_price"] - 1) * 100 for t in grp]
        avg_p = np.mean(pnls) if pnls else 0
        print(f"  {reason:<20}  {len(grp):>5}  {wr:>5.1f}%  {avg_p:>+7.1f}%")

    # Avg rank of winners vs losers
    ranked_winners = [t for t in all_trades if t["gross_pnl"] > 0  and t["entry_rank"] > 0]
    ranked_losers  = [t for t in all_trades if t["gross_pnl"] <= 0 and t["entry_rank"] > 0]
    print()
    print("  Z-SCORE RANK QUALITY (winners vs losers):")
    if ranked_winners or ranked_losers:
        wr_rank = np.mean([t["entry_rank"] for t in ranked_winners]) if ranked_winners else 0
        lr_rank = np.mean([t["entry_rank"] for t in ranked_losers])  if ranked_losers  else 0
        print(f"  Avg rank of winners : {wr_rank:.2f}  (lower = ranked higher by Z-score)")
        print(f"  Avg rank of losers  : {lr_rank:.2f}")
        print()
        print("  Win rate by entry rank:")
        print(f"  {'Rank':>5}  {'Count':>5}  {'Win%':>6}")
        print(f"  {'─'*5}  {'─'*5}  {'─'*6}")
        for r in range(1, MAX_POSITIONS + 1):
            grp  = [t for t in all_trades if t["entry_rank"] == r]
            if not grp:
                continue
            w    = sum(1 for t in grp if t["gross_pnl"] > 0)
            wr_r = w / len(grp) * 100
            print(f"  {r:>5}  {len(grp):>5}  {wr_r:>5.1f}%")

    # Exit timing for deferred exits
    deferred_reasons = ["RS63_NEG", "RSI<40_3D", "TIME_STOP"]
    print()
    print("  EXIT TIMING (X2/X4/X5: Friday signal → Monday execution):")
    print(f"  {'Reason':<15}  {'Count':>5}  {'AvgDelay':>9}")
    print(f"  {'─'*15}  {'─'*5}  {'─'*9}")
    for reason in deferred_reasons:
        infos = [i for i in deferred_exit_info
                 if i["reason"] == reason and "monday" in i]
        if not infos:
            cnt = sum(1 for i in deferred_exit_info if i["reason"] == reason)
            print(f"  {reason:<15}  {cnt:>5}  {'—':>9}")
            continue
        delays = [(i["monday"] - i["friday"]).days for i in infos]
        print(f"  {reason:<15}  {len(infos):>5}  {np.mean(delays):>8.1f}d")

    print()

    # Export monthly NAV
    if monthly_nav:
        nav_csv = os.path.join(BASE_DIR, "rs63w_monthly.csv")
        pd.DataFrame(monthly_nav).to_csv(nav_csv, index=False)
        print(f"  Monthly NAV exported → {os.path.basename(nav_csv)} ({len(monthly_nav)} rows)")

    print()
    print("=" * 76)
    print("  PASS/FAIL vs DAILY BASELINE (CAGR 24.0%, PF 3.08)")
    print("=" * 76)
    verdict = ("SHIP IT  — Operational simplicity worth small CAGR cost"
               if cagr >= 22.0 and pf >= 2.80
               else "BORDERLINE — Dig into diagnostics before deciding"
               if cagr >= 18.0 and pf >= 2.30
               else "REJECT   — Weekly cadence breaks strategy too much")
    print(f"  CAGR {pct(cagr)}  |  PF {pf:.2f}  →  {verdict}")
    secondary = []
    if avg_utilization >= 70:
        secondary.append(f"Avg slot utilization {avg_utilization:.1f}% ≥ 70% ✓")
    else:
        secondary.append(f"Avg slot utilization {avg_utilization:.1f}% < 70% ✗")
    if dead_weeks < 30:
        secondary.append(f"Dead weeks {dead_weeks} < 30 ✓")
    else:
        secondary.append(f"Dead weeks {dead_weeks} ≥ 30 ✗")
    worst_yr = min(
        (sum(t["net_pnl"] for t in year_exits.get(yr, [])) / CAPITAL * 100
         for yr in year_exits),
        default=0
    )
    if worst_yr >= -18:
        secondary.append(f"Worst year {pct(worst_yr)} ≥ −18% ✓")
    else:
        secondary.append(f"Worst year {pct(worst_yr)} < −18% ✗")
    for s in secondary:
        print(f"  {s}")
    print()


def _print_year_summary(yr, year_entries, year_exits, portfolio,
                        stock_data, date_to_iloc, cash, all_trades):
    entries = year_entries.get(yr, [])
    exits   = year_exits.get(yr, [])

    print()
    print("=" * 76)
    print(f"  YEAR {yr}")
    print("=" * 76)

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

    print(f"\n  ENTRIES ({len(entries)})")
    if entries:
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        entry_rows = [(
            e["ticker"],
            e["date"].strftime("%d-%b-%y"),
            f"{e['rs63']:+.1f}%",
            f"{e['rsi']:.1f}",
            f"{e.get('score', e['rs63']):+.2f}",
            day_names[e["trigger_day"]] if 0 <= e["trigger_day"] <= 4 else "?",
            f"{e['price']:,.1f}",
            e["shares"],
            inr(e["price"] * e["shares"]),
        ) for e in entries]
        print_table(
            ["Ticker","Date","RS63","RSI","ZScore","TrigDay","Entry₹","Qty","Capital"],
            entry_rows,
            [10, 10, 8, 6, 8, 7, 10, 5, 12],
        )
    else:
        print("    —")

    closed_pnl = sum(t["net_pnl"] for t in exits)
    yr_ret     = closed_pnl / CAPITAL * 100
    total_closed = sum(t["net_pnl"] for t in all_trades)
    print(f"\n  {yr} Net P&L: {inr(closed_pnl)} ({pct(yr_ret)} on initial capital)  |  "
          f"Cumulative: {inr(total_closed)}  |  Open: {len(portfolio)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RS63 Weekly PIT Backtest Report")
    parser.add_argument("--refresh", action="store_true", help="Re-download price data")
    args = parser.parse_args()
    run(refresh=args.refresh)
