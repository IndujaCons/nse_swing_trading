#!/usr/bin/env python3
"""
Mom15 PIT Backtest — 2-Monthly Rebalance Report
================================================
Capital : ₹20L starting, ₹2L per slot (compounding)
Slots   : 10 initially → unlocks 1 slot per ₹2L profit, max 15
Rebalance: First trading day of Feb, Apr, Jun, Aug, Oct, Dec each year
Scoring : MR_12 (50%) + MR_3 (50%), Z-scored, Normalised Score
Filters : Beta ≤ 1.0 vs Nifty 50, TTM EPS growth > 0%
Buffer  : Existing stays if rank ≤ 30, new enters if rank ≤ 10

Usage:
    python3 mom15_pit_report.py           # first run downloads data (~5-10 min)
    python3 mom15_pit_report.py --refresh # force re-download
"""

import os, sys, json, pickle, warnings, argparse
from datetime import date, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
START_DATE   = date(2015, 1, 1)
MAX_SLOTS    = 15             # Mom15 — always 15 positions
BUFFER_IN    = 10             # new stock enters if rank ≤ 10
BUFFER_OUT   = 30             # existing stays if rank ≤ 30
BETA_CAP     = 1.0
W12, W3      = 0.50, 0.50    # 12m + 3m, 6m dropped
LONG_PD      = 252            # 12m in trading days
SHORT_PD     = 63             # 3m in trading days
WARMUP_DAYS  = 450            # extra history before START_DATE for warmup

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE   = os.path.join(BASE_DIR, 'data', 'cache', 'mom15_daily.pkl')
PIT_FILE     = os.path.join(BASE_DIR, 'nse_const', 'nifty200_pit.json')
EPS_FILE     = os.path.join(BASE_DIR, 'data', 'quarterly_eps.json')

# Zerodha delivery charges
def calc_charges(buy_val, sell_val):
    total  = buy_val + sell_val
    stt    = 0.001    * total
    exch   = 0.0000307 * total
    sebi   = 0.000001  * total
    stamp  = 0.00015   * buy_val
    gst    = 0.18      * (exch + sebi)
    return stt + exch + sebi + stamp + gst

# STCG tax (20%) on net profit after deductible charges
def calc_tax(gross_pnl, charges):
    # STT not deductible; exchange+sebi+stamp+gst are
    deductible = charges * (1 - 0.001 / (0.001 + 0.0000307 + 0.000001 + 0.00015))
    taxable = gross_pnl - deductible
    return max(0, taxable * 0.20)

# ── TICKER ALIASES ────────────────────────────────────────────────────────────
ALIASES = {
    "ETERNAL":    "ZOMATO.NS",
    "ZOMATO":     "ETERNAL.NS",
    "ALBK":       "INDIANB.NS",
    "ANDHRABANK": "UNIONBANK.NS",
    "CROMPGREAV": "CGPOWER.NS",
    "GMRINFRA":   "GMRAIRPORT.NS",
    "HDFC":       "HDFCBANK.NS",
    "IDFC":       "IDFCFIRSTB.NS",
    "IDFCBANK":   "IDFCFIRSTB.NS",
    "JUBILANT":   "JUBLPHARMA.NS",
    "KPIT":       "BSOFT.NS",
    "LTI":        "LTIM.NS",
    "LTM":        "LTIM.NS",
    "MINDTREE":   "LTIM.NS",
    "MAX":        "MFSL.NS",
    "NIITTECH":   "COFORGE.NS",
    "ORIENTBANK": "PNB.NS",
    "RNAM":       "NAM-INDIA.NS",
    "STRTECH":    "STLTECH.NS",
    "SYNDIBANK":  "CANBK.NS",
    "WELSPUNIND": "WELSPUNLIV.NS",
    "ALSTOMT&D":  "532309.BO",
    "AMARAJABAT": "500008.BO",
    "DHFL":       "511072.BO",
    "GSKCONS":    "500676.BO",
    "IBULHSGFIN": "535789.BO",
    "MCDOWELL-N": "UNITDSPR.BO",
    "PEL":        "500302.BO",
    "TMPV":       "TMPV.BO",
}

# ── PIT UNIVERSE ──────────────────────────────────────────────────────────────
def load_pit():
    with open(PIT_FILE) as f:
        raw = json.load(f)
    # raw = {date_str: [tickers], ...}
    parsed = sorted(
        [(pd.Timestamp(k).date(), set(v)) for k, v in raw.items()],
        key=lambda x: x[0]
    )
    return parsed

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

# ── EPS FILTER ────────────────────────────────────────────────────────────────
MON_MAP = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
           "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}

def eps_passes(eps_db, ticker, day):
    """Return True if stock passes TTM EPS growth > 0%, or no data available."""
    if ticker not in eps_db:
        return True
    annual = eps_db[ticker].get("annual", {})
    valid = {}
    for ps, val in annual.items():
        try:
            parts = ps.split()
            m, y = MON_MAP.get(parts[0], 0), int(parts[1])
            if m == 0:
                continue
            avail = date(y, min(m+2,12), 28) if m <= 10 else date(y+1, m-10, 28)
            if avail <= day:
                valid[ps] = val
        except Exception:
            continue
    if len(valid) < 2:
        return True  # insufficient data → pass through
    sorted_ps = sorted(valid.keys(), key=lambda p: (int(p.split()[1]), MON_MAP.get(p.split()[0],0)))
    latest, prev = valid[sorted_ps[-1]], valid[sorted_ps[-2]]
    if abs(prev) < 0.01:
        return True
    return (latest / prev - 1) > 0.0

# ── DATA LOADING ──────────────────────────────────────────────────────────────
def fetch_ticker(ticker, start, end):
    sym = ALIASES.get(ticker, f"{ticker}.NS")
    try:
        df = yf.Ticker(sym).history(start=start, end=end)
        if df.empty or len(df) < 200:
            # Try .NS if alias was .BO
            if ".BO" in sym:
                return pd.DataFrame()
            df2 = yf.Ticker(f"{ticker}.NS").history(start=start, end=end)
            if len(df2) > len(df):
                df = df2
    except Exception:
        df = pd.DataFrame()
    if not df.empty:
        df.index = df.index.tz_localize(None) if df.index.tzinfo else df.index
    return df

def load_or_fetch_data(tickers, fetch_start, fetch_end, refresh=False):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    if not refresh and os.path.exists(CACHE_FILE):
        print(f"Loading cached data from {CACHE_FILE}...")
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)

    print(f"Fetching {len(tickers)} tickers from yfinance (this takes ~5-10 min)...")
    stock_data = {}
    for i, ticker in enumerate(sorted(tickers)):
        df = fetch_ticker(ticker, fetch_start, fetch_end)
        if not df.empty and len(df) >= 200:
            stock_data[ticker] = df
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(tickers)} done, {len(stock_data)} loaded...")
    print(f"  Done. {len(stock_data)} stocks with sufficient history.")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(stock_data, f)
    print(f"  Cached to {CACHE_FILE}")
    return stock_data

# ── SCORING ───────────────────────────────────────────────────────────────────
def compute_scores(day, stock_data, date_to_iloc, pit_data, nifty50_data,
                   n50_iloc, eps_db):
    pit_set = get_pit_universe(pit_data, day)
    raw = {}

    for ticker, df in stock_data.items():
        if ticker not in pit_set:
            continue
        idx_map = date_to_iloc.get(ticker, {})
        ci = idx_map.get(day)
        if ci is None:
            for off in range(1, 6):
                prev = day - timedelta(days=off)
                if prev in idx_map:
                    ci = idx_map[prev]
                    break
        if ci is None or ci < LONG_PD + 20:
            continue

        closes = df["Close"].values.astype(float)
        p_now   = closes[ci]
        p_12m   = closes[ci - LONG_PD]
        p_3m    = closes[ci - SHORT_PD]
        if p_now <= 0 or p_12m <= 0 or p_3m <= 0:
            continue

        ret_12 = p_now / p_12m - 1
        ret_3  = p_now / p_3m  - 1
        log_r  = np.diff(np.log(np.maximum(closes[ci-LONG_PD:ci+1], 0.01)))
        sigma  = float(np.std(log_r)) * np.sqrt(252)
        if sigma < 0.01:
            continue

        mr_12 = ret_12 / sigma
        mr_3  = ret_3  / sigma

        # Beta vs Nifty 50
        beta = None
        n50_ci = n50_iloc.get(day)
        if n50_ci is None:
            for off in range(1, 6):
                prev = day - timedelta(days=off)
                if prev in n50_iloc:
                    n50_ci = n50_iloc[prev]
                    break
        if n50_ci is not None and n50_ci >= 252:
            n50_c  = nifty50_data["Close"].values[n50_ci-252:n50_ci+1].astype(float)
            stk_c  = closes[ci-252:ci+1]
            n50_r  = np.diff(n50_c) / np.maximum(n50_c[:-1], 0.01)
            stk_r  = np.diff(stk_c) / np.maximum(stk_c[:-1], 0.01)
            if len(stk_r) == len(n50_r) and len(stk_r) > 50:
                cov = np.cov(stk_r, n50_r)
                if cov.shape == (2,2) and cov[1,1] > 1e-10:
                    beta = cov[0,1] / cov[1,1]

        raw[ticker] = {
            "price": p_now, "mr_12": mr_12, "mr_3": mr_3,
            "ret_12m": ret_12, "ret_3m": ret_3, "sigma": sigma, "beta": beta,
        }

    # Beta filter
    raw = {t: s for t, s in raw.items()
           if s["beta"] is not None and s["beta"] <= BETA_CAP}

    # EPS filter
    if eps_db:
        raw = {t: s for t, s in raw.items() if eps_passes(eps_db, t, day)}

    if len(raw) < 20:
        return {}

    # Z-score MR_12 and MR_3
    mr12v = np.array([s["mr_12"] for s in raw.values()])
    mr3v  = np.array([s["mr_3"]  for s in raw.values()])
    mu12, sd12 = np.mean(mr12v), np.std(mr12v)
    mu3,  sd3  = np.mean(mr3v),  np.std(mr3v)
    if sd12 < 0.001 or sd3 < 0.001:
        return {}

    for t, s in raw.items():
        z12 = (s["mr_12"] - mu12) / sd12
        z3  = (s["mr_3"]  - mu3)  / sd3
        wz  = W12 * z12 + W3 * z3
        s["wt_z"]      = wz
        s["norm_score"] = (1 + wz) if wz >= 0 else 1.0 / (1 - wz)

    return raw

# ── REBALANCE DATES ───────────────────────────────────────────────────────────
def get_rebal_dates(trading_days, monthly=False):
    """Return rebalance dates.
    monthly=False: first trading day of Feb/Apr/Jun/Aug/Oct/Dec (Mom15 bi-monthly)
    monthly=True:  first trading day of every month (Mom20 monthly)
    """
    rebal_months = set(range(1, 13)) if monthly else {2, 4, 6, 8, 10, 12}
    seen = set()
    result = []
    for d in trading_days:
        ym = (d.year, d.month)
        if d.month in rebal_months and ym not in seen and d >= START_DATE:
            seen.add(ym)
            result.append(d)
    return result

# ── FORMATTING ────────────────────────────────────────────────────────────────
def inr(v):
    """Format as Indian ₹ with commas."""
    return f"₹{v:,.0f}"

def pct(v):
    sign = '+' if v >= 0 else ''
    return f"{sign}{v:.1f}%"

def print_table(headers, rows, col_widths):
    sep = "  ".join("─" * w for w in col_widths)
    hdr = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(f"  {hdr}")
    print(f"  {sep}")
    for row in rows:
        print("  " + "  ".join(str(c).ljust(w) for c, w in zip(row, col_widths)))

# ── MAIN BACKTEST ─────────────────────────────────────────────────────────────
def run(refresh=False, mom20=False, use_regime=True, beta_cap_override=None):
    # Override constants for Mom20 variant
    global MAX_SLOTS, BUFFER_IN, BUFFER_OUT, BETA_CAP
    if mom20:
        MAX_SLOTS, BUFFER_IN, BUFFER_OUT, BETA_CAP = 20, 15, 40, 1.2
        label = "Mom20 — Monthly Rebalance, β≤1.2"
    else:
        label = "Mom15 — Bi-monthly Rebalance, β≤1.0"
    if beta_cap_override is not None:
        BETA_CAP = beta_cap_override
        label = label.split("β≤")[0] + f"β≤{beta_cap_override}"
    regime_label = "Regime ON" if use_regime else "Regime OFF"
    print(f"=== {label} | {regime_label} ===")

    # Load supporting data
    print("Loading PIT universe...")
    pit_data = load_pit()
    all_tickers = get_all_pit_tickers(pit_data)
    print(f"  {len(all_tickers)} unique PIT tickers across all periods")

    print("Loading EPS data...")
    eps_db = None
    if os.path.exists(EPS_FILE):
        with open(EPS_FILE) as f:
            eps_db = json.load(f)
        print(f"  {len(eps_db)} stocks with EPS data")

    # Date range for data fetch
    fetch_start = START_DATE - timedelta(days=WARMUP_DAYS)
    fetch_end   = date.today()

    # Load/fetch stock data
    stock_data = load_or_fetch_data(all_tickers, fetch_start, fetch_end, refresh)

    # Fetch Nifty 50 for beta (retry up to 3 times on transient yfinance errors)
    print("Fetching Nifty 50 (beta)...")
    import time as _time
    n50_raw = pd.DataFrame()
    for _attempt in range(3):
        try:
            n50_raw = yf.Ticker("^NSEI").history(start=fetch_start, end=fetch_end)
            if not n50_raw.empty:
                break
        except Exception as e:
            print(f"  Attempt {_attempt+1} failed: {e}")
        _time.sleep(2)
    if n50_raw.empty:
        print("  ERROR: Could not fetch Nifty 50 — beta filter disabled")
    else:
        n50_raw.index = n50_raw.index.tz_localize(None) if n50_raw.index.tzinfo else n50_raw.index
        print(f"  {len(n50_raw)} bars")
    n50_iloc = {n50_raw.index[i].date(): i for i in range(len(n50_raw))}

    # Fetch Nifty 200 for regime filter (200-day SMA)
    print("Fetching Nifty 200 (regime filter)...")
    n200_raw = pd.DataFrame()
    for _attempt in range(3):
        try:
            n200_raw = yf.Ticker("^CNX200").history(start=fetch_start, end=fetch_end)
            if not n200_raw.empty:
                break
        except Exception as e:
            print(f"  Attempt {_attempt+1} failed: {e}")
        _time.sleep(2)
    if n200_raw.empty:
        print("  WARNING: Could not fetch Nifty 200 — regime filter disabled")
    else:
        n200_raw.index = n200_raw.index.tz_localize(None) if n200_raw.index.tzinfo else n200_raw.index
        n200_raw["sma200"] = n200_raw["Close"].rolling(200).mean()
        print(f"  {len(n200_raw)} bars")
    n200_iloc = {n200_raw.index[i].date(): i for i in range(len(n200_raw))}

    # Build date → iloc maps
    date_to_iloc = {}
    for ticker, df in stock_data.items():
        date_to_iloc[ticker] = {df.index[i].date(): i for i in range(len(df))}

    # All trading days (days where ≥ 50 stocks have data)
    day_counts = {}
    for df in stock_data.values():
        for d in df.index:
            dt = d.date()
            day_counts[dt] = day_counts.get(dt, 0) + 1
    trading_days = sorted(d for d, c in day_counts.items() if c >= 50 and d >= START_DATE)
    print(f"  Trading days in backtest: {len(trading_days)} ({trading_days[0]} → {trading_days[-1]})")

    rebal_dates = get_rebal_dates(trading_days, monthly=mom20)
    print(f"  Rebalance dates: {len(rebal_dates)}")

    # ── Portfolio state ──────────────────────────────────────────────────────
    cash      = 20_00_000.0   # ₹20L
    portfolio = {}            # ticker → {entry_date, entry_price, shares, entry_cost}
    all_trades = []           # closed trades
    total_charges = 0.0
    rebal_nav = []            # rebalance-date NAV snapshots for portfolio correlation analysis

    if mom20:
        banner = f"  MOM20 PIT BACKTEST  |  NAV/20 slot  |  Monthly Rebalance  |  Beta≤1.2  |  {regime_label}"
    else:
        banner = f"  MOM15 PIT BACKTEST  |  NAV/15 slot  |  2-Month Rebalance  |  Beta≤1.0  |  {regime_label}"
    print()
    print("=" * len(banner))
    print(banner)
    print("=" * len(banner))

    for rebal_idx, rebal_day in enumerate(rebal_dates):
        # Portfolio value on rebal day (MTM)
        port_value = cash
        for t, pos in portfolio.items():
            idx_map = date_to_iloc.get(t, {})
            ci = idx_map.get(rebal_day)
            if ci is None:
                for off in range(1, 6):
                    prev = rebal_day - timedelta(days=off)
                    if prev in idx_map:
                        ci = idx_map[prev]
                        break
            if ci is not None:
                pos["curr_price"] = float(stock_data[t]["Close"].iloc[ci])
            else:
                pos["curr_price"] = pos["entry_price"]
            port_value += pos["shares"] * pos["curr_price"]

        rebal_nav.append({"date": rebal_day, "nav": port_value})

        # Slot size = NAV / 15, recomputed each rebalance (Option A)
        slots_available = MAX_SLOTS
        per_slot = port_value / MAX_SLOTS

        print()
        print("=" * 72)
        print(f"  REBALANCE #{rebal_idx+1:02d}  —  {rebal_day.strftime('%d %b %Y')}")
        print(f"  NAV: {inr(port_value)}  |  Slot: {inr(per_slot)}  |  Cash: {inr(cash)}")
        print("=" * 72)

        # Compute scores
        scores = compute_scores(rebal_day, stock_data, date_to_iloc, pit_data,
                                n50_raw, n50_iloc, eps_db)
        if not scores:
            print(f"  ⚠ Insufficient scored stocks — skipping rebalance")
            continue

        # Rank all stocks
        ranked = sorted(scores.items(), key=lambda x: -x[1]["norm_score"])
        ticker_rank = {t: r+1 for r, (t, _) in enumerate(ranked)}

        # Buffer rule — determine new target portfolio
        current_set = set(portfolio.keys())
        new_set = set()

        # Keeps: existing stays if rank ≤ BUFFER_OUT
        for t in current_set:
            if ticker_rank.get(t, 9999) <= BUFFER_OUT:
                new_set.add(t)

        # New entries: add if rank ≤ BUFFER_IN and not already held
        for r, (t, _) in enumerate(ranked):
            if r + 1 > BUFFER_IN:
                break
            if t not in current_set:
                new_set.add(t)

        # Fill remaining slots to reach slots_available from top of ranking
        for r, (t, _) in enumerate(ranked):
            if len(new_set) >= slots_available:
                break
            new_set.add(t)

        # Cap at slots_available (keep highest-ranked)
        if len(new_set) > slots_available:
            new_set = set(sorted(new_set, key=lambda t: ticker_rank.get(t, 9999))[:slots_available])

        # ── REGIME CHECK (before exits and entries) ──────────────────────────
        regime_off = False
        if use_regime and not n200_raw.empty:
            n200_ci = n200_iloc.get(rebal_day)
            if n200_ci is None:
                for off in range(1, 6):
                    n200_ci = n200_iloc.get(rebal_day - timedelta(days=off))
                    if n200_ci is not None:
                        break
            if n200_ci is not None:
                n200_close = float(n200_raw["Close"].iloc[n200_ci])
                n200_sma   = n200_raw["sma200"].iloc[n200_ci]
                if not pd.isna(n200_sma) and n200_close < float(n200_sma):
                    regime_off = True
                    print(f"\n  [REGIME OFF] Nifty200 {n200_close:,.1f} < SMA200 {float(n200_sma):,.1f} — holding all, skipping exits & entries")

        # ── EXITS ────────────────────────────────────────────────────────────
        to_sell = current_set - new_set
        exit_rows = []
        for t in sorted(to_sell):
            if regime_off:
                break
            pos = portfolio[t]
            ep = pos.get("curr_price", pos["entry_price"])
            gross_pnl = (ep - pos["entry_price"]) * pos["shares"]
            buy_val   = pos["entry_price"] * pos["shares"]
            sell_val  = ep * pos["shares"]
            chg       = calc_charges(buy_val, sell_val)
            net_pnl   = gross_pnl - chg
            pnl_pct   = (ep / pos["entry_price"] - 1) * 100
            hold_days = (rebal_day - pos["entry_date"]).days

            cash += sell_val
            total_charges += chg

            all_trades.append({
                "ticker": t, "entry": pos["entry_date"], "exit": rebal_day,
                "entry_price": pos["entry_price"], "exit_price": ep,
                "shares": pos["shares"], "gross_pnl": gross_pnl,
                "charges": chg, "net_pnl": net_pnl, "hold_days": hold_days,
            })
            exit_rows.append((
                t,
                pos["entry_date"].strftime("%d-%b-%y"),
                f"{pos['entry_price']:,.1f}",
                f"{ep:,.1f}",
                pos["shares"],
                inr(gross_pnl),
                pct(pnl_pct),
                f"{hold_days}d",
            ))
            del portfolio[t]

        print(f"\n  EXITS ({len(exit_rows)})")
        if exit_rows:
            print_table(
                ["Ticker","Entry","Entry₹","Exit₹","Qty","Gross P&L","P&L%","Hold"],
                sorted(exit_rows, key=lambda r: float(r[6].replace('+','').replace('%','')), reverse=True),
                [10, 10, 10, 10, 5, 12, 8, 6]
            )
        else:
            print("    —")

        # ── ENTRIES ──────────────────────────────────────────────────────────
        to_buy = new_set - current_set
        entry_rows = []
        skipped_52w = []

        for t in sorted(to_buy, key=lambda t: ticker_rank.get(t, 9999)):
            if regime_off:
                break
            s = scores[t]
            ep = s["price"]

            # 52-week high filter: skip new entries > 20% below 52w high
            idx_map = date_to_iloc.get(t, {})
            ci = idx_map.get(rebal_day)
            if ci is None:
                for off in range(1, 6):
                    prev = rebal_day - timedelta(days=off)
                    if prev in idx_map:
                        ci = idx_map[prev]
                        break
            if ci is not None and ci >= 252:
                high_52w = float(stock_data[t]["High"].iloc[ci-252:ci+1].max())
                dist_from_high = (ep / high_52w - 1) * 100
                if ep < high_52w * 0.80:
                    skipped_52w.append((t, ticker_rank[t], f"{ep:,.1f}",
                                        f"{high_52w:,.1f}", f"{dist_from_high:.1f}%"))
                    continue

            shares = int(per_slot // ep)
            if shares == 0:
                continue
            cost = ep * shares
            if cost > cash:
                continue
            chg = calc_charges(cost, 0)
            cash -= (cost + chg)
            total_charges += chg
            portfolio[t] = {
                "entry_date": rebal_day,
                "entry_price": ep,
                "shares": shares,
            }
            entry_rows.append((
                t,
                ticker_rank[t],
                f"{s['norm_score']:.3f}",
                f"{s['beta']:.2f}",
                f"{s['ret_12m']*100:+.1f}%",
                f"{s['ret_3m']*100:+.1f}%",
                f"{ep:,.1f}",
                shares,
                inr(cost),
            ))

        print(f"\n  ENTRIES ({len(entry_rows)})")
        if skipped_52w:
            print(f"  [52w filter blocked {len(skipped_52w)}: "
                  + ", ".join(f"{r[0]}({r[4]})" for r in skipped_52w) + "]")
        if entry_rows:
            print_table(
                ["Ticker","Rank","Score","Beta","Ret12m","Ret3m","Entry₹","Qty","Capital"],
                entry_rows,
                [10, 5, 7, 5, 8, 8, 10, 5, 12]
            )
        else:
            print("    —")

        # ── HOLDS ────────────────────────────────────────────────────────────
        holds = current_set & new_set
        hold_rows = []
        for t in sorted(holds, key=lambda t: ticker_rank.get(t, 9999)):
            pos = portfolio[t]
            cp  = pos.get("curr_price", pos["entry_price"])
            unreal = (cp - pos["entry_price"]) * pos["shares"]
            pp  = (cp / pos["entry_price"] - 1) * 100
            hold_rows.append((
                t,
                ticker_rank.get(t, "—"),
                pos["entry_date"].strftime("%d-%b-%y"),
                f"{pos['entry_price']:,.1f}",
                f"{cp:,.1f}",
                pos["shares"],
                inr(unreal),
                pct(pp),
            ))

        print(f"\n  HOLDS ({len(hold_rows)})")
        if hold_rows:
            print_table(
                ["Ticker","Rank","Since","Entry₹","Now₹","Qty","Unreal P&L","P&L%"],
                sorted(hold_rows, key=lambda r: float(r[7].replace('+','').replace('%','')), reverse=True),
                [10, 5, 10, 10, 10, 5, 12, 8]
            )
        else:
            print("    —")

        # ── PORTFOLIO SUMMARY ─────────────────────────────────────────────────
        invested = sum(pos["shares"] * pos.get("curr_price", pos["entry_price"])
                       for pos in portfolio.values())
        total_now = cash + invested
        print(f"\n  AFTER: Invested {inr(invested)} | Cash {inr(cash)} | "
              f"Total {inr(total_now)} | Positions {len(portfolio)}/{MAX_SLOTS} | Slot {inr(per_slot)}")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    # Mark open positions at last available price
    open_pnl = 0.0
    for t, pos in portfolio.items():
        idx_map = date_to_iloc.get(t, {})
        last_ci = max(idx_map.values()) if idx_map else None
        lp = float(stock_data[t]["Close"].iloc[last_ci]) if last_ci else pos["entry_price"]
        open_pnl += (lp - pos["entry_price"]) * pos["shares"]

    closed_pnl  = sum(tr["net_pnl"] for tr in all_trades)
    total_gross  = sum(tr["gross_pnl"] for tr in all_trades)
    winners      = [tr for tr in all_trades if tr["gross_pnl"] > 0]
    losers       = [tr for tr in all_trades if tr["gross_pnl"] <= 0]
    avg_hold     = sum(tr["hold_days"] for tr in all_trades) / len(all_trades) if all_trades else 0

    final_value  = cash + sum(
        pos["shares"] * (float(stock_data[t]["Close"].iloc[max(date_to_iloc[t].values())])
                         if t in date_to_iloc else pos["entry_price"])
        for t, pos in portfolio.items()
    )
    total_return = (final_value - 20_00_000) / 20_00_000 * 100

    # Annualised return (CAGR)
    years = (trading_days[-1] - trading_days[0]).days / 365.25
    cagr  = ((final_value / 20_00_000) ** (1/years) - 1) * 100 if years > 0 else 0

    print()
    print("=" * 72)
    print("  FINAL SUMMARY")
    print("=" * 72)
    print(f"  Period        : {trading_days[0]} → {trading_days[-1]}  ({years:.1f} years)")
    print(f"  Starting Cap  : ₹20,00,000")
    print(f"  Final Value   : {inr(final_value)}")
    print(f"  Total Return  : {pct(total_return)}")
    print(f"  CAGR          : {pct(cagr)}")
    print()
    gross_win  = sum(tr["gross_pnl"] for tr in winners)
    gross_loss = abs(sum(tr["gross_pnl"] for tr in losers))
    pf         = gross_win / gross_loss if gross_loss > 0 else float('inf')
    wr         = len(winners) / len(all_trades) * 100 if all_trades else 0

    print(f"  Closed Trades : {len(all_trades)}  |  Open: {len(portfolio)}")
    print(f"  Win Rate      : {wr:.1f}%  ({len(winners)}W / {len(losers)}L)")
    print(f"  Profit Factor : {pf:.2f}")
    print(f"  Avg hold      : {avg_hold:.0f} days")
    print(f"  Total charges : {inr(total_charges)}")
    print(f"  Closed net P&L: {inr(closed_pnl)}")
    print(f"  Open unreal   : {inr(open_pnl)}")

    # Per-year returns — NAV-based (last rebal NAV of year / last rebal NAV of prior year)
    print()
    print("  YEAR-BY-YEAR:")
    year_last_nav = {}
    for row in rebal_nav:
        yr = row["date"].year
        year_last_nav[yr] = row["nav"]
    initial_nav = 20_00_000.0
    prev_nav_yr = initial_nav
    neg_years = 0
    for yr in sorted(year_last_nav):
        end_nav = year_last_nav[yr]
        ret_y = (end_nav / prev_nav_yr - 1) * 100
        if ret_y < 0:
            neg_years += 1
        bar_len = int(abs(ret_y) / 1)
        bar = ("█" * min(bar_len, 40)) if ret_y > 0 else ("░" * min(bar_len, 40))
        sign = "+" if ret_y >= 0 else "-"
        print(f"  {yr}  {sign}{abs(ret_y):5.1f}%  {bar}")
        prev_nav_yr = end_nav

    print()

    # Export rebalance-date NAV series for portfolio correlation analysis
    if rebal_nav:
        import os as _os
        csv_name = "mom20_rebal.csv" if mom20 else "mom15_rebal.csv"
        nav_csv = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), csv_name)
        pd.DataFrame(rebal_nav).to_csv(nav_csv, index=False)
        print(f"  Rebalance NAV exported → {csv_name} ({len(rebal_nav)} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mom15 / Mom20 PIT Backtest Report")
    parser.add_argument("--refresh", action="store_true", help="Re-download price data")
    parser.add_argument("--mom20", action="store_true",
                        help="Run Mom20 variant (top 20, monthly rebalance, β≤1.2)")
    parser.add_argument("--no-regime", action="store_true",
                        help="Disable regime filter (allow entries even when Nifty200 < SMA200)")
    parser.add_argument("--beta-cap", type=float, default=None,
                        help="Override beta cap (e.g. 1.35)")
    args = parser.parse_args()
    run(refresh=args.refresh, mom20=args.mom20, use_regime=not args.no_regime,
        beta_cap_override=args.beta_cap)
