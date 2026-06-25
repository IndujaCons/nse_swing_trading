#!/usr/bin/env python3
"""
Mom20 PIT Backtest — Monthly Rebalance Report  [--mom20, PRODUCTION]
=====================================================================
Capital   : ₹20L starting, equal weight across 20 slots (compounding)
Slots     : Top 20, Nifty200 PIT universe
Rebalance : First trading day of each month
Scoring   : MR_12 (50%) + MR_3 (50%), Z-scored, Normalised Score
Filters   : Beta ≤ 1.2 vs Nifty 50 | Sector cap ≤ 4
Buffer    : Existing stays if rank ≤ 40, new enters if rank ≤ 15
Regime    : Nifty200 < SMA200 → hold all, no entries/exits

Usage:
    python3 mom15_pit_report.py           # Mom15 (2-monthly, β≤1.0, EPS filter)
    python3 mom15_pit_report.py --mom20   # Mom20 (monthly, β≤1.2, sector cap 4) ← PRODUCTION
    python3 mom15_pit_report.py --refresh # force re-download data
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
BETA_MIN     = None           # None = no minimum beta filter (set to 1.2 for overflow)
ADDV_MIN     = None           # None = no ADDV filter; set to e.g. 10_000_000 (₹10 Cr/day)
ADDV_WINDOW  = 90             # trading days for median ADDV
W12, W3          = 0.50, 0.50    # 12m + 3m, 6m dropped
PARABOLIC_FILTER = False          # skip new entries where Ret12m>300% AND Ret3m/Ret12m>0.5
LONG_PD      = 252            # 12m in trading days
SHORT_PD     = 63             # 3m in trading days
WARMUP_DAYS  = 450            # extra history before START_DATE for warmup

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE   = os.path.join(BASE_DIR, 'data', 'cache', 'mom15_daily.pkl')
PIT_FILE     = os.path.join(BASE_DIR, 'nse_const', 'nifty200_pit.json')
EPS_FILE     = os.path.join(BASE_DIR, 'data', 'quarterly_eps.json')
SECTOR_MAP_N200 = os.path.join(BASE_DIR, 'nse_const', 'nifty200_sector_map_pit.json')
SECTOR_MAP_N500 = os.path.join(BASE_DIR, 'nse_const', 'nifty500_sector_map_pit.json')

SECTOR_SHORT = {
    "NIFTY IT": "IT", "NIFTY AUTO": "AUTO", "NIFTY HEALTHCARE": "HEALTH",
    "NIFTY FIN SERVICE": "FIN SVC", "NIFTY PVT BANK": "PVT BNK",
    "NIFTY PSU BANK": "PSU BNK", "NIFTY BANK": "BANK",
    "NIFTY ENERGY": "ENERGY", "NIFTY FMCG": "FMCG", "NIFTY METAL": "METAL",
    "NIFTY INFRA": "INFRA", "NIFTY INDIA MFG": "MFG",
    "NIFTY CONSUMER DURABLES": "CON DUR", "NIFTY CONSUMPTION": "CONSUMP",
    "NIFTY REALTY": "REALTY", "NIFTY MEDIA": "MEDIA",
    "NIFTY OIL & GAS": "OIL&GAS", "NIFTY INDIA DEFENCE": "DEFENCE",
    "NIFTY PSE": "PSE", "NIFTY MNC": "MNC", "OTHER": "OTHER",
}

# Zerodha delivery charges
def calc_charges(buy_val, sell_val):
    total  = buy_val + sell_val
    stt    = 0.001    * total
    exch   = 0.0000307 * total
    sebi   = 0.000001  * total
    stamp  = 0.00015   * buy_val
    gst    = 0.18      * (exch + sebi)
    return stt + exch + sebi + stamp + gst

def xirr(cashflows):
    """cashflows: list of (date, amount) — negative=outflow, positive=inflow. Returns annualised IRR."""
    if not cashflows:
        return float('nan')
    t0 = cashflows[0][0]
    days = [(d - t0).days for d, _ in cashflows]
    amts = [a for _, a in cashflows]
    def npv(r):
        return sum(a / (1 + r) ** (d / 365.0) for a, d in zip(amts, days))
    try:
        lo, hi = -0.9999, 50.0
        if npv(lo) * npv(hi) > 0:
            return float('nan')
        for _ in range(200):
            mid = (lo + hi) / 2
            if npv(mid) > 0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-9:
                break
        return (lo + hi) / 2
    except Exception:
        return float('nan')

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
def load_pit(path=None):
    with open(path or PIT_FILE) as f:
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
def fetch_ticker(ticker, start, end, us_mode=False):
    if us_mode:
        sym = ticker  # US tickers need no suffix
    else:
        sym = ALIASES.get(ticker, f"{ticker}.NS")
    try:
        df = yf.Ticker(sym).history(start=start, end=end)
        if not us_mode and (df.empty or len(df) < 200):
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

def load_or_fetch_data(tickers, fetch_start, fetch_end, refresh=False, cache_file=None, us_mode=False):
    cache_file = cache_file or CACHE_FILE
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    if not refresh and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"Fetching {len(tickers)} tickers from yfinance (this takes ~5-10 min)...")
    stock_data = {}
    for i, ticker in enumerate(sorted(tickers)):
        df = fetch_ticker(ticker, fetch_start, fetch_end, us_mode=us_mode)
        if not df.empty and len(df) >= 200:
            stock_data[ticker] = df
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(tickers)} done, {len(stock_data)} loaded...")
    print(f"  Done. {len(stock_data)} stocks with sufficient history.")
    with open(cache_file, 'wb') as f:
        pickle.dump(stock_data, f)
    print(f"  Cached to {cache_file}")
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

        # ADDV filter: median daily dollar volume over trailing ADDV_WINDOW bars
        if ADDV_MIN is not None and "Volume" in df.columns:
            vols = df["Volume"].values.astype(float)
            win_start = max(0, ci - ADDV_WINDOW)
            addv = np.median(closes[win_start:ci+1] * vols[win_start:ci+1])
            if addv < ADDV_MIN:
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
    if BETA_MIN is not None:
        raw = {t: s for t, s in raw.items() if s["beta"] >= BETA_MIN}

    # EPS filter
    if eps_db:
        raw = {t: s for t, s in raw.items() if eps_passes(eps_db, t, day)}

    min_stocks = 3 if BETA_MIN is not None else 20
    if len(raw) < min_stocks:
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
def get_rebal_dates(trading_days, monthly=False, weekly=False, every_n_days=None, quarterly=False, rebal_day="start"):
    """Return rebalance dates.
    weekly=True:      every Monday (or first trading day of each week)
    monthly=False:    first trading day of Feb/Apr/Jun/Aug/Oct/Dec (Mom15 bi-monthly)
    monthly=True:     first trading day of every month (Mom20 monthly)
    every_n_days=N:   every N calendar days from START_DATE
    rebal_day:        "start" = first trading day of (rebal) month;
                      "mid"   = first trading day on or after the 15th of (rebal) month
    """
    if every_n_days:
        result = []
        last = None
        for d in trading_days:
            if d < START_DATE:
                continue
            if last is None or (d - last).days >= every_n_days:
                result.append(d)
                last = d
        return result
    if weekly:
        seen = set()
        result = []
        for d in trading_days:
            iso = d.isocalendar()
            yw = (iso[0], iso[1])  # (year, week)
            if yw not in seen and d >= START_DATE:
                seen.add(yw)
                result.append(d)
        return result
    rebal_months = set(range(1, 13)) if monthly else ({1, 4, 7, 10} if quarterly else {2, 4, 6, 8, 10, 12})
    seen = set()
    result = []
    for d in trading_days:
        ym = (d.year, d.month)
        if d.month in rebal_months and ym not in seen and d >= START_DATE:
            if rebal_day == "mid" and d.day < 15:
                # Defer until first trading day on or after the 15th
                continue
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

def ema20_ext(stock_data, date_to_iloc, ticker, day):
    """Return (price - EMA20) / EMA20 * 100, or None if insufficient data."""
    idx_map = date_to_iloc.get(ticker, {})
    ci = idx_map.get(day)
    if ci is None:
        for off in range(1, 6):
            ci = idx_map.get(day - timedelta(days=off))
            if ci is not None:
                break
    if ci is None or ci < 19:
        return None
    closes = stock_data[ticker]["Close"].iloc[:ci + 1]
    ema20 = float(closes.ewm(span=20, adjust=False).mean().iloc[-1])
    price = float(closes.iloc[-1])
    return (price / ema20 - 1) * 100

def print_table(headers, rows, col_widths):
    sep = "  ".join("─" * w for w in col_widths)
    hdr = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(f"  {hdr}")
    print(f"  {sep}")
    for row in rows:
        print("  " + "  ".join(str(c).ljust(w) for c, w in zip(row, col_widths)))

# ── MAIN BACKTEST ─────────────────────────────────────────────────────────────
def run(refresh=False, mom20=False, overflow=False, use_regime=True, beta_cap_override=None, regime_exit=False, n500=False, qqq=False, sp500=False, start_override=None,
        top_n_override=None, buffer_in_override=None, buffer_out_override=None,
        ema200_exit=False, rebal_day="start", regime_filter="sma200", parabolic_filter=False,
        sector_cap=None, addv_min=None, niftybees=False, goldbees=False, sip=0,
        skip_extended=None):
    # Override constants for Mom20 / Overflow / N500 / QQQ / SP500 variants
    global MAX_SLOTS, BUFFER_IN, BUFFER_OUT, BETA_CAP, BETA_MIN, ADDV_MIN, W12, W3, START_DATE, PARABOLIC_FILTER
    PARABOLIC_FILTER = parabolic_filter
    BETA_MIN  = None  # reset each run
    ADDV_MIN  = addv_min  # None = disabled
    if start_override:
        START_DATE = date.fromisoformat(start_override)
    W12, W3 = 0.50, 0.50  # reset each run
    if overflow:
        MAX_SLOTS, BUFFER_IN, BUFFER_OUT = 5, 5, 40
        BETA_CAP, BETA_MIN = 99.0, 1.2   # β>1.2 entry only; exit uses full universe
        label = "Overflow — 5-slot β>1.2, Monthly, Top-40 Exit"
        use_regime = False  # no regime filter for overflow
    elif qqq:
        MAX_SLOTS, BUFFER_IN, BUFFER_OUT, BETA_CAP = 10, 7, 20, 99.0  # no beta cap for US
        label = "MomQQQ — Nasdaq-100 Universe, Top10, Monthly Rebalance, No β filter"
    elif sp500:
        MAX_SLOTS, BUFFER_IN, BUFFER_OUT, BETA_CAP = 20, 15, 40, 99.0  # no beta cap for US
        label = "MomSP500 — S&P 500 Universe, 2-Month Rebalance, No β filter"
    elif n500:
        MAX_SLOTS, BUFFER_IN, BUFFER_OUT, BETA_CAP = 20, 15, 40, 1.2
        label = "Mom500 — Nifty500 Universe, Monthly Rebalance, β≤1.2"
    elif mom20:
        MAX_SLOTS, BUFFER_IN, BUFFER_OUT, BETA_CAP = 20, 15, 40, 1.2
        label = "Mom20 — Monthly Rebalance, β≤1.2"
    else:
        label = "Mom15 — Bi-monthly Rebalance, β≤1.0"
    # Default sector cap for N200/N500 mom20 variants
    if sector_cap is None and (mom20 or n500):
        sector_cap = 4
    if beta_cap_override is not None:
        BETA_CAP = beta_cap_override
        label = label.split("β≤")[0] + f"β≤{beta_cap_override}"
    # CLI overrides for top_n / buffer_in / buffer_out
    if top_n_override is not None:
        MAX_SLOTS = top_n_override
    if buffer_in_override is not None:
        BUFFER_IN = buffer_in_override
    if buffer_out_override is not None:
        BUFFER_OUT = buffer_out_override
    # `regime_filter` resolves which moving-average the regime check uses:
    #   'sma200' (default) — original behaviour, Nifty200 < SMA200 → off
    #   'ema200'           — uses EMA(200) instead (slightly more responsive)
    #   'none'             — no regime filter at all (overrides use_regime)
    if regime_filter not in ("sma200", "ema200", "none"):
        regime_filter = "sma200"
    if regime_filter == "none":
        use_regime = False
    regime_label = ("Regime ON [" + regime_filter.upper() + "]"
                    if use_regime else "Regime OFF")
    extra = []
    if ema200_exit:
        extra.append("EMA200-exit")
    if rebal_day == "mid":
        extra.append("Mid-month rebal")
    if sector_cap is not None:
        extra.append(f"Sector≤{sector_cap}")
    if ADDV_MIN is not None:
        extra.append(f"ADDV≥₹{ADDV_MIN/1e7:.0f}Cr")
    if sip > 0:
        extra.append(f"SIP ₹{sip/1e5:.0f}L/mo")
    extra_label = (" | " + " | ".join(extra)) if extra else ""
    print(f"=== {label} | {regime_label}{extra_label} ===")
    print(f"    top_n={MAX_SLOTS} buffer_in={BUFFER_IN} buffer_out={BUFFER_OUT} beta_cap={BETA_CAP}")

    # Load supporting data
    if qqq:
        pit_file     = os.path.join(BASE_DIR, 'nse_const', 'qqq_pit.json')
        cache_file   = os.path.join(BASE_DIR, 'data', 'cache', 'qqq_daily.pkl')
        bench_ticker = "QQQ"
        bench_label  = "QQQ (Nasdaq-100)"
        beta_bench_ticker = "^NDX"
    elif sp500:
        pit_file     = os.path.join(BASE_DIR, 'nse_const', 'sp500_pit.json')
        cache_file   = os.path.join(BASE_DIR, 'data', 'cache', 'sp500_daily.pkl')
        bench_ticker = "SPY"
        bench_label  = "S&P 500 (SPY)"
        beta_bench_ticker = "^GSPC"
    elif n500:
        pit_file     = os.path.join(BASE_DIR, 'nse_const', 'nifty500_pit.json')
        cache_file   = os.path.join(BASE_DIR, 'data', 'cache', 'mom500_daily.pkl')
        bench_ticker = "^CRSLDX"
        bench_label  = "Nifty 500"
        beta_bench_ticker = "^NSEI"
    else:
        pit_file     = PIT_FILE
        cache_file   = CACHE_FILE
        bench_ticker = "^CNX200"
        bench_label  = "Nifty 200"
        beta_bench_ticker = "^NSEI"

    print("Loading PIT universe...")
    pit_data = load_pit(pit_file)
    all_tickers = get_all_pit_tickers(pit_data)
    print(f"  {len(all_tickers)} unique PIT tickers across all periods")

    print("Loading EPS data...")
    eps_db = None
    if os.path.exists(EPS_FILE):
        with open(EPS_FILE) as f:
            eps_db = json.load(f)
        print(f"  {len(eps_db)} stocks with EPS data")

    # Load sector map for sector cap / display
    sector_map_pit = {}
    sector_map_dates = []
    if not qqq and not sp500:
        sector_map_file = SECTOR_MAP_N500 if n500 else SECTOR_MAP_N200
        if os.path.exists(sector_map_file):
            with open(sector_map_file) as f:
                sector_map_pit = json.load(f)
            sector_map_dates = sorted(sector_map_pit.keys())
            print(f"  Sector map loaded: {len(sector_map_dates)} PIT dates")

    def get_sector(ticker, day_str):
        if not sector_map_dates:
            return "—"
        pit_date = next((d for d in reversed(sector_map_dates) if d <= day_str), None)
        if not pit_date:
            return "—"
        sec = sector_map_pit[pit_date].get(ticker, {}).get("primary_sector", "OTHER")
        return SECTOR_SHORT.get(sec, sec)

    # Date range for data fetch
    fetch_start = START_DATE - timedelta(days=WARMUP_DAYS)
    fetch_end   = date.today()

    # Load/fetch stock data
    us_mode = qqq or sp500
    stock_data = load_or_fetch_data(all_tickers, fetch_start, fetch_end, refresh, cache_file=cache_file, us_mode=us_mode)

    # Fetch benchmark index for beta (retry up to 3 times on transient yfinance errors)
    beta_label = "Nasdaq-100 (^NDX)" if qqq else ("S&P 500 (^GSPC)" if sp500 else "Nifty 50")
    print(f"Fetching {beta_label} (beta)...")
    import time as _time
    n50_raw = pd.DataFrame()
    for _attempt in range(3):
        try:
            n50_raw = yf.Ticker(beta_bench_ticker).history(start=fetch_start, end=fetch_end)
            if not n50_raw.empty:
                break
        except Exception as e:
            print(f"  Attempt {_attempt+1} failed: {e}")
        _time.sleep(2)
    if n50_raw.empty:
        print(f"  ERROR: Could not fetch {beta_label} — beta filter disabled")
    else:
        n50_raw.index = n50_raw.index.tz_localize(None) if n50_raw.index.tzinfo else n50_raw.index
        print(f"  {len(n50_raw)} bars")
    n50_iloc = {n50_raw.index[i].date(): i for i in range(len(n50_raw))}

    # Fetch benchmark index for regime filter (200-day SMA)
    print(f"Fetching {bench_label} (regime filter)...")
    n200_raw = pd.DataFrame()
    for _attempt in range(3):
        try:
            n200_raw = yf.Ticker(bench_ticker).history(start=fetch_start, end=fetch_end)
            if not n200_raw.empty:
                break
        except Exception as e:
            print(f"  Attempt {_attempt+1} failed: {e}")
        _time.sleep(2)
    if n200_raw.empty:
        print(f"  WARNING: Could not fetch {bench_label} — regime filter disabled")
    else:
        n200_raw.index = n200_raw.index.tz_localize(None) if n200_raw.index.tzinfo else n200_raw.index
        n200_raw["sma200"] = n200_raw["Close"].rolling(200).mean()
        # EMA(200) populated whether or not we use it — cheap to compute.
        n200_raw["ema200"] = n200_raw["Close"].ewm(span=200, adjust=False).mean()
        print(f"  {len(n200_raw)} bars")
    n200_iloc = {n200_raw.index[i].date(): i for i in range(len(n200_raw))}

    # Build passive ETF list from flags (each gets 5% allocation, rebalanced monthly)
    _passive_cfg = []
    if niftybees:
        _passive_cfg.append(("NIFTYBEES.NS", 0.05, "NIFTYBEES"))
    if goldbees:
        _passive_cfg.append(("GOLDBEES.NS", 0.05, "GOLDBEES"))

    passive_list = []  # [{ns, alloc, label, raw, iloc, pos, charges, curr_price}]
    for _ns, _alloc, _label in _passive_cfg:
        print(f"Fetching {_ns} ({_alloc*100:.0f}% passive allocation)...")
        _raw = pd.DataFrame()
        for _attempt in range(3):
            try:
                _raw = yf.Ticker(_ns).history(start=fetch_start, end=fetch_end)
                if not _raw.empty:
                    break
            except Exception as e:
                print(f"  Attempt {_attempt+1} failed: {e}")
            _time.sleep(2)
        if _raw.empty:
            print(f"  WARNING: Could not fetch {_ns} — skipping")
            continue
        _raw.index = _raw.index.tz_localize(None) if _raw.index.tzinfo else _raw.index
        print(f"  {len(_raw)} bars")
        passive_list.append({
            "ns": _ns, "alloc": _alloc, "label": _label,
            "raw": _raw,
            "iloc": {_raw.index[i].date(): i for i in range(len(_raw))},
            "pos": {"shares": 0, "avg_cost": 0.0},
            "charges": 0.0,
            "curr_price": 0.0,
        })
    total_passive_alloc = sum(p["alloc"] for p in passive_list)

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

    rebal_dates = get_rebal_dates(trading_days, monthly=(mom20 or n500 or qqq or overflow),
                                  weekly=False, rebal_day=rebal_day)
    print(f"  Rebalance dates: {len(rebal_dates)}")

    # ── Portfolio state ──────────────────────────────────────────────────────
    cash      = 10_00_000.0 if overflow else 20_00_000.0
    _starting_cash = cash
    portfolio = {}            # ticker → {entry_date, entry_price, shares, entry_cost}
    all_trades = []           # closed trades
    total_charges = 0.0
    _sip_total = 0.0          # cumulative SIP injected
    _xirr_flows = []          # (date, amount) — negative=outflow for XIRR
    _twr_start_nav = cash     # tracks post-SIP NAV for TWR sub-period denominator
    _twr_data = []            # (date, sub_period_return) for year-by-year TWR
    rebal_nav = []            # rebalance-date NAV snapshots for portfolio correlation analysis

    if overflow:
        banner = f"  OVERFLOW PIT BACKTEST  |  NAV/5 slot  |  Monthly Rebalance  |  Beta>1.2 Entry  |  Top-40 Exit (full universe)  |  {regime_label}"
    elif qqq:
        banner = f"  MOMQQQ PIT BACKTEST  |  NAV/10 slot  |  Monthly Rebalance  |  Nasdaq-100 Universe  |  No Beta Filter  |  {regime_label}"
    elif sp500:
        banner = f"  MOMSP500 PIT BACKTEST  |  NAV/20 slot  |  2-Month Rebalance  |  S&P 500 Universe  |  No Beta Filter  |  {regime_label}"
    elif n500:
        banner = f"  MOM500 PIT BACKTEST  |  NAV/20 slot  |  Monthly Rebalance  |  Nifty500 Universe  |  Beta≤1.2  |  {regime_label}"
    elif mom20:
        _pb_tag = ("  |  +" + "+".join(f"{p['label']} {p['alloc']*100:.0f}%" for p in passive_list)) if passive_list else ""
        banner = f"  MOM20 PIT BACKTEST  |  NAV/20 slot  |  Monthly Rebalance  |  Beta≤1.2  |  {regime_label}{_pb_tag}"
    else:
        banner = f"  MOM15 PIT BACKTEST  |  NAV/15 slot  |  2-Month Rebalance  |  Beta≤1.0  |  {regime_label}"
    print()
    print("=" * len(banner))
    print(banner)
    print("=" * len(banner))

    # Record initial capital as first XIRR outflow
    if sip > 0 and rebal_dates:
        _xirr_flows.append((rebal_dates[0], -_starting_cash))

    for rebal_idx, rebal_day in enumerate(rebal_dates):
        # Portfolio value on rebal day (MTM) — BEFORE SIP injection
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

        # Include passive ETF current values in NAV
        for p in passive_list:
            p_ci = p["iloc"].get(rebal_day)
            if p_ci is None:
                for off in range(1, 6):
                    prev = rebal_day - timedelta(days=off)
                    if prev in p["iloc"]:
                        p_ci = p["iloc"][prev]
                        break
            if p_ci is not None:
                p["curr_price"] = float(p["raw"]["Close"].iloc[p_ci])
            elif p["pos"]["shares"] > 0:
                p["curr_price"] = p["pos"]["avg_cost"]
            else:
                p["curr_price"] = 0.0
            port_value += p["pos"]["shares"] * p["curr_price"]

        # TWR sub-period return (based on pre-SIP NAV vs prior post-SIP NAV)
        if sip > 0:
            _twr_data.append((rebal_day, port_value / _twr_start_nav - 1))
            cash += sip
            _sip_total += sip
            # rebal_idx==0: initial capital already recorded pre-loop; skip to avoid double-count
            if rebal_idx > 0:
                _xirr_flows.append((rebal_day, -sip))
            port_value += sip           # include fresh capital in working NAV
            _twr_start_nav = port_value  # next sub-period starts after injection

        rebal_nav.append({"date": rebal_day, "nav": port_value})

        # Slot size: momentum portion = (1 - total passive alloc) of NAV
        slots_available = MAX_SLOTS
        per_slot = (port_value * (1 - total_passive_alloc)) / MAX_SLOTS if passive_list else port_value / MAX_SLOTS

        print()
        print("=" * 72)
        print(f"  REBALANCE #{rebal_idx+1:02d}  —  {rebal_day.strftime('%d %b %Y')}")
        print(f"  NAV: {inr(port_value)}  |  Slot: {inr(per_slot)}  |  Cash: {inr(cash)}")
        print("=" * 72)

        # ── PASSIVE ETF REBALANCE (always held at target %, independent of regime) ─
        for p in passive_list:
            cp = p["curr_price"]
            if cp <= 0:
                continue
            p_target  = port_value * p["alloc"]
            p_current = p["pos"]["shares"] * cp
            diff      = p_target - p_current
            if diff > cp:  # buy at least 1 share
                shares_to_buy = int(diff // cp)
                buy_val = shares_to_buy * cp
                chg = calc_charges(buy_val, 0)
                if buy_val + chg <= cash:
                    cash -= buy_val + chg
                    total_charges    += chg
                    p["charges"]     += chg
                    old_val = p["pos"]["shares"] * p["pos"]["avg_cost"]
                    p["pos"]["shares"]   += shares_to_buy
                    p["pos"]["avg_cost"]  = (old_val + buy_val) / p["pos"]["shares"]
                    print(f"  [{p['label']}] BUY  {shares_to_buy:>5} @ {cp:,.1f}"
                          f" = {inr(buy_val)} | total {p['pos']['shares']} shares"
                          f" | {p['alloc']*100:.0f}% target {inr(p_target)}")
            elif diff < -cp and p["pos"]["shares"] > 0:  # sell at least 1 share
                shares_to_sell = min(int(abs(diff) // cp), p["pos"]["shares"])
                sell_val = shares_to_sell * cp
                chg = calc_charges(0, sell_val)
                cash += sell_val - chg
                total_charges    += chg
                p["charges"]     += chg
                p["pos"]["shares"] -= shares_to_sell
                if p["pos"]["shares"] == 0:
                    p["pos"]["avg_cost"] = 0.0
                print(f"  [{p['label']}] SELL {shares_to_sell:>5} @ {cp:,.1f}"
                      f" = {inr(sell_val)} | total {p['pos']['shares']} shares"
                      f" | {p['alloc']*100:.0f}% target {inr(p_target)}")

        # Compute scores
        scores = compute_scores(rebal_day, stock_data, date_to_iloc, pit_data,
                                n50_raw, n50_iloc, eps_db)
        if not scores:
            print(f"  ⚠ Insufficient scored stocks — skipping rebalance")
            continue

        # Rank all stocks (β-filtered universe — for entry and exit buffer logic)
        ranked = sorted(scores.items(), key=lambda x: -x[1]["norm_score"])
        ticker_rank = {t: r+1 for r, (t, _) in enumerate(ranked)}

        # For overflow: compute unfiltered rank (full Nifty200, no beta_min) for exits
        ticker_rank_unfiltered = ticker_rank
        if overflow:
            old_beta_min, old_beta_cap = BETA_MIN, BETA_CAP
            BETA_MIN, BETA_CAP = None, 1.2
            scores_unfiltered = compute_scores(rebal_day, stock_data, date_to_iloc,
                                               pit_data, n50_raw, n50_iloc, eps_db)
            BETA_MIN, BETA_CAP = old_beta_min, old_beta_cap
            if scores_unfiltered:
                ranked_unfiltered = sorted(scores_unfiltered.items(),
                                           key=lambda x: -x[1]["norm_score"])
                ticker_rank_unfiltered = {t: r+1 for r, (t, _) in enumerate(ranked_unfiltered)}

        # Full-universe rank (no beta filter) — display only.
        # Stocks whose β drifted above the cap are absent from ticker_rank (show "—");
        # ticker_rank_display restores their true rank for the exit table.
        old_beta_min, old_beta_cap = BETA_MIN, BETA_CAP
        BETA_MIN, BETA_CAP = None, 99.0
        scores_display = compute_scores(rebal_day, stock_data, date_to_iloc,
                                        pit_data, n50_raw, n50_iloc, eps_db)
        BETA_MIN, BETA_CAP = old_beta_min, old_beta_cap
        if scores_display:
            ranked_display = sorted(scores_display.items(), key=lambda x: -x[1]["norm_score"])
            ticker_rank_display = {t: r+1 for r, (t, _) in enumerate(ranked_display)}
        else:
            ticker_rank_display = ticker_rank

        # Buffer rule — determine new target portfolio
        current_set = set(portfolio.keys())
        new_set = set()

        # Keeps: overflow exits by rank in full Nifty200; others use β-filtered rank
        for t in current_set:
            if ticker_rank_unfiltered.get(t, 9999) <= BUFFER_OUT:
                new_set.add(t)

        # New entries: add if rank ≤ BUFFER_IN and not already held
        skipped_parabolic = []
        for r, (t, _) in enumerate(ranked):
            if r + 1 > BUFFER_IN:
                break
            if t not in current_set and t not in new_set:
                if PARABOLIC_FILTER:
                    ret12 = scores[t].get("ret_12m", 0)
                    ret3  = scores[t].get("ret_3m", 0)
                    if ret12 > 3.0 and ret3 > 0 and (ret3 / ret12) > 0.5:
                        skipped_parabolic.append((t, ret12, ret3))
                        continue
                new_set.add(t)

        # Fill remaining slots to reach slots_available from top of ranking
        for r, (t, _) in enumerate(ranked):
            if len(new_set) >= slots_available:
                break
            if t not in new_set:
                if PARABOLIC_FILTER:
                    ret12 = scores[t].get("ret_12m", 0)
                    ret3  = scores[t].get("ret_3m", 0)
                    if ret12 > 3.0 and ret3 > 0 and (ret3 / ret12) > 0.5:
                        continue
                new_set.add(t)

        # Cap at slots_available (keep highest-ranked)
        if len(new_set) > slots_available:
            new_set = set(sorted(new_set, key=lambda t: ticker_rank.get(t, 9999))[:slots_available])

        # ── SECTOR CAP ────────────────────────────────────────────────────────
        if sector_cap is not None and sector_map_dates:
            day_str = rebal_day.isoformat()
            pit_date = next((d for d in reversed(sector_map_dates) if d <= day_str), None)
            if pit_date:
                day_sectors = sector_map_pit[pit_date]
                def _get_sec(t):
                    sec = day_sectors.get(t, {}).get("primary_sector", "OTHER")
                    return sec
                # Sort: held stocks first (by unfiltered rank), then new entries (by filtered rank)
                def _sort_key(t):
                    if t in current_set:
                        return (0, ticker_rank_unfiltered.get(t, 9999))
                    return (1, ticker_rank.get(t, 9999))
                new_set_ranked = sorted(new_set, key=_sort_key)
                sec_counts = {}
                capped_set = set()
                dropped_by_cap = []
                for t in new_set_ranked:
                    sec = _get_sec(t)
                    if sec_counts.get(sec, 0) < sector_cap:
                        capped_set.add(t)
                        sec_counts[sec] = sec_counts.get(sec, 0) + 1
                    else:
                        dropped_by_cap.append(t)
                if dropped_by_cap:
                    print(f"  [SECTOR CAP≤{sector_cap}] dropped: {', '.join(dropped_by_cap)}")
                    # Fill freed slots from next-best in ranking
                    for _, (t, _) in enumerate(ranked):
                        if len(capped_set) >= slots_available:
                            break
                        if t in capped_set:
                            continue
                        sec = _get_sec(t)
                        if sec_counts.get(sec, 0) < sector_cap:
                            capped_set.add(t)
                            sec_counts[sec] = sec_counts.get(sec, 0) + 1
                new_set = capped_set

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
                # Pick MA column based on regime_filter ("sma200" | "ema200").
                ma_col = "ema200" if regime_filter == "ema200" else "sma200"
                ma_label = "EMA200" if regime_filter == "ema200" else "SMA200"
                n200_ma = n200_raw[ma_col].iloc[n200_ci]
                if not pd.isna(n200_ma) and n200_close < float(n200_ma):
                    regime_off = True
                    print(f"\n  [REGIME OFF] {bench_label} {n200_close:,.1f} < {ma_label} {float(n200_ma):,.1f} — holding all, skipping exits & entries")

        # ── EMA200 EXIT FILTER ───────────────────────────────────────────────
        # If --ema200-exit set, force-exit any holding whose close < EMA(200).
        # Skipped under regime-off (when exits are blocked) so we don't churn.
        ema200_forced_exits = set()
        if ema200_exit and not (regime_off and not regime_exit):
            for t in list(current_set):
                idx_map = date_to_iloc.get(t, {})
                ci = idx_map.get(rebal_day)
                if ci is None:
                    for off in range(1, 6):
                        prev = rebal_day - timedelta(days=off)
                        if prev in idx_map:
                            ci = idx_map[prev]
                            break
                if ci is None or ci < 200:
                    continue
                closes_arr = stock_data[t]["Close"].iloc[max(0, ci-400):ci+1]
                if len(closes_arr) < 200:
                    continue
                ema200_val = float(closes_arr.ewm(span=200, adjust=False).mean().iloc[-1])
                close_val = float(closes_arr.iloc[-1])
                if close_val < ema200_val:
                    ema200_forced_exits.add(t)
                    new_set.discard(t)
        if ema200_forced_exits:
            print(f"\n  [EMA200 EXIT] forcing exits on {len(ema200_forced_exits)}: {', '.join(sorted(ema200_forced_exits))}")

        # ── EXITS ────────────────────────────────────────────────────────────
        to_sell = current_set - new_set
        exit_rows = []
        for t in sorted(to_sell):
            if regime_off and not regime_exit:
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
                ticker_rank_display.get(t, "—"),
                get_sector(t, rebal_day.isoformat()),
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
                ["Ticker","Rank","Sector","Entry","Entry₹","Exit₹","Qty","Gross P&L","P&L%","Hold"],
                sorted(exit_rows, key=lambda r: float(r[8].replace('+','').replace('%','')), reverse=True),
                [10, 5, 9, 10, 10, 10, 5, 12, 8, 6]
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
            if not qqq and not sp500 and not overflow and ci is not None and ci >= 252:
                high_52w = float(stock_data[t]["High"].iloc[ci-252:ci+1].max())
                dist_from_high = (ep / high_52w - 1) * 100
                if ep < high_52w * 0.80:
                    skipped_52w.append((t, ticker_rank[t], f"{ep:,.1f}",
                                        f"{high_52w:,.1f}", f"{dist_from_high:.1f}%"))
                    continue

            if skip_extended is not None and ci is not None:
                ext = ema20_ext(stock_data, date_to_iloc, t, rebal_day)
                if ext is not None and ext > skip_extended:
                    skipped_52w.append((t, ticker_rank[t], f"{ep:,.1f}", "EMA20", f"+{ext:.1f}%"))
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
            r12, r3 = s['ret_12m'], s['ret_3m']
            para_warn = r12 > 3.0 and r3 > 0 and (r3 / r12) > 0.5
            ext = ema20_ext(stock_data, date_to_iloc, t, rebal_day)
            ext_str = pct(ext) if ext is not None else "—"
            entry_rows.append((
                t,
                ticker_rank[t],
                get_sector(t, rebal_day.isoformat()),
                f"{s['norm_score']:.3f}",
                f"{s['beta']:.2f}",
                f"{r12*100:+.1f}%" + (" ⚠" if para_warn else ""),
                f"{r3*100:+.1f}%",
                f"{ep:,.1f}",
                shares,
                inr(cost),
                ext_str,
            ))

        print(f"\n  ENTRIES ({len(entry_rows)})")
        if skipped_52w:
            print(f"  [52w filter blocked {len(skipped_52w)}: "
                  + ", ".join(f"{r[0]}({r[4]})" for r in skipped_52w) + "]")
        if skipped_parabolic:
            parts = [f"{t}(12m:{r12*100:+.0f}%,3m:{r3*100:+.0f}%,ratio:{r3/r12:.2f})"
                     for t, r12, r3 in skipped_parabolic]
            print(f"  [PARABOLIC FILTER 3m/12m>0.5] skipped {len(skipped_parabolic)}: "
                  f"{', '.join(parts)}")
        if entry_rows:
            print_table(
                ["Ticker","Rank","Sector","Score","Beta","Ret12m","Ret3m","Entry₹","Qty","Capital","Ext%EMA20"],
                entry_rows,
                [10, 5, 9, 7, 5, 8, 8, 10, 5, 12, 10]
            )
        else:
            print("    —")

        # ── HOLDS ────────────────────────────────────────────────────────────
        # When regime is OFF and exits blocked — show full portfolio, not just new_set intersection
        holds = (current_set if (regime_off and not regime_exit) else (current_set & new_set))
        hold_rows = []
        warn_syms = []
        for t in sorted(holds, key=lambda t: ticker_rank.get(t, 9999)):
            pos   = portfolio[t]
            cp    = pos.get("curr_price", pos["entry_price"])
            unreal = (cp - pos["entry_price"]) * pos["shares"]
            pp    = (cp / pos["entry_price"] - 1) * 100
            sc    = scores.get(t, {}).get("norm_score", 1.0)
            warn  = " ⚠" if sc < 1.0 else ""
            if sc < 1.0:
                warn_syms.append(t)
            ext = ema20_ext(stock_data, date_to_iloc, t, rebal_day)
            ext_str = pct(ext) if ext is not None else "—"
            hold_rows.append((
                t,
                ticker_rank_display.get(t, "—"),
                get_sector(t, rebal_day.isoformat()),
                pos["entry_date"].strftime("%d-%b-%y"),
                f"{pos['entry_price']:,.1f}",
                f"{cp:,.1f}",
                pos["shares"],
                inr(unreal),
                f"{pct(pp)}{warn}",
                ext_str,
            ))

        print(f"\n  HOLDS ({len(hold_rows)})")
        if hold_rows:
            print_table(
                ["Ticker","Rank","Sector","Since","Entry₹","Now₹","Qty","Unreal P&L","P&L%","Ext%EMA20"],
                sorted(hold_rows, key=lambda r: float(r[8].replace('+','').replace('%','').replace(' ⚠','')), reverse=True),
                [10, 5, 9, 10, 10, 10, 5, 12, 10, 10]
            )
            if warn_syms:
                print(f"  ⚠  WAZ < 0 (momentum below universe mean): {', '.join(warn_syms)}")
        else:
            print("    —")

        # ── PORTFOLIO SUMMARY ─────────────────────────────────────────────────
        invested = sum(pos["shares"] * pos.get("curr_price", pos["entry_price"])
                       for pos in portfolio.values())
        passive_now = sum(p["pos"]["shares"] * p["curr_price"] for p in passive_list)
        total_now = cash + invested + passive_now
        p_str = " | " + " | ".join(
            f"{p['label']} {inr(p['pos']['shares'] * p['curr_price'])} ({p['pos']['shares']}sh)"
            for p in passive_list
        ) if passive_list else ""
        print(f"\n  AFTER: Invested {inr(invested)} | Cash {inr(cash)} | "
              f"Total {inr(total_now)} | Positions {len(portfolio)}/{MAX_SLOTS} | Slot {inr(per_slot)}{p_str}")

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

    # Passive ETF final values
    passive_final_total = 0.0
    for p in passive_list:
        if p["pos"]["shares"] > 0 and p["iloc"]:
            last_ci = max(p["iloc"].values())
            p_last_price = float('nan')
            for _off in range(0, 10):
                v = float(p["raw"]["Close"].iloc[last_ci - _off])
                if v == v:
                    p_last_price = v
                    break
            if p_last_price == p_last_price:
                p_final_val = p["pos"]["shares"] * p_last_price
                p_final_pnl = p_final_val - p["pos"]["shares"] * p["pos"]["avg_cost"]
                passive_final_total += p_final_val
                print(f"  {p['label']:<12}: {p['pos']['shares']} sh @ avg {p['pos']['avg_cost']:,.1f}"
                      f" | last ₹{p_last_price:,.1f} | value {inr(p_final_val)}"
                      f" | unreal P&L {inr(p_final_pnl)} | charges {inr(p['charges'])}")

    final_value  = cash + passive_final_total + sum(
        pos["shares"] * (float(stock_data[t]["Close"].iloc[max(date_to_iloc[t].values())])
                         if t in date_to_iloc else pos["entry_price"])
        for t, pos in portfolio.items()
    )
    _total_invested = _starting_cash + _sip_total
    total_return = (final_value - _total_invested) / _total_invested * 100

    # Annualised return: XIRR when SIP is active, simple CAGR otherwise
    years = (trading_days[-1] - trading_days[0]).days / 365.25
    if sip > 0 and _xirr_flows:
        _xirr_flows.append((trading_days[-1], final_value))
        _xirr_rate = xirr(_xirr_flows)
        cagr = _xirr_rate * 100
        cagr_label = "XIRR"
    else:
        cagr = ((final_value / _starting_cash) ** (1/years) - 1) * 100 if years > 0 else 0
        cagr_label = "CAGR"

    print()
    print("=" * 72)
    print("  FINAL SUMMARY")
    print("=" * 72)
    print(f"  Period        : {trading_days[0]} → {trading_days[-1]}  ({years:.1f} years)")
    print(f"  Starting Cap  : {inr(_starting_cash)}")
    if sip > 0:
        print(f"  SIP           : {inr(sip)}/mo × {len(rebal_dates)} rebals = {inr(_sip_total)} total")
        print(f"  Total Invested: {inr(_total_invested)}")
    print(f"  Final Value   : {inr(final_value)}")
    print(f"  Total Return  : {pct(total_return)}  (on total invested)")
    print(f"  {cagr_label:<14}: {pct(cagr)}")
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

    # Per-year returns
    print()
    if sip > 0:
        print("  YEAR-BY-YEAR (Time-Weighted Return — strips SIP distortion):")
        # Chain sub-period TWR returns within each calendar year
        year_twr = {}
        for d, r in _twr_data:
            yr = d.year
            year_twr[yr] = year_twr.get(yr, 1.0) * (1 + r)
        year_annual = {yr: (v - 1) * 100 for yr, v in year_twr.items()}
    else:
        print("  YEAR-BY-YEAR:")
        year_last_nav = {}
        for row in rebal_nav:
            yr = row["date"].year
            year_last_nav[yr] = row["nav"]
        initial_nav = 10_00_000.0 if overflow else 20_00_000.0
        prev_nav_yr = initial_nav
        year_annual = {}
        for yr in sorted(year_last_nav):
            end_nav = year_last_nav[yr]
            year_annual[yr] = (end_nav / prev_nav_yr - 1) * 100
            prev_nav_yr = end_nav
    neg_years = 0
    for yr in sorted(year_annual):
        ret_y = year_annual[yr]
        if ret_y < 0:
            neg_years += 1
        bar_len = int(abs(ret_y) / 1)
        bar = ("█" * min(bar_len, 40)) if ret_y > 0 else ("░" * min(bar_len, 40))
        sign = "+" if ret_y >= 0 else "-"
        print(f"  {yr}  {sign}{abs(ret_y):5.1f}%  {bar}")

    print()

    # Export rebalance-date NAV series for portfolio correlation analysis
    if rebal_nav:
        import os as _os
        csv_name = "overflow_rebal.csv" if overflow else ("qqq_rebal.csv" if qqq else ("sp500_rebal.csv" if sp500 else ("mom500_rebal.csv" if n500 else ("mom20_rebal.csv" if mom20 else "mom15_rebal.csv"))))
        nav_csv = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), csv_name)
        pd.DataFrame(rebal_nav).to_csv(nav_csv, index=False)
        print(f"  Rebalance NAV exported → {csv_name} ({len(rebal_nav)} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mom15 / Mom20 PIT Backtest Report")
    parser.add_argument("--refresh", action="store_true", help="Re-download price data")
    parser.add_argument("--mom20", action="store_true",
                        help="Run Mom20 variant (top 20, monthly rebalance, β≤1.2)")
    parser.add_argument("--overflow", action="store_true",
                        help="Run Overflow variant (5-slot, β>1.2, monthly rebalance)")
    parser.add_argument("--no-regime", action="store_true",
                        help="Disable regime filter (allow entries even when Nifty200 < SMA200)")
    parser.add_argument("--beta-cap", type=float, default=None,
                        help="Override beta cap (e.g. 1.35)")
    parser.add_argument("--regime-exit", action="store_true",
                        help="Regime OFF: block entries but still execute exits")
    parser.add_argument("--n500", action="store_true",
                        help="Run Mom500 variant (Nifty500 universe, top 20, monthly, β≤1.2, Nifty500 regime)")
    parser.add_argument("--sp500", action="store_true",
                        help="Run on S&P 500 PIT universe (monthly, no beta filter)")
    parser.add_argument("--qqq", action="store_true",
                        help="Run MomQQQ variant (Nasdaq-100 universe, top 20, monthly, β≤1.2 vs NDX, QQQ regime)")
    parser.add_argument("--start", default=None,
                        help="Override start date (YYYY-MM-DD), e.g. --start 2025-01-01")
    parser.add_argument("--top-n", type=int, default=None,
                        help="Override top_n / number of slots (default: variant-determined)")
    parser.add_argument("--buffer-in", type=int, default=None,
                        help="Override entry buffer (rank ≤ N qualifies for entry)")
    parser.add_argument("--buffer-out", type=int, default=None,
                        help="Override exit buffer (rank > N triggers exit)")
    parser.add_argument("--ema200-exit", action="store_true",
                        help="Exit a holding if close < EMA(200) at rebalance")
    parser.add_argument("--rebal-day", choices=["start", "mid"], default="start",
                        help="Rebalance day-of-month: 'start' (1st trading day) or 'mid' (15th or next trading day)")
    parser.add_argument("--regime", choices=["none", "sma200", "ema200"], default="sma200",
                        help="Regime filter on the benchmark: 'none' (no filter), 'sma200' (default — Nifty200 < SMA200 → off), 'ema200' (uses EMA(200) instead)")
    parser.add_argument("--parabolic-filter", action="store_true",
                        help="Skip new entries where Ret12m > 300%% AND Ret3m/Ret12m > 0.5 (blowoff top)")
    parser.add_argument("--skip-extended", type=float, default=None, metavar="PCT",
                        help="Skip new entries where Ext%%EMA20 > PCT at rebalance date (e.g. --skip-extended 12)")
    parser.add_argument("--sector-cap", type=int, default=None,
                        help="Max holdings per sector (e.g. --sector-cap 5 limits each sector to 5 stocks)")
    parser.add_argument("--addv-min", type=float, default=None,
                        help="Min 90-day median ADDV in ₹ Cr (e.g. --addv-min 10 for ₹10 Cr/day)")
    parser.add_argument("--niftybees", action="store_true",
                        help="Always hold 5%% of portfolio in NIFTYBEES (passive ETF)")
    parser.add_argument("--goldbees", action="store_true",
                        help="Always hold 5%% of portfolio in GOLDBEES (gold ETF)")
    parser.add_argument("--sip", type=float, default=0,
                        help="Inject this amount (₹) of fresh capital at every rebalance (e.g. --sip 100000 for ₹1L/mo)")
    args = parser.parse_args()
    # `--no-regime` (legacy) takes precedence and forces 'none'.
    regime_filter = "none" if args.no_regime else args.regime
    addv_min = args.addv_min * 1e7 if args.addv_min is not None else None  # Cr → INR
    run(refresh=args.refresh, mom20=args.mom20, overflow=args.overflow,
        use_regime=(regime_filter != "none"), beta_cap_override=args.beta_cap,
        regime_exit=args.regime_exit, n500=args.n500, qqq=args.qqq, sp500=args.sp500,
        start_override=args.start,
        top_n_override=args.top_n, buffer_in_override=args.buffer_in,
        buffer_out_override=args.buffer_out, ema200_exit=args.ema200_exit,
        rebal_day=args.rebal_day, regime_filter=regime_filter,
        parabolic_filter=args.parabolic_filter,
        sector_cap=args.sector_cap, addv_min=addv_min,
        niftybees=args.niftybees, goldbees=args.goldbees, sip=args.sip,
        skip_extended=args.skip_extended)
