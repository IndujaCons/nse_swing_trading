#!/usr/bin/env python3
"""
ETF Mom — Monthly Z-Score Backtest
===================================
Mom20 engine applied to the ETF Core universe.
Capital  : ₹10L, 5 slots × ₹2L (NAV/5 compounding)
Rebalance: 1st trading day of each month
Scoring  : MR_12 (50%) + MR_3 (50%), Z-scored cross-sectionally, Normalised
Filters  : Beta ≤ 1.2 vs Nifty 50
Buffer   : Existing stays if rank ≤ 10, new enters if rank ≤ 5
No event-driven entries or exits — pure monthly rebalance.
LIQUIDBEES (6.5% p.a.) on idle slots.

Usage:
    python3 data/etf_core_zscore_backtest.py
    python3 data/etf_core_zscore_backtest.py --refresh
    python3 data/etf_core_zscore_backtest.py --start 2019-01-01
"""

import os, sys, pickle, argparse, warnings
from datetime import date, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
START_DATE     = date(2016, 1, 1)
MAX_SLOTS      = 5
BUFFER_IN      = 5       # enter if rank ≤ 5
BUFFER_OUT     = 10      # hold if rank ≤ 10
BETA_CAP       = 1.2
W12, W3        = 0.50, 0.50
LONG_PD        = 252     # 12m
SHORT_PD       = 63      # 3m
WARMUP_DAYS    = 450
CAPITAL        = 10_000_000   # ₹10L in paise — easier: use ₹ units directly
SLOT_START     = 200_000      # ₹2L per slot
LIQUIDBEES_PA  = 0.065

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, 'cache', 'etf_zscore_daily.pkl')

# ── ETF UNIVERSE ──────────────────────────────────────────────────────────────
# (sym, name, yf_ticker) — same as etf_core_backtest.py, Indian-listed only
# International ETFs excluded from monthly rebalance (no intraday pricing in INR)
UNIVERSE = [
    ("JUNIORBEES",  "Nifty Next 50",        "^NSEMDCP50"),
    ("MID150BEES",  "Nifty Midcap 150",     "MID150BEES.NS"),
    ("HDFCSMALL",   "Nifty Smallcap 250",   "HDFCSML250.NS"),
    ("BANKBEES",    "Nifty Bank",           "^NSEBANK"),
    ("PSUBNKBEES",  "Nifty PSU Bank",       "^CNXPSUBANK"),
    ("ITBEES",      "Nifty IT",             "^CNXIT"),
    ("HEALTHIETF",  "Nifty Healthcare",     "HEALTHIETF.NS"),
    ("AUTOBEES",    "Nifty Auto",           "^CNXAUTO"),
    ("METALIETF",   "Nifty Metal",          "^CNXMETAL"),
    ("CONSUMBEES",  "Nifty Consumption",    "CONSUMBEES.NS"),
    ("INFRABEES",   "Nifty Infra",          "^CNXINFRA"),
    ("OILIETF",     "Nifty Oil & Gas",      "OILIETF.NS"),
    ("MOREALTY",    "Nifty Realty",         "^CNXREALTY"),
    ("CPSEETF",     "Nifty CPSE",           "CPSEETF.NS"),
    ("SETFMOMET",   "Nifty200 Mom30",       "MOM30IETF.NS"),
    ("QUAL30IETF",  "Nifty100 Quality30",   "QUAL30IETF.NS"),
    ("KOTAKLOWV",   "Nifty100 LowVol30",    "LOWVOLIETF.NS"),
    ("ALPL30IETF",  "Nifty Alpha LowVol30", "ALPL30IETF.NS"),
    ("MODEFENCE",   "Nifty India Defence",  "MODEFENCE.NS"),
    ("MOMOMENTUM",  "Nifty500 Mom50",       "MOMOMENTUM.NS"),
    ("MONIFTY500",  "Nifty 500",            "MONIFTY500.NS"),
    ("GOLDBEES",    "Gold",                 "GOLDBEES.NS"),
    ("SILVERBEES",  "Silver",               "SILVERBEES.NS"),
    ("SETFNN50",    "Nifty Next 50 ETF",    "SETFNN50.NS"),
]

BENCH_SYM = "^NSEI"   # Nifty 50 for beta computation

# ── CHARGES ───────────────────────────────────────────────────────────────────
def calc_charges(buy_val, sell_val):
    T     = buy_val + sell_val
    stt   = 0.001     * T
    exch  = 0.0000307 * T
    sebi  = 0.000001  * T
    stamp = 0.00015   * buy_val
    gst   = 0.18      * (exch + sebi)
    return stt + exch + sebi + stamp + gst

# ── DATA FETCH ────────────────────────────────────────────────────────────────
def fetch_all(refresh=False):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    if not refresh and os.path.exists(CACHE_FILE):
        print(f"Loading cache from {CACHE_FILE}...")
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)

    fetch_start = (pd.Timestamp(START_DATE) - pd.Timedelta(days=WARMUP_DAYS)).strftime('%Y-%m-%d')
    fetch_end   = date.today().isoformat()

    print(f"Fetching Nifty 50 benchmark...")
    bench_df = yf.download(BENCH_SYM, start=fetch_start, end=fetch_end,
                           progress=False, auto_adjust=True)
    bench = bench_df["Close"].squeeze().dropna()
    bench.index = pd.to_datetime(bench.index).tz_localize(None)
    print(f"  Bench: {len(bench)} rows")

    closes = {}
    for sym, name, yf_sym in UNIVERSE:
        df = yf.download(yf_sym, start=fetch_start, end=fetch_end,
                         progress=False, auto_adjust=True)
        if df.empty:
            print(f"  WARN: no data for {sym} ({yf_sym})")
            closes[sym] = pd.Series(dtype=float)
            continue
        s = df["Close"].squeeze().dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        closes[sym] = s
        print(f"  {sym:14s} {yf_sym:30s} {len(s)} rows")

    data = (closes, bench)
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(data, f)
    print(f"Cached to {CACHE_FILE}")
    return data

# ── REBALANCE DATES ───────────────────────────────────────────────────────────
def get_rebal_dates(trading_days):
    """First trading day of each month."""
    dates = []
    seen_months = set()
    for d in sorted(trading_days):
        key = (d.year, d.month)
        if key not in seen_months and d >= pd.Timestamp(START_DATE):
            seen_months.add(key)
            dates.append(d)
    return dates

# ── MAIN BACKTEST ─────────────────────────────────────────────────────────────
def run(refresh=False, use_regime=False, start_override=None):
    global START_DATE
    if start_override:
        START_DATE = date.fromisoformat(start_override)

    closes, bench = fetch_all(refresh=refresh)

    # Build aligned daily price matrix
    all_syms  = [sym for sym, _, _ in UNIVERSE if sym in closes and len(closes[sym]) >= LONG_PD + 10]
    all_dates = sorted(set.union(*[set(closes[sym].index) for sym in all_syms]))
    all_dates = [d for d in all_dates if d >= pd.Timestamp(START_DATE) - pd.Timedelta(days=WARMUP_DAYS)]

    print(f"Universe: {len(all_syms)} ETFs  |  {all_dates[0].date()} → {all_dates[-1].date()}")

    # Pre-compute indicators per symbol
    price = {}     # sym → pd.Series (full history)
    ret12 = {}     # 12m return
    ret3  = {}     # 3m return
    vol   = {}     # annualised vol
    beta  = {}     # 252d rolling beta vs Nifty 50

    bench_ret = bench.pct_change()

    for sym in all_syms:
        s = closes[sym]
        dr = s.pct_change()
        price[sym] = s
        ret12[sym] = s / s.shift(LONG_PD) - 1
        ret3[sym]  = s / s.shift(SHORT_PD) - 1
        ann_vol    = dr.rolling(LONG_PD, min_periods=126).std() * (LONG_PD ** 0.5)
        vol[sym]   = ann_vol

        # Rolling 252-day beta vs Nifty 50
        bench_aligned = bench_ret.reindex(s.index, method='ffill').fillna(0)
        cov_series = dr.rolling(LONG_PD, min_periods=126).cov(bench_aligned)
        var_series = bench_aligned.rolling(LONG_PD, min_periods=126).var()
        beta[sym]  = cov_series / var_series.replace(0, np.nan)

    # Date → integer index maps for fast lookup
    date_to_idx = {d: i for i, d in enumerate(all_syms)}

    trading_days = sorted(d for d in all_dates if d >= pd.Timestamp(START_DATE))
    rebal_dates  = get_rebal_dates(trading_days)

    # Nifty 200 for regime filter (^CNX200)
    n200_raw = pd.Series(dtype=float)
    if use_regime:
        try:
            df200 = yf.download("^CNX200",
                                start=(pd.Timestamp(START_DATE) - pd.Timedelta(days=WARMUP_DAYS)).strftime('%Y-%m-%d'),
                                end=date.today().isoformat(), progress=False, auto_adjust=True)
            n200_raw = df200["Close"].squeeze().dropna()
            n200_raw.index = pd.to_datetime(n200_raw.index).tz_localize(None)
            n200_raw = n200_raw.to_frame("Close")
            n200_raw["sma200"] = n200_raw["Close"].rolling(200, min_periods=150).mean()
            n200_iloc = {d.date(): i for i, d in enumerate(n200_raw.index)}
            print(f"Regime data: {len(n200_raw)} rows")
        except Exception as e:
            print(f"Regime data unavailable: {e}")
            use_regime = False

    # ── Portfolio state ───────────────────────────────────────────────────────
    portfolio     = {}    # sym → {entry_price, entry_date, shares, slot_capital}
    all_trades    = []
    total_charges = 0.0
    rebal_nav     = []
    liquidbees_income = 0.0

    slot_capital  = SLOT_START                 # grows with NAV/5 compounding
    cash          = float(SLOT_START * MAX_SLOTS)   # ₹10L initial capital

    def portfolio_nav(rebal_day):
        """Current portfolio NAV at a given date."""
        nav = cash
        for sym, pos in portfolio.items():
            px = _price_on(sym, rebal_day, pos["entry_price"])
            nav += px / pos["entry_price"] * pos["slot_capital"]
        return nav

    def _price_on(sym, day, fallback):
        s = price.get(sym)
        if s is None:
            return fallback
        idx = s.index.searchsorted(day)
        if idx >= len(s):
            idx = len(s) - 1
        # find closest prior date
        while idx > 0 and s.index[idx] > day:
            idx -= 1
        if idx < 0 or s.index[idx] > day:
            return fallback
        return float(s.iloc[idx])

    def _indicator_on(series_dict, sym, day):
        s = series_dict.get(sym)
        if s is None:
            return None
        idx = s.index.searchsorted(day)
        if idx >= len(s):
            idx = len(s) - 1
        while idx > 0 and s.index[idx] > day:
            idx -= 1
        if idx < 0 or pd.isna(s.iloc[idx]):
            return None
        return float(s.iloc[idx])

    print(f"\n=== ETF MOM — Monthly Z-Score | β≤{BETA_CAP} | {MAX_SLOTS} slots ===")
    print(f"{'Rebal':12s}  {'NAV':>12s}  {'Port':50s}")

    for rebal_idx, rebal_day in enumerate(rebal_dates):
        nav = portfolio_nav(rebal_day)

        # Update slot capital (NAV / MAX_SLOTS, no compounding cap for ETFs)
        new_sc = nav / MAX_SLOTS if nav > SLOT_START * MAX_SLOTS else SLOT_START
        slot_capital = max(SLOT_START, new_sc)

        # LIQUIDBEES income on idle slots since last rebal
        days_since = (rebal_day - rebal_dates[rebal_idx - 1]).days if rebal_idx > 0 else 0
        idle_slots = MAX_SLOTS - len(portfolio)
        lb_income  = idle_slots * slot_capital * LIQUIDBEES_PA * days_since / 365
        cash += lb_income
        liquidbees_income += lb_income

        # ── Regime check ─────────────────────────────────────────────────────
        regime_off = False
        if use_regime and not n200_raw.empty:
            ri = n200_iloc.get(rebal_day.date())
            if ri is None:
                for off in range(1, 6):
                    ri = n200_iloc.get((rebal_day - pd.Timedelta(days=off)).date())
                    if ri is not None:
                        break
            if ri is not None:
                nc = float(n200_raw["Close"].iloc[ri])
                ns = n200_raw["sma200"].iloc[ri]
                if not pd.isna(ns) and nc < float(ns):
                    regime_off = True

        # ── Score all ETFs ────────────────────────────────────────────────────
        scores = {}
        mr12_vals, mr3_vals = {}, {}
        for sym in all_syms:
            r12 = _indicator_on(ret12, sym, rebal_day)
            r3  = _indicator_on(ret3,  sym, rebal_day)
            v   = _indicator_on(vol,   sym, rebal_day)
            b   = _indicator_on(beta,  sym, rebal_day)
            if None in (r12, r3, v) or v == 0:
                continue
            if b is not None and abs(b) > BETA_CAP:
                continue   # beta filter
            mr12_vals[sym] = r12 / v
            mr3_vals[sym]  = r3  / v

        eligible = [s for s in mr12_vals if s in mr3_vals]
        if len(eligible) >= 3:
            arr12 = np.array([mr12_vals[s] for s in eligible])
            arr3  = np.array([mr3_vals[s]  for s in eligible])
            def zscore(a):
                mu, sd = a.mean(), a.std()
                return (a - mu) / sd if sd > 0 else np.zeros_like(a)
            z12 = zscore(arr12)
            z3  = zscore(arr3)
            waz = W12 * z12 + W3 * z3
            sc  = np.where(waz >= 0, 1 + waz, 1.0 / (1 - waz))
            for i, sym in enumerate(eligible):
                scores[sym] = sc[i]

        ranked = sorted(scores, key=lambda s: scores[s], reverse=True)
        rank_map = {sym: i + 1 for i, sym in enumerate(ranked)}

        # ── Determine holds and exits ─────────────────────────────────────────
        current_set = set(portfolio.keys())
        # Keep if rank ≤ BUFFER_OUT and not regime_off
        if regime_off:
            to_sell = set()   # hold everything during regime-off
        else:
            to_sell = {sym for sym in current_set
                       if rank_map.get(sym, 9999) > BUFFER_OUT or sym not in scores}

        exit_rows = []
        for sym in sorted(to_sell):
            pos      = portfolio[sym]
            ep       = _price_on(sym, rebal_day, pos["entry_price"])
            sc_      = pos["slot_capital"]
            sell_val = sc_ * ep / pos["entry_price"]
            gp       = sell_val - sc_
            chrg     = calc_charges(sc_, sell_val)
            total_charges += chrg
            pnl  = gp - chrg
            cash += sell_val
            all_trades.append({
                "sym": sym, "entry": pos["entry_date"], "exit": rebal_day.date(),
                "entry_px": pos["entry_price"], "exit_px": ep,
                "gross_pnl": gp, "net_pnl": pnl, "charges": chrg,
                "hold_days": (rebal_day.date() - pos["entry_date"]).days,
            })
            exit_rows.append(f"{sym}({gp/sc_*100:+.0f}%)")
            del portfolio[sym]

        # ── Entries ───────────────────────────────────────────────────────────
        if not regime_off:
            slots_free = MAX_SLOTS - len(portfolio)
            candidates = [s for s in ranked
                          if s not in portfolio
                          and rank_map.get(s, 9999) <= BUFFER_IN
                          and slots_free > 0]
            entry_rows = []
            for sym in candidates[:slots_free]:
                ep  = _price_on(sym, rebal_day, None)
                if ep is None:
                    continue
                sc_ = slot_capital
                shares = sc_ / ep
                cash -= sc_
                portfolio[sym] = {
                    "entry_price": ep,
                    "entry_date":  rebal_day.date(),
                    "shares":      shares,
                    "slot_capital": sc_,
                }
                entry_rows.append(sym)
                slots_free -= 1
        else:
            entry_rows = []

        # ── Log ───────────────────────────────────────────────────────────────
        nav_after = portfolio_nav(rebal_day)
        port_str  = " ".join(
            f"{s}#{rank_map.get(s,'?')}" for s in sorted(portfolio)
        ) or "(empty)"
        regime_tag = " [REGIME OFF]" if regime_off else ""
        print(f"{rebal_day.date()}  ₹{nav_after/1e5:>8.2f}L  {port_str}{regime_tag}")
        if exit_rows:
            print(f"  EXIT : {' '.join(exit_rows)}")
        if entry_rows:
            print(f"  ENTRY: {' '.join(entry_rows)}")

        rebal_nav.append({"date": rebal_day.date(), "nav": nav_after})

    # ── Close all open positions at last price ────────────────────────────────
    last_day = rebal_dates[-1] if rebal_dates else pd.Timestamp(date.today())
    for sym, pos in list(portfolio.items()):
        ep       = _price_on(sym, last_day, pos["entry_price"])
        sc_      = pos["slot_capital"]
        sell_val = sc_ * ep / pos["entry_price"]
        gp       = sell_val - sc_
        chrg     = calc_charges(sc_, sell_val)
        total_charges += chrg
        all_trades.append({
            "sym": sym, "entry": pos["entry_date"], "exit": last_day.date(),
            "entry_px": pos["entry_price"], "exit_px": ep,
            "gross_pnl": gp, "net_pnl": gp - chrg, "charges": chrg,
            "hold_days": (last_day.date() - pos["entry_date"]).days,
            "open": True,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    closed    = [t for t in all_trades if not t.get("open")]
    open_pos  = [t for t in all_trades if t.get("open")]
    winners   = [t for t in closed if t["gross_pnl"] > 0]
    losers    = [t for t in closed if t["gross_pnl"] <= 0]
    gross_win = sum(t["gross_pnl"] for t in winners)
    gross_los = abs(sum(t["gross_pnl"] for t in losers))
    pf        = gross_win / gross_los if gross_los > 0 else float('inf')
    wr        = len(winners) / len(closed) * 100 if closed else 0

    total_net = sum(t["net_pnl"] for t in all_trades) + liquidbees_income
    initial   = SLOT_START * MAX_SLOTS
    n_years   = (last_day.date() - START_DATE).days / 365.25
    cagr      = (1 + total_net / initial) ** (1 / n_years) - 1 if n_years > 0 else 0

    avg_hold  = np.mean([t["hold_days"] for t in closed]) if closed else 0

    # Year-by-year from rebal_nav
    year_navs = {}
    for row in rebal_nav:
        yr = row["date"].year
        year_navs.setdefault(yr, []).append(row["nav"])

    print(f"\n{'='*72}")
    print(f"  ETF MOM Z-SCORE  |  Monthly  |  β≤{BETA_CAP}  |  {MAX_SLOTS} slots")
    print(f"{'='*72}")
    print(f"  Period        : {START_DATE} → {last_day.date()}")
    print(f"  Initial capital: ₹{initial/1e5:.1f}L  ({MAX_SLOTS} slots × ₹{SLOT_START/1e5:.1f}L)")
    print()

    prev_nav = initial
    neg_years = 0
    for yr in sorted(year_navs):
        end_nav = year_navs[yr][-1]
        ret = (end_nav / prev_nav - 1) * 100
        tag = " ◄ NEG" if ret < 0 else ""
        if ret < 0:
            neg_years += 1
        print(f"  {yr}: {ret:+6.1f}%  (NAV ₹{end_nav/1e5:.2f}L){tag}")
        prev_nav = end_nav

    final_nav = rebal_nav[-1]["nav"] if rebal_nav else initial
    total_ret = (final_nav - initial) / initial * 100

    print()
    print(f"  Total Return  : {total_ret:+.1f}%")
    print(f"  CAGR          : {cagr*100:+.1f}%")
    print(f"  Neg years     : {neg_years}")
    print()
    print(f"  Closed Trades : {len(closed)}  |  Open: {len(open_pos)}")
    print(f"  Win Rate      : {wr:.1f}%  ({len(winners)}W / {len(losers)}L)")
    print(f"  Profit Factor : {pf:.2f}")
    print(f"  Avg hold      : {avg_hold:.0f} days")
    print(f"  Total charges : ₹{total_charges:,.0f}")
    print(f"  LIQUIDBEES    : ₹{liquidbees_income:,.0f}  ({LIQUIDBEES_PA*100:.1f}% p.a. on idle slots)")
    print(f"  Net P&L       : ₹{total_net:+,.0f}")
    print(f"  Final NAV     : ₹{final_nav:,.0f}  (₹{final_nav/1e5:.2f}L)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh",    action="store_true", help="Re-download data")
    parser.add_argument("--regime",     action="store_true", help="Enable regime filter (Nifty200 < SMA200 → hold all)")
    parser.add_argument("--start",      default=None,        help="Override start date (YYYY-MM-DD)")
    args = parser.parse_args()
    run(refresh=args.refresh, use_regime=args.regime, start_override=args.start)
