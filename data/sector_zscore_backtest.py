#!/usr/bin/env python3
"""
Sector Z-Score Backtest — V3 Phase 1
=====================================
Top-5 NSE sectoral indices by Z-score, monthly rebalance, equal-weight 20%.
Mirrors the ETF Core scoring math (data/etf_core_zscore_backtest.py:344-376)
but with a dynamic sectoral-index universe (per-rebalance eligibility — new
indices enter the eligible pool as their data accrues).

Pre-registered pass criteria (V3 Phase 1 gate):
  * CAGR vs Nifty 200            ≥ +4 pp
  * Sharpe                        ≥ 0.85
  * Information Ratio vs Nifty 200 ≥ 0.4
  * MaxDD vs Nifty 200           ≤ +5 pp wider
  * Worst monthly relative return ≥ -8 pp

If ≥3 of 5 fail, the gate fails and Layer 2 (stock picking inside chosen
sectors) is not pursued.

Usage:
    python3 data/sector_zscore_backtest.py
    python3 data/sector_zscore_backtest.py --refresh
    python3 data/sector_zscore_backtest.py --start 2018-01-01
    python3 data/sector_zscore_backtest.py --no-liquidbees
"""

import os, sys, io, pickle, argparse, warnings
from datetime import date, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')


class _Tee:
    """Write to multiple output streams simultaneously."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
    def flush(self):
        for st in self.streams:
            st.flush()

# ── CONFIG ────────────────────────────────────────────────────────────────────
EARLIEST_START = date(2010, 1, 1)   # don't probe data older than this
MAX_SLOTS      = 5
BUFFER_IN      = 5       # enter if rank ≤ 5 (top 5)
BUFFER_OUT     = 7       # hold if rank ≤ 7  (exit when rank ≥ 8 — 2-rank hysteresis)
W12, W3        = 0.50, 0.50
LONG_PD        = 252     # 12m
SHORT_PD       = 63      # 3m  (2m tested, made signal noisier — see commit log)
MIN_HISTORY    = LONG_PD + 10        # eligibility threshold per index
MIN_ELIGIBLE   = 6                   # backtest start = ≥ this many sectors eligible
WARMUP_DAYS    = 450
SLOT_START     = 200_000             # ₹2L per slot — total ₹10L initial
LIQUIDBEES_PA  = 0.065

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR  = os.path.join(BASE_DIR, 'cache')

# ── SECTOR UNIVERSE ───────────────────────────────────────────────────────────
# Merge of existing SECTORAL_INDICES from data/live_signals_engine.py and the
# spec's additions. Tentatives marked — script silently drops any symbol that
# returns < MIN_HISTORY days at probe.
SECTOR_UNIVERSE = [
    # 20-symbol universe — Phase 1 widened to match Phase 2 sector map.
    # Sectors with no yfinance time series fall back to synthetic indices
    # built from member stocks (data/synth_sector_index.py).
    ("NIFTY BANK",              "^NSEBANK"),
    ("NIFTY PVT BANK",          "NIFTY_PVT_BANK.NS"),
    ("NIFTY PSU BANK",          "^CNXPSUBANK"),
    ("NIFTY FIN SERVICE",       "NIFTY_FIN_SERVICE.NS"),
    ("NIFTY IT",                "^CNXIT"),
    ("NIFTY AUTO",              "^CNXAUTO"),
    ("NIFTY METAL",             "^CNXMETAL"),
    ("NIFTY ENERGY",            "^CNXENERGY"),
    ("NIFTY FMCG",              "^CNXFMCG"),
    ("NIFTY REALTY",            "^CNXREALTY"),
    ("NIFTY MEDIA",             "^CNXMEDIA"),
    ("NIFTY MNC",               "^CNXMNC"),
    ("NIFTY PSE",               "^CNXPSE"),
    # Re-added from Phase 2 sector map for wider breadth
    ("NIFTY CONSUMPTION",       "^CNXCONSUM"),
    ("NIFTY INFRA",             "^CNXINFRA"),
    # PHARMA dropped — strict subset of HEALTHCARE in N200
    # Synthetic indices (built via data/synth_sector_index.py — no yfinance series)
    ("NIFTY HEALTHCARE",        "synth:NIFTY_HEALTHCARE"),
    ("NIFTY OIL & GAS",         "synth:NIFTY_OIL_GAS"),
    ("NIFTY INDIA DEFENCE",     "synth:NIFTY_INDIA_DEFENCE"),
    ("NIFTY INDIA MFG",         "synth:NIFTY_INDIA_MFG"),
    ("NIFTY CONSUMER DURABLES", "synth:NIFTY_CONSUMER_DURABLES"),
]

BENCH_SYM = "^CNX200"   # Nifty 200 — both benchmark and (unused) beta reference

# ── CHARGES (Zerodha equity, mirrored from etf_core_zscore_backtest.py:104) ───
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
    """Fetch sectoral closes + Nifty 200 benchmark.
    Cache: data/cache/sector_zscore_daily_<YYYY-MM-DD>.pkl
    Date-stamped so a re-run on a new day creates a fresh snapshot
    while older snapshots remain pinned for reproducibility."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    today_str = date.today().isoformat()
    cache_file = os.path.join(CACHE_DIR, f'sector_zscore_daily_{today_str}.pkl')

    if not refresh and os.path.exists(cache_file):
        print(f"Loading cache from {cache_file}...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    fetch_start = (pd.Timestamp(EARLIEST_START) - pd.Timedelta(days=WARMUP_DAYS)).strftime('%Y-%m-%d')
    fetch_end   = date.today().isoformat()

    print(f"Fetching Nifty 200 benchmark...")
    bench_df = yf.download(BENCH_SYM, start=fetch_start, end=fetch_end,
                           progress=False, auto_adjust=True)
    if bench_df.empty:
        raise RuntimeError(f"Could not fetch Nifty 200 ({BENCH_SYM}) — backtest cannot proceed.")
    bench = bench_df["Close"].squeeze().dropna()
    bench.index = pd.to_datetime(bench.index).tz_localize(None)
    print(f"  Bench: {len(bench)} rows  ({bench.index[0].date()} → {bench.index[-1].date()})")

    closes, opens = {}, {}
    print(f"\nFetching {len(SECTOR_UNIVERSE)} sectoral indices...")
    for sym, source in SECTOR_UNIVERSE:
        # Synthetic indices: load latest .pkl from data/cache/ — no Open data,
        # so Open := Close (rebal execution becomes effectively at close).
        if source.startswith("synth:"):
            synth_name = source.split(":", 1)[1].lower()
            import glob
            candidates = sorted(glob.glob(os.path.join(CACHE_DIR, f"synth_{synth_name}_*.pkl")))
            if not candidates:
                print(f"  SKIP {sym:30s} synth:{synth_name:25s} (no .pkl found — run synth_sector_index.py)")
                continue
            with open(candidates[-1], "rb") as f:
                c = pickle.load(f)
            c.index = pd.to_datetime(c.index).tz_localize(None) if c.index.tz else c.index
            closes[sym] = c
            opens[sym]  = c   # synthetic: open ≡ close
            print(f"  SYN  {sym:30s} {os.path.basename(candidates[-1]):35s} {len(c)} rows  "
                  f"({c.index[0].date()} → {c.index[-1].date()})")
            continue
        try:
            df = yf.download(source, start=fetch_start, end=fetch_end,
                             progress=False, auto_adjust=True)
        except Exception as e:
            print(f"  SKIP {sym:30s} {source:30s}  ({e})")
            continue
        if df.empty or len(df) < 30:
            print(f"  SKIP {sym:30s} {source:30s}  (no data / <30 rows)")
            continue
        c = df["Close"].squeeze().dropna()
        o = df["Open"].squeeze().dropna()
        c.index = pd.to_datetime(c.index).tz_localize(None)
        o.index = pd.to_datetime(o.index).tz_localize(None)
        closes[sym] = c
        opens[sym]  = o
        print(f"  OK   {sym:30s} {source:30s}  {len(c)} rows  ({c.index[0].date()} → {c.index[-1].date()})")

    if len(closes) < MIN_ELIGIBLE:
        raise RuntimeError(f"Only {len(closes)} sectors have data — need ≥ {MIN_ELIGIBLE} for backtest.")

    data = (closes, opens, bench)
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"\nCached to {cache_file}")
    return data

# ── REBALANCE DATES (mirrored from etf_core_zscore_backtest.py:156) ───────────
def get_rebal_dates(trading_days, start_date):
    """First trading day of each calendar month, on or after start_date."""
    dates, seen = [], set()
    for d in sorted(trading_days):
        if d < pd.Timestamp(start_date):
            continue
        key = (d.year, d.month)
        if key in seen:
            continue
        seen.add(key)
        dates.append(d)
    return dates

# ── DISPLAY HELPERS ───────────────────────────────────────────────────────────
def inr(v):
    return f"₹{v:>+,.0f}" if v != 0 else "₹0"

def pct(v):
    return f"{v:+.1f}%"

def print_table(headers, rows, col_widths):
    hdr = "  ".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    print("  " + hdr)
    print("  " + "  ".join("─" * w for w in col_widths))
    for row in rows:
        print("  " + "  ".join(str(c).ljust(w) for c, w in zip(row, col_widths)))


# ── METRICS / GATE ────────────────────────────────────────────────────────────
def compute_metrics(rebal_nav, bench_series, start_date, end_date):
    """Compute strategy + benchmark metrics from monthly rebal NAVs.

    Returns dict with: cagr, bench_cagr, sharpe, ir, max_dd, bench_max_dd,
    win_rate_vs_bench, worst_monthly_rel, avg_turnover (set elsewhere).
    """
    if not rebal_nav:
        return None

    nav_series = pd.Series(
        [r["nav"] for r in rebal_nav],
        index=pd.DatetimeIndex([pd.Timestamp(r["date"]) for r in rebal_nav]),
    ).sort_index()

    # Align bench to rebal dates
    bench_at_rebal = bench_series.reindex(nav_series.index, method='ffill').dropna()
    nav_aligned = nav_series.reindex(bench_at_rebal.index)

    if len(nav_aligned) < 3:
        return None

    # Monthly returns
    strat_ret = nav_aligned.pct_change().dropna()
    bench_ret = bench_at_rebal.pct_change().dropna()
    common_idx = strat_ret.index.intersection(bench_ret.index)
    strat_ret = strat_ret.loc[common_idx]
    bench_ret = bench_ret.loc[common_idx]
    active = strat_ret - bench_ret

    n_years = max((end_date - start_date).days / 365.25, 1e-9)

    # CAGR
    initial = nav_series.iloc[0]
    final   = nav_series.iloc[-1]
    cagr = (final / initial) ** (1 / n_years) - 1 if initial > 0 else 0
    bench_cagr = (bench_at_rebal.iloc[-1] / bench_at_rebal.iloc[0]) ** (1 / n_years) - 1

    # Sharpe (monthly), annualised
    sharpe = (strat_ret.mean() / strat_ret.std() * np.sqrt(12)) if strat_ret.std() > 0 else 0

    # Information Ratio (monthly active), annualised
    ir = (active.mean() / active.std() * np.sqrt(12)) if active.std() > 0 else 0

    # MaxDD on monthly NAV series
    def max_dd(s):
        peak = s.cummax()
        return float((s / peak - 1).min())
    strat_dd = max_dd(nav_aligned)
    bench_dd = max_dd(bench_at_rebal)

    # Win rate (% of months strategy beats bench)
    win_rate = float((active > 0).sum() / len(active) * 100) if len(active) else 0

    # Worst monthly relative return
    worst_monthly_rel = float(active.min()) if len(active) else 0

    return {
        "cagr": cagr,
        "bench_cagr": bench_cagr,
        "sharpe": float(sharpe),
        "ir": float(ir),
        "max_dd": strat_dd,
        "bench_max_dd": bench_dd,
        "win_rate_vs_bench": win_rate,
        "worst_monthly_rel": worst_monthly_rel,
    }


def evaluate_gate(metrics, avg_turnover):
    """Print pass-criteria gate verdict. Returns (n_pass, n_fail)."""
    if not metrics:
        print("\n  Gate evaluation skipped — insufficient data.\n")
        return (0, 5)

    cagr_excess  = (metrics["cagr"] - metrics["bench_cagr"]) * 100
    # Both DDs are negative; "widening" = how much DEEPER strat is vs bench.
    # bench_dd=-31.7%, strat_dd=-41.4%  →  widening = (-31.7) - (-41.4) = +9.7pp
    dd_widening  = (metrics["bench_max_dd"] - metrics["max_dd"]) * 100
    worst_rel_pp = metrics["worst_monthly_rel"] * 100

    criteria = [
        ("CAGR vs Nifty 200 (≥ +4pp)",          f"{cagr_excess:+.1f}pp", cagr_excess >= 4.0),
        ("Sharpe (≥ 0.85)",                     f"{metrics['sharpe']:.2f}", metrics["sharpe"] >= 0.85),
        ("Information Ratio (≥ 0.40)",          f"{metrics['ir']:.2f}",     metrics["ir"] >= 0.40),
        ("MaxDD widening vs N200 (≤ +5pp)",     f"{dd_widening:+.1f}pp", dd_widening <= 5.0),
        ("Worst monthly relative (≥ -8pp)",     f"{worst_rel_pp:+.1f}pp", worst_rel_pp >= -8.0),
    ]

    print(f"\n{'='*72}")
    print(f"  V3 Phase 1 — Pass Criteria Gate")
    print(f"{'='*72}")
    n_pass = 0
    for label, value, ok in criteria:
        mark = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {label:<42s}  {value:>10s}     {mark}")
        if ok:
            n_pass += 1
    n_fail = len(criteria) - n_pass

    print(f"  {'─'*70}")
    if n_fail >= 3:
        verdict = f"GATE FAILED — {n_pass}/5 pass — DO NOT proceed to Layer 2"
    else:
        verdict = f"GATE PASSED — {n_pass}/5 pass — proceed to Layer 2 (stock picking within sectors)"
    print(f"  VERDICT: {verdict}")
    print(f"  Avg sector turnover per rebalance: {avg_turnover:.2f} (out of {MAX_SLOTS})")
    print(f"{'='*72}")
    return (n_pass, n_fail)


# ── MAIN BACKTEST ─────────────────────────────────────────────────────────────
def run(refresh=False, start_override=None, no_liquidbees=False):
    closes, opens, bench = fetch_all(refresh=refresh)

    # Filter cache against the *current* SECTOR_UNIVERSE constant — so
    # dropping symbols from the constant takes effect without --refresh.
    defined_syms = {sym for sym, _ in SECTOR_UNIVERSE}
    universe_syms = [sym for sym in closes.keys() if sym in defined_syms]
    dropped_from_cache = [sym for sym in closes.keys() if sym not in defined_syms]
    if dropped_from_cache:
        print(f"  (excluding from cached snapshot: {', '.join(dropped_from_cache)})")
    print(f"\nUniverse: {len(universe_syms)} sectoral indices in scope")

    # All trading days = union of all symbol dates ∪ bench dates
    all_dates = sorted(set.union(
        *(set(closes[sym].index) for sym in universe_syms),
        set(bench.index)
    ))

    # Determine dynamic START_DATE: first month-start where ≥ MIN_ELIGIBLE
    # sectors have ≥ MIN_HISTORY days of history.
    # Build candidate first-trading-days of each month across all_dates
    candidate_starts = []
    seen = set()
    for d in all_dates:
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            candidate_starts.append(d)

    auto_start = None
    for d in candidate_starts:
        eligible = sum(1 for sym in universe_syms
                       if (closes[sym].index <= d).sum() >= MIN_HISTORY)
        if eligible >= MIN_ELIGIBLE:
            auto_start = d
            break

    if start_override:
        START_DATE = date.fromisoformat(start_override)
        print(f"START_DATE override: {START_DATE} ({sum(1 for sym in universe_syms if (closes[sym].index <= pd.Timestamp(START_DATE)).sum() >= MIN_HISTORY)} eligible)")
    elif auto_start is not None:
        START_DATE = auto_start.date()
        print(f"START_DATE auto-detected: {START_DATE} (first month-start with ≥{MIN_ELIGIBLE} eligible sectors)")
    else:
        raise RuntimeError(f"Never reached ≥{MIN_ELIGIBLE} eligible sectors — universe too thin.")

    # Pre-compute indicators per symbol (full history)
    price, open_price = {}, {}
    ret12, ret3, vol, ema200 = {}, {}, {}, {}

    for sym in universe_syms:
        s = closes[sym]
        dr = s.pct_change()
        price[sym]      = s
        open_price[sym] = opens.get(sym, pd.Series(dtype=float))
        ret12[sym] = s / s.shift(LONG_PD) - 1
        ret3[sym]  = s / s.shift(SHORT_PD) - 1
        vol[sym]   = dr.rolling(LONG_PD, min_periods=126).std() * (LONG_PD ** 0.5)
        # Trend filter: EMA(200) — more responsive than SMA(200), reacts
        # quicker to trend breaks but can be slightly whippier.
        ema200[sym] = s.ewm(span=200, adjust=False, min_periods=150).mean()

    trading_days = sorted(d for d in all_dates if d >= pd.Timestamp(START_DATE))
    rebal_dates  = get_rebal_dates(trading_days, START_DATE)

    print(f"Rebalance schedule: {len(rebal_dates)} months  ({rebal_dates[0].date()} → {rebal_dates[-1].date()})")

    # ── Helpers (mirrored from etf_core_zscore_backtest.py) ──────────────────
    def _price_on(sym, day, fallback):
        s = price.get(sym)
        if s is None or s.empty:
            return fallback
        idx = s.index.searchsorted(day)
        if idx >= len(s): idx = len(s) - 1
        while idx > 0 and s.index[idx] > day:
            idx -= 1
        if idx < 0 or s.index[idx] > day:
            return fallback
        return float(s.iloc[idx])

    def _open_on(sym, day, fallback):
        s = open_price.get(sym)
        if s is None or s.empty:
            return fallback
        idx = s.index.searchsorted(day, side='left')
        if idx >= len(s):
            return fallback
        return float(s.iloc[idx])

    def _indicator_on(series_dict, sym, day):
        s = series_dict.get(sym)
        if s is None:
            return None
        idx = s.index.searchsorted(day)
        if idx >= len(s): idx = len(s) - 1
        while idx > 0 and s.index[idx] > day:
            idx -= 1
        if idx < 0 or pd.isna(s.iloc[idx]):
            return None
        return float(s.iloc[idx])

    def eligible_at(d):
        """Sectors with ≥ MIN_HISTORY days of history at date d."""
        return [sym for sym in universe_syms
                if (closes[sym].index <= d).sum() >= MIN_HISTORY]

    # ── Portfolio state ───────────────────────────────────────────────────────
    portfolio     = {}
    all_trades    = []
    total_charges = 0.0
    rebal_nav     = []
    liquidbees_income = 0.0
    turnover_per_rebal = []      # number of entries+exits per rebal
    rebal_log     = []           # structured per-rebal data for the .md report

    slot_capital = SLOT_START
    cash         = float(SLOT_START * MAX_SLOTS)

    def portfolio_nav(d):
        nav = cash
        for sym, pos in portfolio.items():
            px = _price_on(sym, d, pos["entry_price"])
            nav += px / pos["entry_price"] * pos["slot_capital"]
        return nav

    for rebal_idx, rebal_day in enumerate(rebal_dates):
        next_idx = trading_days.index(rebal_day) + 1 if rebal_day in trading_days else None
        exec_day = trading_days[next_idx] if next_idx and next_idx < len(trading_days) else rebal_day + pd.Timedelta(days=1)

        nav = portfolio_nav(rebal_day)
        new_sc = nav / MAX_SLOTS if nav > SLOT_START * MAX_SLOTS else SLOT_START
        slot_capital = max(SLOT_START, new_sc)

        days_since = (rebal_day - rebal_dates[rebal_idx - 1]).days if rebal_idx > 0 else 0
        idle_slots = MAX_SLOTS - len(portfolio)
        if not no_liquidbees:
            lb_income = idle_slots * slot_capital * LIQUIDBEES_PA * days_since / 365
            cash += lb_income
            liquidbees_income += lb_income

        # ── Score eligible sectors  (mirrored from etf_core_zscore_backtest.py:344-376) ──
        elig = eligible_at(rebal_day)
        score_data, mr12_vals, mr3_vals = {}, {}, {}
        for sym in elig:
            r12 = _indicator_on(ret12, sym, rebal_day)
            r3  = _indicator_on(ret3,  sym, rebal_day)
            v   = _indicator_on(vol,   sym, rebal_day)
            if None in (r12, r3, v) or v == 0:
                continue
            mr12_vals[sym] = r12 / v
            mr3_vals[sym]  = r3  / v
            score_data[sym] = {"ret12": r12, "ret3": r3}

        eligible_set = [s for s in mr12_vals if s in mr3_vals]
        if len(eligible_set) >= 3:
            arr12 = np.array([mr12_vals[s] for s in eligible_set])
            arr3  = np.array([mr3_vals[s]  for s in eligible_set])
            def zscore(a):
                mu, sd = a.mean(), a.std()
                return (a - mu) / sd if sd > 0 else np.zeros_like(a)
            z12 = zscore(arr12)
            z3  = zscore(arr3)
            waz = W12 * z12 + W3 * z3
            sc  = np.where(waz >= 0, 1 + waz, 1.0 / (1 - waz))
            for i, sym in enumerate(eligible_set):
                score_data[sym]["score"] = float(sc[i])

        ranked   = sorted((s for s in score_data if "score" in score_data[s]),
                          key=lambda s: score_data[s]["score"], reverse=True)
        rank_map = {sym: i + 1 for i, sym in enumerate(ranked)}

        # ── Header ───────────────────────────────────────────────────────────
        print()
        print("=" * 72)
        print(f"  REBALANCE #{rebal_idx+1:02d}  —  {rebal_day.strftime('%d %b %Y')}  "
              f"(eligible: {len(eligible_set)}/{len(universe_syms)})")
        print(f"  NAV: ₹{nav:,.0f}  |  Slot: ₹{slot_capital:,.0f}  |  Cash: ₹{cash:,.0f}")
        print("=" * 72)

        # ── Exits  (rank slip OR close < EMA200 trend filter) ───────────────
        current = set(portfolio.keys())
        exit_reasons = {}   # sym → "rank>7" / "ema200" / "rank>7+ema200" / "missing"
        for sym in current:
            reasons = []
            rk = rank_map.get(sym, 9999)
            if rk > BUFFER_OUT or sym not in score_data:
                reasons.append("rank>" + str(BUFFER_OUT) if sym in score_data else "missing")
            ema = _indicator_on(ema200, sym, rebal_day)
            px  = _price_on(sym, rebal_day, None)
            if ema is not None and px is not None and px < ema:
                reasons.append("ema200")
            if reasons:
                exit_reasons[sym] = "+".join(reasons)
        to_sell = set(exit_reasons.keys())

        exit_rows = []
        for sym in sorted(to_sell):
            pos      = portfolio[sym]
            ep       = _open_on(sym, exec_day, _price_on(sym, rebal_day, pos["entry_price"]))
            sc_      = pos["slot_capital"]
            sell_val = sc_ * ep / pos["entry_price"]
            gp       = sell_val - sc_
            chrg     = calc_charges(sc_, sell_val)
            total_charges += chrg
            pnl      = gp - chrg
            pnl_pct  = gp / sc_ * 100
            hold_days= (rebal_day.date() - pos["entry_date"]).days
            reason   = exit_reasons[sym]
            cash += sell_val
            all_trades.append({
                "sym": sym, "entry": pos["entry_date"], "exit": rebal_day.date(),
                "entry_px": pos["entry_price"], "exit_px": ep,
                "gross_pnl": gp, "net_pnl": pnl, "charges": chrg, "hold_days": hold_days,
                "reason": reason,
            })
            exit_rows.append((sym, rank_map.get(sym, "—"),
                              pos["entry_date"].strftime("%d-%b-%y"),
                              f"{pos['entry_price']:,.1f}", f"{ep:,.1f}",
                              inr(gp), pct(pnl_pct), f"{hold_days}d", reason))
            del portfolio[sym]

        print(f"\n  EXITS ({len(exit_rows)})")
        if exit_rows:
            print_table(["Sector", "Rank", "Entry", "Entry₹", "Exit₹", "Gross P&L", "P&L%", "Hold", "Reason"],
                        sorted(exit_rows, key=lambda r: float(r[6].replace('+','').replace('%','')), reverse=True),
                        [22, 5, 10, 10, 10, 12, 8, 6, 16])
        else:
            print("    —")

        # ── Entries ──────────────────────────────────────────────────────────
        to_buy = {s for s in ranked if s not in current and rank_map.get(s, 9999) <= BUFFER_IN}
        slots_free = MAX_SLOTS - len(portfolio)

        entry_rows = []
        for sym in sorted(to_buy, key=lambda s: rank_map.get(s, 9999)):
            if slots_free <= 0:
                break
            ep = _open_on(sym, exec_day, _price_on(sym, rebal_day, None))
            if ep is None:
                continue
            sc_ = min(slot_capital, cash) if cash < slot_capital else slot_capital
            if sc_ < 1000:
                continue
            shares = sc_ / ep
            cash  -= sc_
            portfolio[sym] = {
                "entry_price": ep, "entry_date": exec_day.date(),
                "shares": shares, "slot_capital": sc_,
            }
            sd = score_data.get(sym, {})
            entry_rows.append((sym, rank_map[sym], f"{sd.get('score', 0):.3f}",
                               f"{sd.get('ret12', 0)*100:+.1f}%",
                               f"{sd.get('ret3',  0)*100:+.1f}%",
                               f"{ep:,.1f}", inr(sc_)))
            slots_free -= 1

        print(f"\n  ENTRIES ({len(entry_rows)})")
        if entry_rows:
            print_table(["Sector", "Rank", "Score", "Ret12m", "Ret3m", "Entry₹", "Capital"],
                        entry_rows, [22, 5, 7, 8, 8, 10, 12])
        else:
            print("    —")

        # ── Holds ────────────────────────────────────────────────────────────
        holds = current - to_sell
        hold_rows = []
        for sym in sorted(holds, key=lambda s: rank_map.get(s, 9999)):
            pos   = portfolio[sym]
            cp    = _price_on(sym, rebal_day, pos["entry_price"])
            gp    = (cp / pos["entry_price"] - 1) * 100
            sd    = score_data.get(sym, {})
            score = sd.get("score", 1.0)
            warn  = " ⚠" if score < 1.0 else ""
            hold_rows.append((sym, rank_map.get(sym, "—"),
                              pos["entry_date"].strftime("%d-%b-%y"),
                              f"{score:.3f}{warn}", f"{pos['entry_price']:,.1f}",
                              f"{cp:,.1f}", pct(gp)))

        print(f"\n  HOLDS ({len(hold_rows)})")
        if hold_rows:
            print_table(["Sector", "Rank", "Since", "Score", "Entry₹", "Now₹", "P&L%"],
                        hold_rows, [22, 5, 10, 10, 10, 10, 8])
        else:
            print("    —")

        # Track turnover (entries + exits this rebal)
        turnover_per_rebal.append(len(entry_rows) + len(exit_rows))

        nav_after = portfolio_nav(rebal_day)
        invested  = sum(pos["slot_capital"] * _price_on(sym, rebal_day, pos["entry_price"]) / pos["entry_price"]
                        for sym, pos in portfolio.items())
        print(f"\n  AFTER: Invested ₹{invested:,.0f} | Cash ₹{cash:,.0f} | "
              f"Total ₹{nav_after:,.0f} | Positions {len(portfolio)}/{MAX_SLOTS}")
        rebal_nav.append({"date": rebal_day.date(), "nav": nav_after})

        # Structured per-rebal record for the markdown report
        rebal_log.append({
            "idx":       rebal_idx + 1,
            "date":      rebal_day.date(),
            "eligible":  len(eligible_set),
            "nav_before": nav,
            "nav_after":  nav_after,
            "entries":   [(sym, rank_map[sym], score_data.get(sym, {}).get("score", 0))
                          for sym in [r[0] for r in entry_rows]],
            "exits":     [(r[0], r[1], r[5], r[6]) for r in exit_rows],   # sym, rank, gross_pnl_str, pnl_pct_str
            "holds":     [(r[0], r[1], r[6]) for r in hold_rows],          # sym, rank, pnl%
        })

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
            "hold_days": (last_day.date() - pos["entry_date"]).days, "open": True,
        })

    # ── Summary (basic ETF Core style) ────────────────────────────────────────
    closed   = [t for t in all_trades if not t.get("open")]
    open_pos = [t for t in all_trades if t.get("open")]
    winners  = [t for t in closed if t["gross_pnl"] > 0]
    losers   = [t for t in closed if t["gross_pnl"] <= 0]
    pf       = sum(t["gross_pnl"] for t in winners) / abs(sum(t["gross_pnl"] for t in losers)) \
               if losers and sum(t["gross_pnl"] for t in losers) < 0 else float('inf')
    wr       = len(winners) / len(closed) * 100 if closed else 0

    total_net = sum(t["net_pnl"] for t in all_trades) + liquidbees_income
    initial   = SLOT_START * MAX_SLOTS
    n_years   = (last_day.date() - START_DATE).days / 365.25
    cagr      = (1 + total_net / initial) ** (1 / n_years) - 1 if n_years > 0 else 0
    avg_hold  = np.mean([t["hold_days"] for t in closed]) if closed else 0

    year_navs = {}
    for row in rebal_nav:
        year_navs.setdefault(row["date"].year, []).append(row["nav"])

    print(f"\n{'='*72}")
    print(f"  SECTOR Z-SCORE V3 PHASE 1  |  Top {MAX_SLOTS} sectors  |  monthly rebal")
    print(f"{'='*72}")
    print(f"  Period         : {START_DATE} → {last_day.date()}  ({n_years:.2f} yr)")
    print(f"  Initial capital: ₹{initial/1e5:.1f}L  ({MAX_SLOTS} slots × ₹{SLOT_START/1e5:.1f}L)")
    print()

    prev_nav, neg_years = initial, 0
    for yr in sorted(year_navs):
        end_nav = year_navs[yr][-1]
        ret = (end_nav / prev_nav - 1) * 100
        tag = " ◄ NEG" if ret < 0 else ""
        if ret < 0: neg_years += 1
        print(f"  {yr}: {ret:+6.1f}%  (NAV ₹{end_nav/1e5:.2f}L){tag}")
        prev_nav = end_nav

    final_nav = rebal_nav[-1]["nav"] if rebal_nav else initial
    total_ret = (final_nav - initial) / initial * 100
    print()
    print(f"  Total Return  : {total_ret:+.1f}%")
    print(f"  CAGR (P&L)    : {cagr*100:+.1f}%")
    print(f"  Neg years     : {neg_years}")
    print()
    print(f"  Closed Trades : {len(closed)}  |  Open: {len(open_pos)}")
    print(f"  Win Rate (trades) : {wr:.1f}%  ({len(winners)}W / {len(losers)}L)")
    print(f"  Profit Factor : {pf:.2f}")
    print(f"  Avg hold      : {avg_hold:.0f} days")
    # Exit-reason breakdown
    reason_counts = {}
    for t in closed:
        r = t.get("reason", "—")
        reason_counts[r] = reason_counts.get(r, 0) + 1
    if reason_counts:
        breakdown = "  |  ".join(f"{r}: {n}" for r, n in sorted(reason_counts.items(),
                                                                key=lambda kv: -kv[1]))
        print(f"  Exit reasons  : {breakdown}")
    print(f"  Total charges : ₹{total_charges:,.0f}")
    print(f"  LIQUIDBEES    : ₹{liquidbees_income:,.0f}  ({LIQUIDBEES_PA*100:.1f}% p.a. on idle slots)")
    print(f"  Net P&L       : ₹{total_net:+,.0f}")
    print(f"  Final NAV     : ₹{final_nav:,.0f}  (₹{final_nav/1e5:.2f}L)")
    print()

    # ── Pass-criteria gate ────────────────────────────────────────────────────
    metrics = compute_metrics(rebal_nav, bench, START_DATE, last_day.date())
    if metrics:
        print(f"  Sharpe (mthly→ann) : {metrics['sharpe']:.2f}")
        print(f"  IR vs Nifty 200    : {metrics['ir']:.2f}")
        print(f"  MaxDD strategy     : {metrics['max_dd']*100:.1f}%")
        print(f"  MaxDD Nifty 200    : {metrics['bench_max_dd']*100:.1f}%")
        print(f"  Win rate vs N200   : {metrics['win_rate_vs_bench']:.1f}% of months")
        print(f"  Worst monthly rel  : {metrics['worst_monthly_rel']*100:+.1f}pp")
        print(f"  Bench CAGR (N200)  : {metrics['bench_cagr']*100:+.1f}%")
        print(f"  Strategy CAGR      : {metrics['cagr']*100:+.1f}%")

    avg_turnover = np.mean(turnover_per_rebal) if turnover_per_rebal else 0
    n_pass, n_fail = evaluate_gate(metrics, avg_turnover)

    # Export NAV CSV for correlation analysis
    if rebal_nav:
        nav_csv = os.path.join(os.path.dirname(BASE_DIR), "sector_zscore_rebal.csv")
        pd.DataFrame(rebal_nav).rename(columns={"date": "Date", "nav": "NAV"}).to_csv(nav_csv, index=False)
        print(f"\n  Rebalance NAV exported → sector_zscore_rebal.csv ({len(rebal_nav)} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sector Z-Score V3 Phase 1 backtest")
    parser.add_argument("--refresh",       action="store_true", help="Re-download data")
    parser.add_argument("--start",         default=None,        help="Override auto-detected start date (YYYY-MM-DD)")
    parser.add_argument("--no-liquidbees", action="store_true", help="Disable LIQUIDBEES idle accrual (ablation)")
    parser.add_argument("--no-md",         action="store_true", help="Skip writing sector_zscore_report.md")
    args = parser.parse_args()

    # Capture stdout to a markdown report (matches etf_core_report.md style:
    # raw fixed-width console output saved verbatim).
    md_path = os.path.join(os.path.dirname(BASE_DIR), "sector_zscore_report.md")
    if args.no_md:
        run(refresh=args.refresh, start_override=args.start, no_liquidbees=args.no_liquidbees)
    else:
        buf = io.StringIO()
        real_stdout = sys.stdout
        sys.stdout = _Tee(real_stdout, buf)
        try:
            run(refresh=args.refresh, start_override=args.start, no_liquidbees=args.no_liquidbees)
        finally:
            sys.stdout = real_stdout
            with open(md_path, "w") as f:
                f.write(buf.getvalue())
            print(f"\n  Console output captured → {os.path.basename(md_path)}  "
                  f"({len(buf.getvalue()):,} chars)")
