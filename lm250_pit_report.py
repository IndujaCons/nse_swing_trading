#!/usr/bin/env python3
"""
LM250 PIT Backtest — Nifty LargeMidcap 250, Monthly Rebalance
==============================================================
Capital  : ₹20L starting, NAV/20 per slot (compounding)
Universe : Nifty LargeMidcap 250 (PIT, 417 historical tickers)
Params   : Top 20 | Buffer 15/40 | β≤1.2 vs Nifty50 | Monthly | Sector≤4
Regime   : Nifty200 (^CRSLDX) < SMA200 → hold all, no churn
Scoring  : MR_12 (50%) + MR_3 (50%), Z-scored, Normalised Score

Usage:
    python3 lm250_pit_report.py              # uses cached data
    python3 lm250_pit_report.py --refresh    # re-download
    python3 lm250_pit_report.py --no-regime  # disable regime filter
"""

import os, sys, json, pickle, warnings, argparse
from datetime import date, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
START_DATE   = date(2017, 12, 1)   # LM250 launched Nov 30 2017
MAX_SLOTS    = 20
BUFFER_IN    = 15
BUFFER_OUT   = 40
BETA_CAP     = 1.2
SECTOR_CAP   = 4
W12, W3      = 0.50, 0.50
LONG_PD      = 252
SHORT_PD     = 63
WARMUP_DAYS  = 450

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE     = os.path.join(BASE_DIR, 'data', 'cache', 'lm250_daily.pkl')
PIT_FILE       = os.path.join(BASE_DIR, 'nse_const', 'largemid250_pit.json')
SECTOR_FILE    = os.path.join(BASE_DIR, 'nse_const', 'largemid250_sectors.json')
BENCH_TICKER   = "^CRSLDX"   # Nifty200 — closest proxy for LM250 regime
BENCH_LABEL    = "Nifty 200"
BETA_TICKER    = "^NSEI"

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
    "ETERNAL":    "ZOMATO.NS",
    "ZOMATO":     "ETERNAL.NS",
    "ALBK":       "INDIANB.NS",
    "ANDHRABANK": "UNIONBANK.NS",
    "CORPBANK":   "PNB.NS",
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
    "NIITTECH":   "COFORGE.NS",
    "ORIENTBANK": "PNB.NS",
    "RNAM":       "NAM-INDIA.NS",
    "STRTECH":    "STLTECH.NS",
    "SYNDIBANK":  "CANBK.NS",
    "WELSPUNIND": "WELSPUNLIV.NS",
    "AMARAJABAT": "500008.BO",
    "DHFL":       "511072.BO",
    "GSKCONS":    "500676.BO",
    "IBULHSGFIN": "535789.BO",
    "MCDOWELL-N": "UNITDSPR.BO",
    "PEL":        "500302.BO",
    "TMPV":       "TMPV.BO",
    "GET&D":      "GVT&D.NS",
}

# ── PIT UNIVERSE ──────────────────────────────────────────────────────────────
def load_pit():
    with open(PIT_FILE) as f:
        raw = json.load(f)
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

# ── DATA LOADING ──────────────────────────────────────────────────────────────
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

def load_or_fetch_data(tickers, fetch_start, fetch_end, refresh=False):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    if not refresh and os.path.exists(CACHE_FILE):
        print(f"Loading cached data from {CACHE_FILE}...")
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    print(f"Fetching {len(tickers)} tickers from yfinance (~5-10 min)...")
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
    print(f"  Cached → {CACHE_FILE}")
    return stock_data

# ── SCORING ───────────────────────────────────────────────────────────────────
def compute_scores(day, stock_data, date_to_iloc, pit_data, n50_data, n50_iloc):
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
        p_now  = closes[ci]
        p_12m  = closes[ci - LONG_PD]
        p_3m   = closes[ci - SHORT_PD]
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

        # Beta vs Nifty50
        beta = None
        n50_ci = n50_iloc.get(day)
        if n50_ci is None:
            for off in range(1, 6):
                prev = day - timedelta(days=off)
                if prev in n50_iloc:
                    n50_ci = n50_iloc[prev]
                    break
        if n50_ci is not None and n50_ci >= 252:
            n50_c = n50_data["Close"].values[n50_ci-252:n50_ci+1].astype(float)
            stk_c = closes[ci-252:ci+1]
            n50_r = np.diff(n50_c) / np.maximum(n50_c[:-1], 0.01)
            stk_r = np.diff(stk_c) / np.maximum(stk_c[:-1], 0.01)
            if len(stk_r) == len(n50_r) and len(stk_r) > 50:
                cov = np.cov(stk_r, n50_r)
                if cov.shape == (2, 2) and cov[1, 1] > 1e-10:
                    beta = cov[0, 1] / cov[1, 1]

        raw[ticker] = {
            "price": p_now, "mr_12": mr_12, "mr_3": mr_3,
            "ret_12m": ret_12, "ret_3m": ret_3, "sigma": sigma, "beta": beta,
        }

    raw = {t: s for t, s in raw.items()
           if s["beta"] is not None and s["beta"] <= BETA_CAP}

    if len(raw) < 20:
        return {}

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
        s["wt_z"]       = wz
        s["norm_score"] = (1 + wz) if wz >= 0 else 1.0 / (1 - wz)

    return raw

# ── REBALANCE DATES ───────────────────────────────────────────────────────────
def get_rebal_dates(trading_days):
    seen, result = set(), []
    for d in trading_days:
        ym = (d.year, d.month)
        if ym not in seen and d >= START_DATE:
            seen.add(ym)
            result.append(d)
    return result

# ── FORMATTING ────────────────────────────────────────────────────────────────
def inr(v):
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

# ── MAIN ─────────────────────────────────────────────────────────────────────
def run(refresh=False, use_regime=True, beta_cap_override=None):
    global BETA_CAP
    import time as _time

    if beta_cap_override is not None:
        BETA_CAP = beta_cap_override
    regime_label = "Regime ON [SMA200]" if use_regime else "Regime OFF"
    label = f"LM250 — Nifty LargeMidcap 250, Monthly, β≤{BETA_CAP}, Sector≤4 | {regime_label}"
    print(f"=== {label} ===")
    print(f"    top_n={MAX_SLOTS} buffer_in={BUFFER_IN} buffer_out={BUFFER_OUT} "
          f"beta_cap={BETA_CAP} sector_cap={SECTOR_CAP}")

    # Load PIT
    print("Loading PIT universe...")
    pit_data    = load_pit()
    all_tickers = get_all_pit_tickers(pit_data)
    print(f"  {len(all_tickers)} unique PIT tickers across all periods")

    # Load flat sector map
    sector_map = {}
    if os.path.exists(SECTOR_FILE):
        with open(SECTOR_FILE) as f:
            sector_map = json.load(f)
        print(f"  Sector map loaded: {len(sector_map)} tickers")

    def get_sector(ticker):
        return sector_map.get(ticker, "OTHER")

    # Date range
    fetch_start = START_DATE - timedelta(days=WARMUP_DAYS)
    fetch_end   = date.today()

    stock_data = load_or_fetch_data(all_tickers, fetch_start, fetch_end, refresh)

    # Nifty50 for beta
    print(f"Fetching Nifty 50 ({BETA_TICKER}) for beta...")
    n50_raw = pd.DataFrame()
    for attempt in range(3):
        try:
            n50_raw = yf.Ticker(BETA_TICKER).history(start=fetch_start, end=fetch_end)
            if not n50_raw.empty:
                break
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
        _time.sleep(2)
    if n50_raw.empty:
        print("  ERROR: Could not fetch Nifty50 — beta filter disabled")
    else:
        n50_raw.index = n50_raw.index.tz_localize(None) if n50_raw.index.tzinfo else n50_raw.index
        print(f"  {len(n50_raw)} bars")
    n50_iloc = {n50_raw.index[i].date(): i for i in range(len(n50_raw))}

    # Nifty200 for regime
    print(f"Fetching {BENCH_LABEL} ({BENCH_TICKER}) for regime...")
    n200_raw = pd.DataFrame()
    for attempt in range(3):
        try:
            n200_raw = yf.Ticker(BENCH_TICKER).history(start=fetch_start, end=fetch_end)
            if not n200_raw.empty:
                break
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
        _time.sleep(2)
    if n200_raw.empty:
        print(f"  WARNING: Could not fetch {BENCH_LABEL} — regime filter disabled")
        use_regime = False
    else:
        n200_raw.index = n200_raw.index.tz_localize(None) if n200_raw.index.tzinfo else n200_raw.index
        n200_raw["sma200"] = n200_raw["Close"].rolling(200).mean()
        print(f"  {len(n200_raw)} bars")
    n200_iloc = {n200_raw.index[i].date(): i for i in range(len(n200_raw))}

    # date → iloc maps
    date_to_iloc = {}
    for ticker, df in stock_data.items():
        date_to_iloc[ticker] = {df.index[i].date(): i for i in range(len(df))}

    # Trading days
    day_counts = {}
    for df in stock_data.values():
        for d in df.index:
            dt = d.date()
            day_counts[dt] = day_counts.get(dt, 0) + 1
    trading_days = sorted(d for d, c in day_counts.items() if c >= 50 and d >= START_DATE)
    print(f"  Trading days: {len(trading_days)} ({trading_days[0]} → {trading_days[-1]})")

    rebal_dates = get_rebal_dates(trading_days)
    print(f"  Rebalance dates: {len(rebal_dates)}")

    # ── Portfolio state ──────────────────────────────────────────────────────
    cash           = 20_00_000.0
    _starting_cash = cash
    portfolio      = {}
    all_trades     = []
    total_charges  = 0.0
    rebal_nav      = []

    banner = (f"  LM250 PIT BACKTEST  |  NAV/20 slot  |  Monthly Rebalance  |  "
              f"Nifty LargeMidcap 250  |  β≤1.2  |  Sector≤4  |  {regime_label}")
    print()
    print("=" * len(banner))
    print(banner)
    print("=" * len(banner))

    for rebal_idx, rebal_day in enumerate(rebal_dates):
        # MTM portfolio
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
        per_slot = port_value / MAX_SLOTS

        print()
        print("=" * 72)
        print(f"  REBALANCE #{rebal_idx+1:02d}  —  {rebal_day.strftime('%d %b %Y')}")
        print(f"  NAV: {inr(port_value)}  |  Slot: {inr(per_slot)}  |  Cash: {inr(cash)}")
        print("=" * 72)

        scores = compute_scores(rebal_day, stock_data, date_to_iloc, pit_data, n50_raw, n50_iloc)
        if not scores:
            print(f"  ⚠ Insufficient scored stocks — skipping rebalance")
            continue

        ranked      = sorted(scores.items(), key=lambda x: -x[1]["norm_score"])
        ticker_rank = {t: r+1 for r, (t, _) in enumerate(ranked)}

        # Buffer rule
        current_set = set(portfolio.keys())
        new_set     = set()

        for t in current_set:
            if ticker_rank.get(t, 9999) <= BUFFER_OUT:
                new_set.add(t)

        for r, (t, _) in enumerate(ranked):
            if r + 1 > BUFFER_IN:
                break
            if t not in current_set and t not in new_set:
                new_set.add(t)

        for r, (t, _) in enumerate(ranked):
            if len(new_set) >= MAX_SLOTS:
                break
            if t not in new_set:
                new_set.add(t)

        if len(new_set) > MAX_SLOTS:
            new_set = set(sorted(new_set, key=lambda t: ticker_rank.get(t, 9999))[:MAX_SLOTS])

        # ── SECTOR CAP ────────────────────────────────────────────────────────
        if sector_map:
            def _sort_key(t):
                if t in current_set:
                    return (0, ticker_rank.get(t, 9999))
                return (1, ticker_rank.get(t, 9999))
            new_set_ranked = sorted(new_set, key=_sort_key)
            sec_counts, capped_set, dropped_by_cap = {}, set(), []
            for t in new_set_ranked:
                sec = get_sector(t)
                if sec_counts.get(sec, 0) < SECTOR_CAP:
                    capped_set.add(t)
                    sec_counts[sec] = sec_counts.get(sec, 0) + 1
                else:
                    dropped_by_cap.append(t)
            if dropped_by_cap:
                print(f"  [SECTOR CAP≤{SECTOR_CAP}] dropped: {', '.join(dropped_by_cap)}")
                for _, (t, _) in enumerate(ranked):
                    if len(capped_set) >= MAX_SLOTS:
                        break
                    if t in capped_set:
                        continue
                    sec = get_sector(t)
                    if sec_counts.get(sec, 0) < SECTOR_CAP:
                        capped_set.add(t)
                        sec_counts[sec] = sec_counts.get(sec, 0) + 1
            new_set = capped_set

        # ── REGIME CHECK ──────────────────────────────────────────────────────
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
                sma200     = n200_raw["sma200"].iloc[n200_ci]
                if not pd.isna(sma200) and n200_close < float(sma200):
                    regime_off = True
                    print(f"\n  [REGIME OFF] {BENCH_LABEL} {n200_close:,.1f} < SMA200 "
                          f"{float(sma200):,.1f} — holding all, skipping exits & entries")

        # ── EXITS ─────────────────────────────────────────────────────────────
        to_sell   = current_set - new_set
        exit_rows = []
        for t in sorted(to_sell):
            if regime_off:
                break
            pos       = portfolio[t]
            ep        = pos.get("curr_price", pos["entry_price"])
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
                t, ticker_rank.get(t, "—"), get_sector(t),
                pos["entry_date"].strftime("%d-%b-%y"),
                f"{pos['entry_price']:,.1f}", f"{ep:,.1f}",
                pos["shares"], inr(gross_pnl), pct(pnl_pct), f"{hold_days}d",
            ))
            del portfolio[t]

        print(f"\n  EXITS ({len(exit_rows)})")
        if exit_rows:
            print_table(
                ["Ticker","Rank","Sector","Entry","Entry₹","Exit₹","Qty","Gross P&L","P&L%","Hold"],
                sorted(exit_rows, key=lambda r: float(r[8].replace('+','').replace('%','')), reverse=True),
                [10, 5, 20, 10, 10, 10, 5, 12, 8, 6]
            )
        else:
            print("    —")

        # ── ENTRIES ───────────────────────────────────────────────────────────
        to_buy     = new_set - current_set
        entry_rows = []
        skipped_52w = []

        for t in sorted(to_buy, key=lambda t: ticker_rank.get(t, 9999)):
            if regime_off:
                break
            s  = scores[t]
            ep = s["price"]

            # 52w high filter
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
                if ep < high_52w * 0.80:
                    dist = (ep / high_52w - 1) * 100
                    skipped_52w.append((t, ticker_rank[t], f"{ep:,.1f}",
                                        f"{high_52w:,.1f}", f"{dist:.1f}%"))
                    continue

            shares = int(per_slot // ep)
            if shares == 0:
                continue
            cost = ep * shares
            if cost > cash:
                continue
            chg   = calc_charges(cost, 0)
            cash -= (cost + chg)
            total_charges += chg
            portfolio[t] = {
                "entry_date":  rebal_day,
                "entry_price": ep,
                "shares":      shares,
            }
            entry_rows.append((
                t, ticker_rank[t], get_sector(t),
                f"{s['norm_score']:.3f}", f"{s['beta']:.2f}",
                f"{s['ret_12m']*100:+.1f}%", f"{s['ret_3m']*100:+.1f}%",
                f"{ep:,.1f}", shares, inr(cost),
            ))

        print(f"\n  ENTRIES ({len(entry_rows)})")
        if skipped_52w:
            print(f"  [52w filter blocked {len(skipped_52w)}: "
                  + ", ".join(f"{r[0]}({r[4]})" for r in skipped_52w) + "]")
        if entry_rows:
            print_table(
                ["Ticker","Rank","Sector","Score","Beta","Ret12m","Ret3m","Entry₹","Qty","Capital"],
                entry_rows,
                [10, 5, 20, 7, 5, 8, 8, 10, 5, 12]
            )
        else:
            print("    —")

        # ── HOLDS ─────────────────────────────────────────────────────────────
        holds     = current_set if regime_off else (current_set & new_set)
        hold_rows = []
        warn_syms = []
        for t in sorted(holds, key=lambda t: ticker_rank.get(t, 9999)):
            pos    = portfolio[t]
            cp     = pos.get("curr_price", pos["entry_price"])
            unreal = (cp - pos["entry_price"]) * pos["shares"]
            pp     = (cp / pos["entry_price"] - 1) * 100
            sc     = scores.get(t, {}).get("norm_score", 1.0)
            warn   = " ⚠" if sc < 1.0 else ""
            if sc < 1.0:
                warn_syms.append(t)
            hold_rows.append((
                t, ticker_rank.get(t, "—"), get_sector(t),
                pos["entry_date"].strftime("%d-%b-%y"),
                f"{pos['entry_price']:,.1f}", f"{cp:,.1f}",
                pos["shares"], inr(unreal), f"{pct(pp)}{warn}",
            ))

        print(f"\n  HOLDS ({len(hold_rows)})")
        if hold_rows:
            print_table(
                ["Ticker","Rank","Sector","Since","Entry₹","Now₹","Qty","Unreal P&L","P&L%"],
                sorted(hold_rows, key=lambda r: float(r[8].replace('+','').replace('%','').replace(' ⚠','')), reverse=True),
                [10, 5, 20, 10, 10, 10, 5, 12, 10]
            )
            if warn_syms:
                print(f"  ⚠  WAZ < 0 (momentum below universe mean): {', '.join(warn_syms)}")
        else:
            print("    —")

        # ── AFTER SUMMARY ─────────────────────────────────────────────────────
        invested  = sum(pos["shares"] * pos.get("curr_price", pos["entry_price"])
                        for pos in portfolio.values())
        total_now = cash + invested
        print(f"\n  AFTER: Invested {inr(invested)} | Cash {inr(cash)} | "
              f"Total {inr(total_now)} | Positions {len(portfolio)}/{MAX_SLOTS} | Slot {inr(per_slot)}")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    open_pnl = 0.0
    for t, pos in portfolio.items():
        idx_map = date_to_iloc.get(t, {})
        last_ci = max(idx_map.values()) if idx_map else None
        lp = float(stock_data[t]["Close"].iloc[last_ci]) if last_ci else pos["entry_price"]
        open_pnl += (lp - pos["entry_price"]) * pos["shares"]

    closed_pnl  = sum(tr["net_pnl"] for tr in all_trades)
    winners     = [tr for tr in all_trades if tr["gross_pnl"] > 0]
    losers      = [tr for tr in all_trades if tr["gross_pnl"] <= 0]
    avg_hold    = sum(tr["hold_days"] for tr in all_trades) / len(all_trades) if all_trades else 0

    final_value  = cash + sum(
        pos["shares"] * (float(stock_data[t]["Close"].iloc[max(date_to_iloc[t].values())])
                         if t in date_to_iloc else pos["entry_price"])
        for t, pos in portfolio.items()
    )
    total_return = (final_value - _starting_cash) / _starting_cash * 100
    years        = (trading_days[-1] - trading_days[0]).days / 365.25
    cagr         = ((final_value / _starting_cash) ** (1/years) - 1) * 100 if years > 0 else 0

    print()
    print("=" * 72)
    print("  FINAL SUMMARY")
    print("=" * 72)
    print(f"  Period        : {trading_days[0]} → {trading_days[-1]}  ({years:.1f} years)")
    print(f"  Starting Cap  : {inr(_starting_cash)}")
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

    # Per-year returns
    print()
    print("  YEAR-BY-YEAR:")
    year_last_nav = {}
    for row in rebal_nav:
        year_last_nav[row["date"].year] = row["nav"]
    prev_nav = _starting_cash
    neg_years = 0
    for yr in sorted(year_last_nav):
        end_nav = year_last_nav[yr]
        ret_y   = (end_nav / prev_nav - 1) * 100
        if ret_y < 0:
            neg_years += 1
        bar_len = int(abs(ret_y))
        bar     = ("█" * min(bar_len, 40)) if ret_y > 0 else ("░" * min(bar_len, 40))
        sign    = "+" if ret_y >= 0 else "-"
        print(f"  {yr}  {sign}{abs(ret_y):5.1f}%  {bar}")
        prev_nav = end_nav
    print(f"\n  Negative years: {neg_years}")
    print()

    # Export rebalance NAV CSV
    nav_csv = os.path.join(BASE_DIR, "lm250_rebal.csv")
    pd.DataFrame(rebal_nav).to_csv(nav_csv, index=False)
    print(f"  Rebalance NAV exported → lm250_rebal.csv ({len(rebal_nav)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LM250 PIT Momentum Backtest")
    parser.add_argument("--refresh",   action="store_true", help="Re-download price data")
    parser.add_argument("--no-regime", action="store_true", help="Disable regime filter")
    parser.add_argument("--beta-cap",  type=float, default=None, help="Override beta cap (default 1.2)")
    args = parser.parse_args()
    run(refresh=args.refresh, use_regime=not args.no_regime, beta_cap_override=args.beta_cap)
