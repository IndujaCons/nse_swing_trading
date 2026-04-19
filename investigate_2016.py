#!/usr/bin/env python3
"""
Investigates 2016 (-2.8%) in the new Mom15 system.
Spot-checks the Feb-2016 and Apr-2016 rebalance books.
Shows: what was held, what was blocked by 52w filter, what the raw scorer ranked.
"""

import os, sys, json, pickle, warnings
from datetime import date, timedelta
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mom15_pit_report import (
    load_pit, get_pit_universe, eps_passes,
    compute_scores, BUFFER_IN, BUFFER_OUT, MAX_SLOTS, W12, W3
)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, 'data', 'cache', 'mom15_daily.pkl')
PIT_FILE   = os.path.join(BASE_DIR, 'nse_const', 'nifty200_pit.json')
EPS_FILE   = os.path.join(BASE_DIR, 'data', 'quarterly_eps.json')
BETA_CAP   = 1.0

print("Loading cached data...")
with open(CACHE_FILE, 'rb') as f:
    stock_data = pickle.load(f)

print("Loading PIT + EPS...")
pit_data = load_pit()
eps_db = json.load(open(EPS_FILE)) if os.path.exists(EPS_FILE) else None

import yfinance as yf
n50_raw = yf.Ticker("^NSEI").history(start="2013-01-01", end=date.today())
n50_raw.index = n50_raw.index.tz_localize(None) if n50_raw.index.tzinfo else n50_raw.index
n50_iloc = {n50_raw.index[i].date(): i for i in range(len(n50_raw))}

date_to_iloc = {t: {df.index[i].date(): i for i in range(len(df))}
                for t, df in stock_data.items()}

# Rebalance dates to investigate
CHECK_DATES = [
    date(2015, 12, 1),   # Dec 2015 — what was the book going into 2016?
    date(2016, 2, 1),    # Feb 2016
    date(2016, 4, 1),    # Apr 2016
    date(2016, 6, 1),    # Jun 2016
]

def nearest_trading_day(target, stock_data):
    """Find nearest actual trading day on or after target."""
    for off in range(0, 10):
        d = target + timedelta(days=off)
        # Check if at least 50 stocks have data on this day
        count = sum(1 for df in stock_data.values()
                    if any(df.index[i].date() == d for i in range(min(len(df), 5000))))
        if count >= 50:
            return d
    return target

def get_52w_high(ticker, day, stock_data, date_to_iloc):
    idx_map = date_to_iloc.get(ticker, {})
    ci = idx_map.get(day)
    if ci is None:
        for off in range(1, 6):
            if (day - timedelta(days=off)) in idx_map:
                ci = idx_map[day - timedelta(days=off)]
                break
    if ci is None or ci < 252:
        return None
    return float(stock_data[ticker]["High"].iloc[ci-252:ci+1].max())

def analyze_rebal(day):
    # Find actual trading day
    for off in range(0, 8):
        d = day + timedelta(days=off)
        cnt = sum(1 for idxm in date_to_iloc.values() if d in idxm)
        if cnt >= 50:
            day = d
            break

    print(f"\n{'='*70}")
    print(f"  REBALANCE CHECK — {day.strftime('%d %b %Y')}")
    print(f"{'='*70}")

    scores = compute_scores(day, stock_data, date_to_iloc, pit_data,
                            n50_raw, n50_iloc, eps_db)
    if not scores:
        print("  ⚠ No scores computed")
        return

    ranked = sorted(scores.items(), key=lambda x: -x[1]["norm_score"])
    ticker_rank = {t: r+1 for r, (t, _) in enumerate(ranked)}

    # Top 20 by score
    print(f"\n  TOP 20 BY SCORE (universe: {len(scores)} stocks after beta+EPS filter):")
    print(f"  {'Rank':<5} {'Ticker':<12} {'Score':>6} {'Beta':>5} {'Ret12m':>7} {'Ret3m':>6} "
          f"{'Price':>8} {'52wHi':>8} {'Dist52w':>8} {'Entry?':>7}")
    print(f"  {'-'*80}")

    blocked_count = 0
    for r, (t, s) in enumerate(ranked[:20]):
        price   = s["price"]
        hi52    = get_52w_high(t, day, stock_data, date_to_iloc)
        dist    = (price / hi52 - 1) * 100 if hi52 else 0
        blocked = hi52 is not None and price < hi52 * 0.80
        if blocked:
            blocked_count += 1
        entry_ok = "✓" if not blocked else f"✗ 52w"
        print(f"  {r+1:<5} {t:<12} {s['norm_score']:>6.3f} {s['beta']:>5.2f} "
              f"{s['ret_12m']*100:>+6.1f}% {s['ret_3m']*100:>+5.1f}% "
              f"{price:>8,.0f} {hi52 or 0:>8,.0f} {dist:>+7.1f}% {entry_ok:>7}")

    # Summary of blocks
    total_blocked = sum(
        1 for t, s in ranked[:20]
        if (lambda h: h is not None and s["price"] < h * 0.80)(
            get_52w_high(t, day, stock_data, date_to_iloc))
    )

    print(f"\n  52w filter blocks {total_blocked} of top 20 candidates on this date")

    # Nifty 200 index level context
    pit_set = get_pit_universe(pit_data, day)
    print(f"\n  PIT universe on {day}: {len(pit_set)} stocks")
    print(f"  Stocks with scores computed: {len(scores)}")

    # What % of all scored stocks are >20% off their 52w high?
    all_blocked = []
    all_ok = []
    for t, s in ranked:
        h = get_52w_high(t, day, stock_data, date_to_iloc)
        if h and s["price"] < h * 0.80:
            all_blocked.append((t, (s["price"]/h - 1)*100))
        else:
            all_ok.append(t)
    print(f"  Stocks >20% off 52w high (blocked from entry): {len(all_blocked)} "
          f"({len(all_blocked)/len(scores)*100:.0f}% of universe)")
    if all_blocked:
        worst = sorted(all_blocked, key=lambda x: x[1])[:8]
        print(f"  Worst: " + ", ".join(f"{t}({d:+.0f}%)" for t, d in worst))

# Run analysis for each quarter
for d in CHECK_DATES:
    analyze_rebal(d)

# Also: reconstruct approximate 2016 P&L month by month
print(f"\n\n{'='*70}")
print("  NIFTY 200 INDEX — 2015/2016 CONTEXT")
print(f"{'='*70}")
try:
    import yfinance as yf
    n200 = yf.Ticker("^CNX200").history(start="2015-01-01", end="2016-12-31")
    n200.index = n200.index.tz_localize(None) if n200.index.tzinfo else n200.index
    closes = n200["Close"]
    for m in ["2015-01", "2015-06", "2015-12", "2016-01", "2016-02",
              "2016-03", "2016-04", "2016-06", "2016-12"]:
        mask = closes.index.to_series().dt.strftime("%Y-%m") == m
        vals = closes[mask]
        if not vals.empty:
            print(f"  {m}: {vals.iloc[-1]:,.0f}  "
                  f"({'first' if m.endswith('-01') else 'last'} of month)")
except Exception as e:
    print(f"  Could not fetch index: {e}")
