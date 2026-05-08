#!/usr/bin/env python3
"""
refresh_n200_baseline.py — daily 4:30 PM IST cron job. Bulk-fetches 500 days
of OHLCV for the entire scan universe + Nifty 50 + Nifty 200, pickles the
result to `data_store/cache/n200_baseline_<YYYY-MM-DD>.pkl`.

The Live Signals scan path (data/live_signals_engine.py:_fetch_baseline)
reads this pkl on every refresh and only does a small intraday-delta fetch
for today's bar — turning a 30-60s scan into ~5-10s during market hours.

Usage:
    python3 scripts/refresh_n200_baseline.py                  # default
    python3 scripts/refresh_n200_baseline.py --universe 200   # explicit
    python3 scripts/refresh_n200_baseline.py --days 500       # window

Cron (08:00 IST weekdays — *before* market open so the intraday-patch path
serves the latest data; rerun at 16:30 IST after close to capture the day):
    0 8,16 * * 1-5 /home/ubuntu/relative_strength/scripts/refresh_n200_baseline.sh \
        >> /home/ubuntu/logs/n200_baseline.log 2>&1
"""

import argparse
import os
import pickle
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from data.live_signals_engine import get_scan_tickers   # noqa: E402

CACHE_DIR = os.path.join(ROOT, "data_store", "cache")


def _fetch_chunked(yf_tickers, start, end, chunk_size=15, chunk_timeout=25):
    """Bulk-fetch with chunking + timeout. Mirrors live_signals_engine logic."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
    bulk_data = {}
    n = len(yf_tickers)
    for chunk_start in range(0, n, chunk_size):
        chunk = yf_tickers[chunk_start: chunk_start + chunk_size]
        print(f"  chunk {chunk_start:>3}-{chunk_start+len(chunk):>3} / {n}…", flush=True)
        try:
            with ThreadPoolExecutor(max_workers=1) as _ex:
                df = _ex.submit(
                    lambda c=chunk: yf.download(
                        c, start=start, end=end, progress=False,
                        auto_adjust=True, group_by="ticker",
                        threads=True, timeout=15)
                ).result(timeout=chunk_timeout)
        except FutTimeout:
            print(f"    timed out — skipping {chunk}")
            continue
        except Exception as e:
            print(f"    fetch failed: {e}")
            continue
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            for sym in chunk:
                if sym in df.columns.get_level_values(0):
                    sub = df[sym].dropna(how="all")
                    if len(sub) > 0:
                        if hasattr(sub.index, "tz") and sub.index.tz is not None:
                            sub.index = sub.index.tz_localize(None)
                        bulk_data[sym] = sub
        elif len(chunk) == 1:
            sub = df.dropna(how="all")
            if len(sub) > 0:
                if hasattr(sub.index, "tz") and sub.index.tz is not None:
                    sub.index = sub.index.tz_localize(None)
                bulk_data[chunk[0]] = sub
    return bulk_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--universe", type=int, default=200,
                   help="N50/N100/N200 universe size (default 200)")
    p.add_argument("--days", type=int, default=500,
                   help="Days of history to fetch (default 500)")
    args = p.parse_args()

    tickers = get_scan_tickers(args.universe)
    end_date = datetime.now() + timedelta(days=1)
    start_date = end_date - timedelta(days=args.days)

    print(f"Refreshing N{args.universe} baseline ({len(tickers)} tickers, "
          f"{args.days}d)")
    print(f"Date range: {start_date.date()} → {end_date.date()}")

    t0 = time.time()
    print("Fetching Nifty 50…")
    try:
        nifty_raw = yf.Ticker("^NSEI").history(start=start_date, end=end_date)
        if hasattr(nifty_raw.index, "tz") and nifty_raw.index.tz is not None:
            nifty_raw.index = nifty_raw.index.tz_localize(None)
    except Exception:
        nifty_raw = pd.DataFrame()

    print("Fetching Nifty 200…")
    try:
        bench_raw = yf.Ticker("^CNX200").history(start=start_date, end=end_date)
        if hasattr(bench_raw.index, "tz") and bench_raw.index.tz is not None:
            bench_raw.index = bench_raw.index.tz_localize(None)
    except Exception:
        bench_raw = nifty_raw

    print(f"Fetching {len(tickers)} N{args.universe} tickers…")
    yf_tickers = [f"{t}.NS" for t in tickers]
    bulk_data = _fetch_chunked(yf_tickers, start_date, end_date)

    elapsed = time.time() - t0
    print(f"\nFetched: nifty {len(nifty_raw)} rows, bench {len(bench_raw)} rows, "
          f"{len(bulk_data)}/{len(tickers)} tickers in {elapsed:.1f}s")

    os.makedirs(CACHE_DIR, exist_ok=True)
    out_path = os.path.join(CACHE_DIR,
                            f"n200_baseline_{datetime.now().strftime('%Y-%m-%d')}.pkl")
    payload = {
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "universe":   args.universe,
        "days":       args.days,
        "tickers":    tickers,
        "nifty_raw":  nifty_raw,
        "bench_raw":  bench_raw,
        "bulk_data":  bulk_data,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    size_kb = os.path.getsize(out_path) // 1024
    print(f"Saved: {out_path} ({size_kb} KB)")

    # Prune old baselines (keep last 7)
    import glob
    old_files = sorted(glob.glob(os.path.join(CACHE_DIR, "n200_baseline_*.pkl")))
    for old in old_files[:-7]:
        try:
            os.remove(old)
            print(f"Pruned: {os.path.basename(old)}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
