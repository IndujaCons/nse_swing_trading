#!/usr/bin/env python3
"""
synth_sector_index.py — Reconstruct a sectoral index time series from its
constituent member stocks (equal-weight).

Use case: yfinance doesn't carry historical data for several NSE sectoral
indices (HEALTHCARE, INDIA DEFENCE, INDIA MFG, CONSUMER DURABLES, OIL &
GAS). niftyindices.com publishes the constituent list but not the index
time series. This script fetches each member's daily price from yfinance
and aggregates into an equal-weight synthetic index, rebased to 1000 at
the first common trading day.

Output: pickled dict {date → synthetic close} saved to
data/cache/synth_<index>_<YYYY-MM-DD>.pkl

Usage:
    python3 data/synth_sector_index.py NIFTY_HEALTHCARE
    python3 data/synth_sector_index.py NIFTY_INDIA_DEFENCE
    python3 data/synth_sector_index.py --all
"""

import argparse, csv, os, pickle, sys, time
from datetime import date, timedelta
import requests
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(HERE, "cache")

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

# Constituent CSV URLs at niftyindices.com
INDEX_URLS = {
    "NIFTY_HEALTHCARE":         "ind_niftyhealthcarelist.csv",
    "NIFTY_CONSUMER_DURABLES":  "ind_niftyconsumerdurableslist.csv",
    # NIFTY_INDIA_MFG moved to HARDCODED (clean 24-stock list, see below)
    "NIFTY_INDIA_DEFENCE":      "ind_niftyindiadefence_list.csv",
}

# OIL & GAS — hardcoded since niftyindices doesn't publish its CSV
HARDCODED = {
    "NIFTY_OIL_GAS": ["RELIANCE", "ONGC", "IOC", "BPCL", "GAIL", "HINDPETRO",
                      "PETRONET", "OIL", "ATGL", "MGL", "IGL", "GUJGASLTD"],
    # NIFTY INDIA MFG — "clean" version: only the 24 stocks unique to MFG
    # (excluded from NIFTY AUTO/METAL/PHARMA/HEALTHCARE/DEFENCE/ENERGY/
    # OIL&GAS/CONS DURABLES/FMCG). Captures chemicals + capital goods +
    # specialty industrials. Removes 68% overlap with other sectors that
    # the broad 80-stock index has — see commit log.
    "NIFTY_INDIA_MFG": [
        "AIAENG", "ANTHEM", "ASTRAL", "BALKRISIND", "COROMANDEL", "CUMMINSIND",
        "ESCORTS", "HONAUT", "HSCL", "HYUNDAI", "KAYNES", "KEI", "KPRMILL",
        "LINDEINDIA", "MRF", "NAVINFLUOR", "PAGEIND", "PIDILITIND", "PIIND",
        "POLYCAB", "SRF", "SUPREMEIND", "TMCV", "UPL",
    ],
}

# Sectors whose Yahoo Finance sub-index ticker (^CNXxxx) stopped serving
# historical data (broke sometime between 2026-05-08 and 2026-07-05 — see
# memory/mom20.md-adjacent notes). Rebuilt as equal-weight synthetics from
# the existing sector_mapping.STOCK_SECTOR_MAP constituent lists (already
# curated for the Mom20/Mom500 sector-cap logic — no new scraping needed).
FROM_STOCK_MAP = [
    "NIFTY_PVT_BANK", "NIFTY_PSU_BANK", "NIFTY_FIN_SERVICE", "NIFTY_AUTO",
    "NIFTY_METAL", "NIFTY_ENERGY", "NIFTY_FMCG", "NIFTY_REALTY",
    "NIFTY_MEDIA", "NIFTY_MNC", "NIFTY_PSE", "NIFTY_CONSUMPTION",
    "NIFTY_INFRA",
]


def fetch_constituents(index_name: str, retries: int = 3) -> list:
    """Pull constituent symbols. Retries on timeout (niftyindices is flaky)."""
    if index_name in HARDCODED:
        return HARDCODED[index_name]
    if index_name in FROM_STOCK_MAP:
        sys.path.insert(0, ROOT)
        from sector_mapping import STOCK_SECTOR_MAP
        sector_label = index_name.replace("_", " ")  # NIFTY_AUTO -> NIFTY AUTO
        syms = [t for t, s in STOCK_SECTOR_MAP.items() if s == sector_label]
        if not syms:
            raise ValueError(f"No constituents found in STOCK_SECTOR_MAP for {sector_label}")
        return syms
    if index_name not in INDEX_URLS:
        raise ValueError(f"Unknown index: {index_name}")
    url = f"https://www.niftyindices.com/IndexConstituent/{INDEX_URLS[index_name]}"
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 200 and r.text[:50].startswith("Company"):
                return [row["Symbol"].strip()
                        for row in csv.DictReader(r.text.splitlines())]
            last_err = f"HTTP {r.status_code} or non-CSV body"
        except Exception as e:
            last_err = str(e)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {index_name}: {last_err}")


def fetch_constituent_prices(symbols: list, start: str = "2014-01-01") -> dict:
    """Pull daily close from yfinance for each symbol. Skip non-resolvable."""
    closes = {}
    for sym in symbols:
        try:
            df = yf.download(f"{sym}.NS", start=start, progress=False,
                             auto_adjust=True)
            if df.empty:
                print(f"  SKIP  {sym} — no yfinance data", file=sys.stderr)
                continue
            s = df["Close"].squeeze().dropna()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            closes[sym] = s
            print(f"  OK    {sym:14s} {len(s)} rows  ({s.index[0].date()} → {s.index[-1].date()})")
        except Exception as e:
            print(f"  ERR   {sym} — {e}", file=sys.stderr)
        time.sleep(0.1)
    return closes


def build_synthetic_index(closes: dict, base_value: float = 1000.0) -> pd.Series:
    """Equal-weight daily return → compound → rebase to base_value at first
    common date. Returns synthetic 'index' close series."""
    if not closes:
        return pd.Series(dtype=float)
    df = pd.DataFrame(closes).sort_index()
    # Daily returns per member; mean across available members each day
    rets = df.pct_change()
    # Equal-weight: simple mean of available constituent returns
    eq_ret = rets.mean(axis=1, skipna=True).fillna(0)
    # Find the first day with at least 3 members (avoid bootstrapping noise)
    first_day = (df.notna().sum(axis=1) >= 3).idxmax()
    eq_ret = eq_ret.loc[first_day:]
    # Compound to a level series
    level = (1 + eq_ret).cumprod() * base_value
    return level


def save(name: str, series: pd.Series):
    os.makedirs(CACHE, exist_ok=True)
    out = os.path.join(CACHE, f"synth_{name.lower()}_{date.today().isoformat()}.pkl")
    with open(out, "wb") as f:
        pickle.dump(series, f)
    print(f"\n  Saved {len(series)} rows  →  {out}")
    print(f"  Range  : {series.index[0].date()} → {series.index[-1].date()}")
    print(f"  Levels : start={series.iloc[0]:.1f}  recent={series.iloc[-1]:.1f}  "
          f"({(series.iloc[-1]/series.iloc[0]-1)*100:+.1f}% total)")
    # CAGR
    yrs = (series.index[-1] - series.index[0]).days / 365.25
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1/yrs) - 1 if yrs > 0 else 0
    print(f"  CAGR   : {cagr*100:+.2f}%  over {yrs:.2f} years")


def build_one(index_name: str):
    print(f"\n{'='*60}")
    print(f"  Building synthetic {index_name}")
    print(f"{'='*60}")
    syms = fetch_constituents(index_name)
    print(f"  Constituents ({len(syms)}): {', '.join(syms)}")
    print(f"\n  Fetching member price history from yfinance:")
    closes = fetch_constituent_prices(syms)
    if not closes:
        print(f"  ERROR: no member data fetched for {index_name}")
        return
    series = build_synthetic_index(closes)
    save(index_name, series)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("index", nargs="?", default=None,
                   help="One of: " + ", ".join(list(INDEX_URLS) + list(HARDCODED) + FROM_STOCK_MAP))
    p.add_argument("--all", action="store_true",
                   help="Build all known synthetic indices")
    args = p.parse_args()

    if args.all:
        for name in list(INDEX_URLS) + list(HARDCODED) + FROM_STOCK_MAP:
            try:
                build_one(name)
            except Exception as e:
                print(f"  FAILED {name}: {e}")
    elif args.index:
        build_one(args.index)
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
