#!/usr/bin/env python3
"""
Scrape historical EPS from Screener.in (no login required for annual data).
Builds a JSON database of quarterly + annual Diluted EPS for all PIT Nifty 200 stocks.

Output: data/quarterly_eps.json
Format: {
  "RELIANCE": {
    "quarterly": {"Dec 2022": 11.67, "Mar 2023": 14.26, ...},
    "annual": {"Mar 2014": 16.31, "Mar 2015": 17.07, ...}
  }, ...
}

Usage: python3 data/scrape_screener_eps.py
"""

import json
import os
import sys
import time

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "quarterly_eps.json")
DELAY_SECONDS = 1.5  # be polite to Screener's servers
BASE_URL = "https://www.screener.in"

# Alias map: PIT ticker → Screener company name
# For tickers that were renamed/merged and exist on Screener under a different name
SCREENER_ALIASES = {
    "ZOMATO": "ETERNAL",
    "CROMPGREAV": "CGPOWER",
    "GMRINFRA": "GMRAIRPORT",
    "IDFCBANK": "IDFCFIRSTB",
    "MAX": "MFSL",
    "NIITTECH": "COFORGE",
    "RNAM": "NAM-INDIA",
    "STRTECH": "STLTECH",
    "WELSPUNIND": "WELSPUNLIV",
    "JUBILANT": "JUBLPHARMA",
    "LTI": "LTIM",
    "MCDOWELL-N": "UNITDSPR",
    "KPIT": "BSOFT",
    "IBULHSGFIN": "PIRAMALENT",     # Indiabulls Housing → Piramal
    "DALMIABHA": "DALBHARAT",       # Dalmia Bharat
}


def fetch_eps(session, ticker):
    """Fetch quarterly + annual EPS for a stock. Returns dict or None."""
    # Try consolidated first, then standalone
    for suffix in ["/consolidated/", "/"]:
        url = f"{BASE_URL}/company/{ticker}{suffix}"
        try:
            r = session.get(url, timeout=30)
        except requests.RequestException as e:
            return None

        if r.status_code == 404:
            continue
        if r.status_code != 200:
            return None

        soup = BeautifulSoup(r.text, "html.parser")
        tables = soup.find_all("table", class_="data-table")
        if len(tables) < 2:
            continue

        result = {"quarterly": {}, "annual": {}}

        # Table 0: Quarterly results
        # Table 1: Annual P&L
        for table_idx, key in [(0, "quarterly"), (1, "annual")]:
            if table_idx >= len(tables):
                continue
            table = tables[table_idx]
            rows = table.find_all("tr")
            if len(rows) < 3:
                continue

            # Header row: period dates
            header_cells = [td.text.strip() for td in rows[0].find_all(["th", "td"])]
            periods = header_cells[1:]

            # Find EPS row
            for row in rows:
                cells = [td.text.strip() for td in row.find_all(["th", "td"])]
                if cells and "EPS" in cells[0]:
                    eps_values = cells[1:]
                    for p, v in zip(periods, eps_values):
                        if not p or not v or p == "TTM":
                            continue
                        try:
                            result[key][p] = float(v.replace(",", "").strip())
                        except (ValueError, TypeError):
                            continue
                    break

        if result["quarterly"] or result["annual"]:
            return result

    return None


def main():
    from momentum_backtest import load_pit_nifty200, get_all_pit_tickers

    pit_data = load_pit_nifty200()
    if pit_data is None:
        print("ERROR: PIT data not found!")
        return

    all_tickers = sorted(get_all_pit_tickers(pit_data))
    print(f"PIT universe: {len(all_tickers)} unique tickers\n")

    # Load existing data (resume support)
    existing = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            existing = json.load(f)
        # Migrate old format (flat dict) to new format (quarterly+annual)
        migrated = 0
        for t in list(existing.keys()):
            if isinstance(existing[t], dict) and "quarterly" not in existing[t]:
                existing[t] = {"quarterly": existing[t], "annual": {}}
                migrated += 1
        if migrated:
            print(f"Migrated {migrated} stocks from old format")
        print(f"Loaded {len(existing)} stocks from existing file")

    # Setup session (no login needed — annual data is public)
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0",
    })

    # Verify with RELIANCE
    test = fetch_eps(session, "RELIANCE")
    if test:
        print(f"Verification: RELIANCE")
        print(f"  Quarterly: {len(test['quarterly'])} quarters ({sorted(test['quarterly'].keys())[0]} → {sorted(test['quarterly'].keys())[-1]})")
        print(f"  Annual:    {len(test['annual'])} years ({sorted(test['annual'].keys())[0]} → {sorted(test['annual'].keys())[-1]})")
    else:
        print("ERROR: Could not fetch RELIANCE")
        return

    # Build scrape list: all tickers + alias lookups
    scrape_list = []
    for ticker in all_tickers:
        # Skip if already have annual data with sufficient depth
        if ticker in existing:
            ex = existing[ticker]
            if isinstance(ex, dict) and "annual" in ex and len(ex.get("annual", {})) >= 8:
                continue

        # Determine Screener name
        screener_name = SCREENER_ALIASES.get(ticker, ticker)
        scrape_list.append((ticker, screener_name))

    print(f"\nTo scrape: {len(scrape_list)} stocks (skipping {len(all_tickers) - len(scrape_list)} already done)")
    print(f"Estimated time: {len(scrape_list) * DELAY_SECONDS / 60:.0f} minutes\n")

    success = 0
    failed = []

    for i, (pit_ticker, screener_name) in enumerate(scrape_list):
        eps = fetch_eps(session, screener_name)

        if eps and (eps["quarterly"] or eps["annual"]):
            existing[pit_ticker] = eps
            success += 1
            nq = len(eps["quarterly"])
            na = len(eps["annual"])
            annual_range = ""
            if eps["annual"]:
                dates = sorted(eps["annual"].keys())
                annual_range = f" ({dates[0]} → {dates[-1]})"
            alias_tag = f" (as {screener_name})" if screener_name != pit_ticker else ""
            print(f"  [{i+1}/{len(scrape_list)}] {pit_ticker:<20} Q:{nq:>2}  A:{na:>2}{annual_range}{alias_tag}")
        else:
            failed.append(pit_ticker)
            alias_tag = f" (tried {screener_name})" if screener_name != pit_ticker else ""
            print(f"  [{i+1}/{len(scrape_list)}] {pit_ticker:<20} FAILED{alias_tag}")

        # Save periodically
        if (i + 1) % 25 == 0:
            with open(OUTPUT_FILE, "w") as f:
                json.dump(existing, f, indent=2, sort_keys=True)

        time.sleep(DELAY_SECONDS)

    # Final save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(existing, f, indent=2, sort_keys=True)

    # Summary
    print(f"\n{'=' * 50}")
    print(f"DONE")
    print(f"  Success:    {success}")
    print(f"  Failed:     {len(failed)}")
    print(f"  Total DB:   {len(existing)} stocks")
    print(f"  Output:     {OUTPUT_FILE}")

    if failed:
        print(f"\n  Failed tickers ({len(failed)}):")
        for t in failed:
            print(f"    {t}")


if __name__ == "__main__":
    main()
