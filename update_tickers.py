#!/usr/bin/env python3
"""
Fetches the latest Nifty 500 constituents from NSE India website.
Run this weekly to keep the ticker list updated.
"""

import requests
import json
from datetime import datetime


def fetch_nifty500_tickers():
    """Fetch Nifty 500 tickers from NSE India."""

    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500"

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/market-data/live-equity-market?symbol=NIFTY%20500",
    }

    # NSE requires a session with cookies
    session = requests.Session()

    # First hit the main page to get cookies
    session.get("https://www.nseindia.com", headers=headers)

    # Now fetch the data
    response = session.get(url, headers=headers, timeout=30)

    if response.status_code != 200:
        print(f"Error: HTTP {response.status_code}")
        return None

    data = response.json()

    tickers = []
    for stock in data.get("data", []):
        symbol = stock.get("symbol")
        if symbol and symbol != "NIFTY 500":  # Exclude index itself
            tickers.append(symbol)

    return sorted(tickers)


def save_tickers(tickers):
    """Save tickers to nifty500_tickers.py"""

    # Format tickers into lines
    lines = []
    line = "    "
    for i, ticker in enumerate(tickers):
        line += f'"{ticker}", '
        if (i + 1) % 7 == 0:  # 7 tickers per line
            lines.append(line)
            line = "    "
    if line.strip():
        lines.append(line)

    content = f'''# Nifty 500 Stock Symbols
# Auto-generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}
# These are NSE symbols without the .NS suffix (added by screener)

NIFTY_500_TICKERS = [
{chr(10).join(lines)}
]
'''

    with open("nifty500_tickers.py", "w") as f:
        f.write(content)

    print(f"Saved {len(tickers)} tickers to nifty500_tickers.py")


def main():
    print("Fetching Nifty 500 constituents from NSE India...")

    tickers = fetch_nifty500_tickers()

    if tickers:
        print(f"Found {len(tickers)} stocks")
        save_tickers(tickers)
        print("Done! Run 'python screener.py' to screen stocks.")
    else:
        print("Failed to fetch tickers. Try again later or update manually.")
        print("You can also download from: https://www.nseindia.com/market-data/live-equity-market?symbol=NIFTY%20500")


if __name__ == "__main__":
    main()
