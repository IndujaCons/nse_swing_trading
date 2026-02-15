#!/usr/bin/env python3
"""
Nifty 500 Relative Strength Screener
Calculates 63-day relative strength vs Nifty 50 index
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sys

from nifty500_tickers import NIFTY_500_TICKERS

# Configuration
RS_PERIOD = 63  # Trading days
EMA_PERIOD = 63  # EMA period
NIFTY_50_SYMBOL = "^NSEI"
OUTPUT_FILE = "relative_strength_results.csv"


def fetch_price_data(symbol: str, days: int = 100) -> pd.DataFrame:
    """Fetch historical price data for a symbol."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days * 2)  # Extra buffer for trading days

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)
    return df


def calculate_return(prices: pd.DataFrame, days: int) -> float:
    """Calculate percentage return over specified trading days."""
    if len(prices) < days:
        return None

    recent_prices = prices.tail(days + 1)
    if len(recent_prices) < days + 1:
        return None

    start_price = recent_prices.iloc[0]["Close"]
    end_price = recent_prices.iloc[-1]["Close"]

    return ((end_price - start_price) / start_price) * 100


def calculate_ema(prices: pd.DataFrame, period: int) -> tuple:
    """Calculate EMA and return (current_price, ema_value, price_vs_ema_percent)."""
    if len(prices) < period:
        return None, None, None

    close_prices = prices["Close"]
    ema = close_prices.ewm(span=period, adjust=False).mean()

    current_price = close_prices.iloc[-1]
    current_ema = ema.iloc[-1]
    price_vs_ema = ((current_price - current_ema) / current_ema) * 100

    return current_price, current_ema, price_vs_ema


def run_screener():
    """Main screener function."""
    print(f"Nifty 500 Relative Strength Screener ({RS_PERIOD}-day)")
    print("=" * 60)

    # Fetch Nifty 50 index data
    print("\nFetching Nifty 50 index data...")
    nifty50_data = fetch_price_data(NIFTY_50_SYMBOL)
    nifty50_return = calculate_return(nifty50_data, RS_PERIOD)

    if nifty50_return is None:
        print("Error: Could not fetch Nifty 50 data")
        sys.exit(1)

    print(f"Nifty 50 {RS_PERIOD}-day return: {nifty50_return:.2f}%")
    print("\nFetching stock data...")

    results = []
    total = len(NIFTY_500_TICKERS)
    failed = []

    for i, ticker in enumerate(NIFTY_500_TICKERS, 1):
        nse_symbol = f"{ticker}.NS"

        # Progress indicator
        progress = (i / total) * 100
        print(f"\r  Processing: {i}/{total} ({progress:.1f}%) - {ticker}          ", end="")

        try:
            stock_data = fetch_price_data(nse_symbol)
            stock_return = calculate_return(stock_data, RS_PERIOD)
            current_price, ema_50, price_vs_ema = calculate_ema(stock_data, EMA_PERIOD)

            if stock_return is not None and ema_50 is not None:
                rs = stock_return - nifty50_return
                results.append({
                    "Ticker": ticker,
                    "Price": round(current_price, 2),
                    "EMA_50": round(ema_50, 2),
                    "Price_vs_EMA_%": round(price_vs_ema, 2),
                    "Stock_Return_%": round(stock_return, 2),
                    "Nifty50_Return_%": round(nifty50_return, 2),
                    "Relative_Strength": round(rs, 2)
                })
            else:
                failed.append(ticker)
        except Exception as e:
            failed.append(ticker)

    print("\n")

    # Create DataFrame and sort by RS
    df = pd.DataFrame(results)
    df = df.sort_values("Relative_Strength", ascending=False)
    df = df.reset_index(drop=True)
    df.index = df.index + 1  # 1-based ranking

    # Display results
    print("=" * 60)
    print(f"TOP 20 STOCKS BY RELATIVE STRENGTH")
    print("=" * 60)
    print(df.head(20).to_string())

    print("\n" + "=" * 60)
    print(f"BOTTOM 20 STOCKS BY RELATIVE STRENGTH")
    print("=" * 60)
    print(df.tail(20).to_string())

    # Save to CSV
    df.to_csv(OUTPUT_FILE, index_label="Rank")
    print(f"\nFull results saved to: {OUTPUT_FILE}")

    # Summary
    print(f"\nSummary:")
    print(f"  - Stocks processed: {len(results)}")
    print(f"  - Failed to fetch: {len(failed)}")
    if failed:
        print(f"  - Failed tickers: {', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}")


if __name__ == "__main__":
    run_screener()
