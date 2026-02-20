"""
Watchlist Engine
Calculates relative strength and EMA positions for Nifty 50 / Next 50 / Midcap 150 stocks
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import (
    NIFTY_50_SYMBOL, SCREENER_CACHE_FILE,
    get_rs_period, get_ema_period, get_cache_ttl
)
from nifty500_tickers import NIFTY_500_TICKERS

# Sectoral Indices (Yahoo Finance symbols)
SECTORAL_INDICES = {
    "NIFTY BANK": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY METAL": "^CNXMETAL",
    "NIFTY REALTY": "^CNXREALTY",
    "NIFTY ENERGY": "^CNXENERGY",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY MEDIA": "^CNXMEDIA",
    "NIFTY PSU BANK": "^CNXPSUBANK",
    "NIFTY INFRA": "^CNXINFRA",
    "NIFTY FIN SERVICE": "^CNXFINANCE",
    "NIFTY PVT BANK": "^CNXPVTBANK",
    "NIFTY COMMODITIES": "^CNXCOMMODITIES",
    "NIFTY CONSUMPTION": "^CNXCONSUMPTION",
    "NIFTY HEALTHCARE": "^CNXHEALTH",
    "NIFTY MNC": "^CNXMNC",
    "NIFTY PSE": "^CNXPSE",
}

# Actual Nifty 50 constituents
NIFTY_50_TICKERS = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BPCL",
    "BHARTIARTL", "BRITANNIA", "CIPLA", "COALINDIA", "DRREDDY",
    "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH", "HDFCBANK",
    "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK",
    "ITC", "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK",
    "LT", "M&M", "MARUTI", "NTPC", "NESTLEIND",
    "ONGC", "PIDILITIND", "POWERGRID", "RELIANCE", "SBILIFE",
    "SBIN", "SUNPHARMA", "TCS", "TATACONSUM",
    "TATASTEEL", "TECHM", "TITAN", "ULTRACEMCO", "WIPRO",
]

NIFTY_NEXT50_TICKERS = [
    "ABB", "ADANIENSOL", "ADANIGREEN", "ADANIPOWER", "AMBUJACEM",
    "ATGL", "AUROPHARMA", "BAJAJHLDNG", "BANKBARODA", "BHEL",
    "BOSCHLTD", "CANBK", "CHOLAFIN", "COLPAL", "DABUR",
    "DLF", "GAIL", "GODREJCP", "HAL", "HAVELLS",
    "ICICIPRULI", "ICICIGI", "INDIGO", "INDUSTOWER", "IOC",
    "IRFC", "JIOFIN", "JSWENERGY", "LICI", "LODHA",
    "LUPIN", "MANKIND", "MARICO", "MOTHERSON", "NAUKRI",
    "NHPC", "PFC", "PNB", "POLYCAB", "RECLTD",
    "SHREECEM", "SIEMENS", "SRF", "TATAPOWER", "TORNTPHARM",
    "TRENT", "TVSMOTOR", "UNITDSPR", "VEDL", "ZOMATO",
]

# Midcap 150: Nifty 500 stocks outside the top 100
_nifty100_set = set(NIFTY_50_TICKERS + NIFTY_NEXT50_TICKERS)
MIDCAP_150_TICKERS = [t for t in NIFTY_500_TICKERS if t not in _nifty100_set][:150]


class ScreenerEngine:
    """Core engine for watchlist RS calculations."""

    def __init__(self):
        self.cache_file = SCREENER_CACHE_FILE
        self._ensure_cache_dir()

    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

    def _fetch_price_data(self, symbol: str, days: int = 150) -> pd.DataFrame:
        """Fetch historical price data for a symbol."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days * 2)

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        return df

    def _calculate_return(self, prices: pd.DataFrame, days: int) -> Optional[float]:
        """Calculate percentage return over specified trading days."""
        if len(prices) < days:
            return None

        recent_prices = prices.tail(days + 1)
        if len(recent_prices) < days + 1:
            return None

        start_price = recent_prices.iloc[0]["Close"]
        end_price = recent_prices.iloc[-1]["Close"]

        return ((end_price - start_price) / start_price) * 100

    def _calculate_ema_data(self, prices: pd.DataFrame, period: int) -> Dict:
        """
        Calculate EMA and related data.

        Returns dict with:
            - current_price
            - ema_value
            - price_vs_ema_percent
            - pct_change (today vs yesterday)
        """
        if len(prices) < period + 1:
            return None

        close_prices = prices["Close"]
        ema = close_prices.ewm(span=period, adjust=False).mean()

        # Today's values
        current_price = close_prices.iloc[-1]
        current_ema = ema.iloc[-1]
        price_vs_ema = ((current_price - current_ema) / current_ema) * 100

        # Pct change: today vs yesterday
        yesterday_close = close_prices.iloc[-2]
        pct_change = ((current_price - yesterday_close) / yesterday_close) * 100

        return {
            "current_price": float(current_price),
            "ema_value": float(current_ema),
            "price_vs_ema_percent": float(price_vs_ema),
            "pct_change": float(pct_change),
        }

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if not os.path.exists(self.cache_file):
            return False

        try:
            with open(self.cache_file, 'r') as f:
                cache = json.load(f)

            cache_time = datetime.fromisoformat(cache["timestamp"])
            age_minutes = (datetime.now() - cache_time).total_seconds() / 60
            return age_minutes < get_cache_ttl()
        except (json.JSONDecodeError, KeyError, IOError):
            return False

    def _load_cache(self) -> Optional[Dict]:
        """Load cached results."""
        if not self._is_cache_valid():
            return None

        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def _save_cache(self, data: Dict):
        """Save results to cache."""
        cache = {
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        with open(self.cache_file, 'w') as f:
            json.dump(cache, f)

    def _scan_group(self, tickers, nifty50_return, rs_period, ema_period, progress_callback, progress_offset, total_all):
        """Scan a group of tickers and return stock dicts + failed count."""
        results = []
        failed = 0

        for i, ticker in enumerate(tickers):
            nse_symbol = f"{ticker}.NS"

            if progress_callback:
                progress_callback(progress_offset + i + 1, total_all, ticker)

            try:
                stock_data = self._fetch_price_data(nse_symbol)
                stock_return = self._calculate_return(stock_data, rs_period)
                ema_data = self._calculate_ema_data(stock_data, ema_period)

                if stock_return is None or ema_data is None:
                    failed += 1
                    continue

                rs = stock_return - nifty50_return

                results.append({
                    "ticker": ticker,
                    "price": round(ema_data["current_price"], 2),
                    "pct_change": round(ema_data["pct_change"], 2),
                    "rs": round(rs, 2),
                    "price_vs_ema": round(ema_data["price_vs_ema_percent"], 2),
                })
            except Exception:
                failed += 1

        # Sort by RS descending
        results.sort(key=lambda x: x["rs"], reverse=True)
        return results, failed

    def run_screener(self, force_refresh: bool = False, progress_callback=None) -> Dict:
        """
        Run the watchlist screener across all 3 groups + sectoral indices.

        Returns:
            Dict with nifty50, nifty100, midcap150, sectoral_indices lists
        """
        # Check cache first
        if not force_refresh:
            cache = self._load_cache()
            if cache:
                return cache["data"]

        rs_period = get_rs_period()
        ema_period = get_ema_period()

        # Fetch Nifty 50 index data
        nifty50_data = self._fetch_price_data(NIFTY_50_SYMBOL)
        nifty50_return = self._calculate_return(nifty50_data, rs_period)

        if nifty50_return is None:
            raise RuntimeError("Could not fetch Nifty 50 data")

        # Calculate sectoral indices
        sectoral_indices = []
        for name, symbol in SECTORAL_INDICES.items():
            try:
                index_data = self._fetch_price_data(symbol)
                index_return = self._calculate_return(index_data, rs_period)
                index_ema = self._calculate_ema_data(index_data, ema_period)

                if index_return is not None and index_ema is not None:
                    rs = index_return - nifty50_return
                    sectoral_indices.append({
                        "ticker": name,
                        "price": round(index_ema["current_price"], 2),
                        "pct_change": round(index_ema["pct_change"], 2),
                        "rs": round(rs, 2),
                        "price_vs_ema": round(index_ema["price_vs_ema_percent"], 2),
                    })
            except Exception:
                pass

        sectoral_indices.sort(key=lambda x: x["rs"], reverse=True)

        # Scan all 3 stock groups
        total_all = len(NIFTY_50_TICKERS) + len(NIFTY_NEXT50_TICKERS) + len(MIDCAP_150_TICKERS)
        total_failed = 0

        nifty50, f1 = self._scan_group(
            NIFTY_50_TICKERS, nifty50_return, rs_period, ema_period,
            progress_callback, 0, total_all)
        total_failed += f1

        nifty100, f2 = self._scan_group(
            NIFTY_NEXT50_TICKERS, nifty50_return, rs_period, ema_period,
            progress_callback, len(NIFTY_50_TICKERS), total_all)
        total_failed += f2

        midcap150, f3 = self._scan_group(
            MIDCAP_150_TICKERS, nifty50_return, rs_period, ema_period,
            progress_callback, len(NIFTY_50_TICKERS) + len(NIFTY_NEXT50_TICKERS), total_all)
        total_failed += f3

        result = {
            "sectoral_indices": sectoral_indices,
            "nifty50": nifty50,
            "nifty100": nifty100,
            "midcap150": midcap150,
            "nifty50_return": round(nifty50_return, 2),
            "last_updated": datetime.now().isoformat(),
            "settings": {
                "rs_period": rs_period,
                "ema_period": ema_period
            },
            "stats": {
                "total_processed": total_all - total_failed,
                "failed": total_failed
            }
        }

        # Cache results
        self._save_cache(result)

        return result

    def get_cache_age_minutes(self) -> Optional[float]:
        """Get the age of cached data in minutes."""
        if not os.path.exists(self.cache_file):
            return None

        try:
            with open(self.cache_file, 'r') as f:
                cache = json.load(f)
            cache_time = datetime.fromisoformat(cache["timestamp"])
            return (datetime.now() - cache_time).total_seconds() / 60
        except:
            return None

    def clear_cache(self):
        """Clear the cached data."""
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
