"""
RS/EMA Screener Engine
Calculates relative strength and EMA positions for Nifty 500 stocks
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
    get_stock_universe, get_rs_period, get_ema_period, get_cache_ttl
)
from nifty500_tickers import NIFTY_500_TICKERS
from sector_mapping import STOCK_SECTOR_MAP

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
from data.state_tracker import StateTracker


class ScreenerEngine:
    """Core engine for RS screening calculations."""

    def __init__(self):
        self.state_tracker = StateTracker()
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
        Calculate EMA and related data including yesterday's position.

        Returns dict with:
            - current_price
            - ema_value
            - price_vs_ema_percent
            - today_above_ema
            - yesterday_above_ema
            - yesterday_close
            - yesterday_ema
        """
        if len(prices) < period + 1:
            return None

        close_prices = prices["Close"]
        ema = close_prices.ewm(span=period, adjust=False).mean()

        # Today's values
        current_price = close_prices.iloc[-1]
        current_ema = ema.iloc[-1]
        price_vs_ema = ((current_price - current_ema) / current_ema) * 100
        today_above_ema = current_price > current_ema

        # Yesterday's values
        yesterday_close = close_prices.iloc[-2]
        yesterday_ema = ema.iloc[-2]
        yesterday_above_ema = yesterday_close > yesterday_ema

        return {
            "current_price": float(current_price),
            "ema_value": float(current_ema),
            "price_vs_ema_percent": float(price_vs_ema),
            "today_above_ema": bool(today_above_ema),
            "yesterday_above_ema": bool(yesterday_above_ema),
            "yesterday_close": float(yesterday_close),
            "yesterday_ema": float(yesterday_ema)
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

    def run_screener(self, force_refresh: bool = False, progress_callback=None) -> Dict:
        """
        Run the screener and categorize stocks.

        Args:
            force_refresh: Force refresh even if cache is valid
            progress_callback: Optional callback for progress updates (current, total, ticker)

        Returns:
            Dict with categorized stocks:
                - fresh_entries: Stocks that just crossed above EMA with RS > 0
                - in_rs: Stocks already above EMA with RS > 0
                - exit_signals: Stocks that just crossed below EMA
                - nifty50_return: Index return for reference
                - last_updated: Timestamp
        """
        # Check cache first
        if not force_refresh:
            cache = self._load_cache()
            if cache:
                return cache["data"]

        # Get current settings
        stock_universe = get_stock_universe()
        rs_period = get_rs_period()
        ema_period = get_ema_period()

        # Limit tickers based on universe setting
        tickers = NIFTY_500_TICKERS[:stock_universe]

        # Fetch Nifty 50 data
        nifty50_data = self._fetch_price_data(NIFTY_50_SYMBOL)
        nifty50_return = self._calculate_return(nifty50_data, rs_period)

        if nifty50_return is None:
            raise RuntimeError("Could not fetch Nifty 50 data")

        # Calculate sectoral indices FIRST (needed for sector strength lookup)
        sectoral_indices = []
        sector_rs_lookup = {}
        for name, symbol in SECTORAL_INDICES.items():
            try:
                index_data = self._fetch_price_data(symbol)
                index_return = self._calculate_return(index_data, rs_period)
                index_ema = self._calculate_ema_data(index_data, ema_period)

                if index_return is not None and index_ema is not None:
                    rs = index_return - nifty50_return
                    sector_rs_lookup[name] = round(rs, 2)
                    sectoral_indices.append({
                        "ticker": name,
                        "price": round(index_ema["current_price"], 2),
                        "ema": round(index_ema["ema_value"], 2),
                        "price_vs_ema": round(index_ema["price_vs_ema_percent"], 2),
                        "stock_return": round(index_return, 2),
                        "rs": round(rs, 2)
                    })
            except:
                pass

        # Sort sectoral indices by RS (highest first)
        sectoral_indices.sort(key=lambda x: x["rs"], reverse=True)

        fresh_entries = []
        in_rs = []
        exit_signals = []
        failed_tickers = []

        total = len(tickers)
        today_positions = {}

        for i, ticker in enumerate(tickers):
            nse_symbol = f"{ticker}.NS"

            if progress_callback:
                progress_callback(i + 1, total, ticker)

            try:
                stock_data = self._fetch_price_data(nse_symbol)
                stock_return = self._calculate_return(stock_data, rs_period)
                ema_data = self._calculate_ema_data(stock_data, ema_period)

                if stock_return is None or ema_data is None:
                    failed_tickers.append(ticker)
                    continue

                rs = stock_return - nifty50_return

                # Save today's position for state tracking
                today_positions[ticker] = ema_data["today_above_ema"]

                # Get sector info
                sector_name = STOCK_SECTOR_MAP.get(ticker, None)
                sector_rs = sector_rs_lookup.get(sector_name, None) if sector_name else None

                stock_info = {
                    "ticker": ticker,
                    "price": round(ema_data["current_price"], 2),
                    "ema": round(ema_data["ema_value"], 2),
                    "price_vs_ema": round(ema_data["price_vs_ema_percent"], 2),
                    "stock_return": round(stock_return, 2),
                    "rs": round(rs, 2),
                    "sector": sector_name,
                    "sector_rs": sector_rs
                }

                today_above = ema_data["today_above_ema"]
                yesterday_above = ema_data["yesterday_above_ema"]

                # Categorize based on state transitions
                if today_above and not yesterday_above and rs > 0:
                    # Fresh entry: just crossed above EMA with positive RS
                    fresh_entries.append(stock_info)
                elif today_above and yesterday_above and rs > 0:
                    # In RS: already above EMA with positive RS
                    in_rs.append(stock_info)
                elif not today_above and yesterday_above:
                    # Exit signal: just crossed below EMA (show regardless of RS)
                    exit_signals.append(stock_info)
                # Stocks with RS <= 0 that are above EMA are not shown
                # Stocks that were below and remain below are not shown

            except Exception as e:
                failed_tickers.append(ticker)

        # Sort all panels by RS (highest first)
        fresh_entries.sort(key=lambda x: x["rs"], reverse=True)
        in_rs.sort(key=lambda x: x["rs"], reverse=True)
        exit_signals.sort(key=lambda x: x["rs"], reverse=True)

        # Save today's positions for tomorrow's comparison
        self.state_tracker.save_state(today_positions)

        result = {
            "fresh_entries": fresh_entries,
            "sectoral_indices": sectoral_indices,
            "in_rs": in_rs,
            "exit_signals": exit_signals,
            "nifty50_return": round(nifty50_return, 2),
            "last_updated": datetime.now().isoformat(),
            "settings": {
                "stock_universe": stock_universe,
                "rs_period": rs_period,
                "ema_period": ema_period
            },
            "stats": {
                "total_processed": total - len(failed_tickers),
                "failed": len(failed_tickers)
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
