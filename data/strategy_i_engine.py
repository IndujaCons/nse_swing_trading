"""
Strategy I (Support Bounce) Live Screener Engine
Scans stocks for proximity to 6M/1Y support levels with entry conditions.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import (
    STRATEGY_I_CACHE_FILE,
    get_cache_ttl, load_config
)
from nifty500_tickers import NIFTY_500_TICKERS

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

# Actual Nifty Next 50 constituents
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


class StrategyIEngine:
    """Engine for Strategy I (Support Bounce) live screening."""

    def __init__(self):
        self.cache_file = STRATEGY_I_CACHE_FILE
        self._ensure_cache_dir()

    def _ensure_cache_dir(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

    def _fetch_price_data(self, symbol: str, days: int = 400) -> pd.DataFrame:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(days * 1.5))
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        return df

    # ---- Caching ----

    def _is_cache_valid(self) -> bool:
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
        if not self._is_cache_valid():
            return None
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def _save_cache(self, data: Dict):
        cache = {
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        with open(self.cache_file, 'w') as f:
            json.dump(cache, f)

    def get_cache_age_minutes(self) -> Optional[float]:
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
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)

    # ---- Main Screener ----

    def run_screener(self, force_refresh: bool = False, progress_callback=None) -> Dict:
        """
        Run the Strategy I (Support Bounce) screener.

        Returns dict with 4 categories:
            - entry_signals: Near support + IBS > 0.3 + green candle (ALL conditions met)
            - near_6m_support: Low within 3% of 120-day low
            - near_1y_support: Low within 3% of 252-day low
            - below_support: Close below both support levels
        """
        if not force_refresh:
            cache = self._load_cache()
            if cache:
                return cache["data"]

        config = load_config()
        stock_universe = config.get("strategy_i_universe", 50)

        if stock_universe <= 50:
            tickers = NIFTY_50_TICKERS
        elif stock_universe <= 100:
            tickers = NIFTY_50_TICKERS + NIFTY_NEXT50_TICKERS
        else:
            tickers = NIFTY_500_TICKERS[:stock_universe]

        entry_signals = []
        near_6m_support = []
        near_1y_support = []
        below_support = []
        failed_tickers = []

        total = len(tickers)

        for i, ticker in enumerate(tickers):
            nse_symbol = f"{ticker}.NS"

            if progress_callback:
                progress_callback(i + 1, total, ticker)

            try:
                prices = self._fetch_price_data(nse_symbol, days=400)

                if len(prices) < 60:
                    failed_tickers.append(ticker)
                    continue

                # Compute support from prior bars (excluding today)
                hist = prices.iloc[:-1]
                prior_support_6m = float(hist["Low"].rolling(window=120, min_periods=60).min().iloc[-1])
                prior_support_1y = float(hist["Low"].rolling(window=252, min_periods=120).min().iloc[-1])

                # Latest candle
                latest = prices.iloc[-1]
                close = float(latest["Close"])
                low = float(latest["Low"])
                open_price = float(latest["Open"])
                high = float(latest["High"])

                # IBS = (Close - Low) / (High - Low)
                candle_range = high - low
                ibs = (close - low) / candle_range if candle_range > 0 else 0.5

                is_green = close > open_price

                # Today's low can update support only if candle is green
                # (green candle dipping below prior support = new support floor)
                support_6m = min(prior_support_6m, low) if is_green else prior_support_6m
                support_1y = min(prior_support_1y, low) if is_green else prior_support_1y

                # Distance from CLOSE to support levels (%)
                dist_6m_pct = ((close - support_6m) / support_6m) * 100 if support_6m > 0 else 999
                dist_1y_pct = ((close - support_1y) / support_1y) * 100 if support_1y > 0 else 999

                # Determine which support close is near (within 3%)
                near_6m = 0 <= dist_6m_pct <= 3.0
                near_1y = 0 <= dist_1y_pct <= 3.0
                close_above_6m = close > support_6m
                close_above_1y = close > support_1y

                # Entry: close within 1% above support, close > support, IBS > 0.3, green
                near_support_entry = (0 < dist_6m_pct <= 1.0) or \
                                     (0 < dist_1y_pct <= 1.0)
                close_above_support = close_above_6m or close_above_1y

                which_support = []
                if near_6m:
                    which_support.append("6M")
                if near_1y:
                    which_support.append("1Y")

                stock_info = {
                    "ticker": ticker,
                    "price": round(close, 2),
                    "low": round(low, 2),
                    "support_6m": round(support_6m, 2),
                    "support_1y": round(support_1y, 2),
                    "dist_6m_pct": round(dist_6m_pct, 2),
                    "dist_1y_pct": round(dist_1y_pct, 2),
                    "ibs": round(ibs, 2),
                    "is_green": is_green,
                    "which_support": ", ".join(which_support) if which_support else "--",
                }

                # Categorize
                # Entry Signals: near support (1%), close > support, IBS > 0.3, green candle
                if near_support_entry and close_above_support and ibs > 0.3 and is_green:
                    entry_signals.append(stock_info)
                elif dist_6m_pct < 0 and dist_1y_pct < 0:
                    # Below both support levels
                    below_support.append(stock_info)
                else:
                    # Watchlist panels
                    if near_6m:
                        near_6m_support.append(stock_info)
                    if near_1y:
                        near_1y_support.append(stock_info)

            except Exception:
                failed_tickers.append(ticker)

        # Sort panels
        entry_signals.sort(key=lambda x: x.get("ibs", 0), reverse=True)
        near_6m_support.sort(key=lambda x: x.get("dist_6m_pct", 999))
        near_1y_support.sort(key=lambda x: x.get("dist_1y_pct", 999))
        below_support.sort(key=lambda x: min(abs(x.get("dist_6m_pct", 0)), abs(x.get("dist_1y_pct", 0))))

        result = {
            "entry_signals": entry_signals,
            "near_6m_support": near_6m_support,
            "near_1y_support": near_1y_support,
            "below_support": below_support,
            "last_updated": datetime.now().isoformat(),
            "settings": {
                "stock_universe": stock_universe,
            },
            "stats": {
                "total_processed": total - len(failed_tickers),
                "failed": len(failed_tickers),
            },
        }

        self._save_cache(result)
        return result
