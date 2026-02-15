"""
Support & Hammer Candle Screener Engine
Detects hammer candle patterns near support levels for Nifty 500 stocks.
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
    HAMMER_CACHE_FILE,
    get_stock_universe, get_cache_ttl, load_config
)
from nifty500_tickers import NIFTY_500_TICKERS


class HammerEngine:
    """Engine for support detection and hammer candle screening."""

    def __init__(self):
        self.cache_file = HAMMER_CACHE_FILE
        self._ensure_cache_dir()

    def _ensure_cache_dir(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

    def _fetch_price_data(self, symbol: str, days: int = 300) -> pd.DataFrame:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(days * 1.5))
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        return df

    # ---- Support Detection ----

    def _find_swing_lows(self, prices: pd.DataFrame, window: int = 5) -> List[Dict]:
        """Find swing lows: candle whose Low < Low of `window` candles on each side."""
        lows = prices["Low"].values
        swing_lows = []

        for i in range(window, len(lows) - window):
            is_swing = True
            for j in range(1, window + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing = False
                    break
            if is_swing:
                swing_lows.append({
                    "index": i,
                    "price": float(lows[i]),
                    "date": str(prices.index[i].date()),
                })

        return swing_lows

    def _cluster_swing_lows(self, swing_lows: List[Dict], cluster_pct: float = 1.5) -> List[Dict]:
        """
        Cluster swing lows within cluster_pct% of each other.
        Returns clusters sorted by number of touches (descending).
        """
        if not swing_lows:
            return []

        sorted_lows = sorted(swing_lows, key=lambda x: x["price"])
        clusters = []
        used = [False] * len(sorted_lows)

        for i, low in enumerate(sorted_lows):
            if used[i]:
                continue

            cluster_members = [low]
            used[i] = True

            for j in range(i + 1, len(sorted_lows)):
                if used[j]:
                    continue
                pct_diff = abs(sorted_lows[j]["price"] - low["price"]) / low["price"] * 100
                if pct_diff <= cluster_pct:
                    cluster_members.append(sorted_lows[j])
                    used[j] = True

            clusters.append({
                "level": sum(m["price"] for m in cluster_members) / len(cluster_members),
                "touches": len(cluster_members),
                "dates": [m["date"] for m in cluster_members],
            })

        return clusters

    def _get_support_levels(self, prices: pd.DataFrame, lookback_days: int) -> List[Dict]:
        """
        Get support levels from price data.
        Only keeps clusters with 2+ touches.
        Returns top 5 sorted by touches (strongest first).
        """
        if len(prices) < lookback_days:
            lookback_days = len(prices)

        recent = prices.tail(lookback_days)
        swing_lows = self._find_swing_lows(recent)
        clusters = self._cluster_swing_lows(swing_lows)

        # Require 2+ touches
        valid = [c for c in clusters if c["touches"] >= 2]
        # Sort by touches descending, keep top 5
        valid.sort(key=lambda x: x["touches"], reverse=True)
        return valid[:5]

    # ---- Hammer Candle Detection ----

    def _detect_hammer(self, candle: pd.Series, body_ratio: float,
                       lower_shadow_ratio: float, upper_shadow_max_pct: float) -> Optional[float]:
        """
        Check if a candle is a hammer pattern.
        Returns quality score (0.0-1.0) or None if not a hammer.
        """
        open_price = candle["Open"]
        high = candle["High"]
        low = candle["Low"]
        close = candle["Close"]

        total_range = high - low
        if total_range <= 0:
            return None

        body = abs(close - open_price)
        body_top = max(close, open_price)
        body_bottom = min(close, open_price)
        upper_shadow = high - body_top
        lower_shadow = body_bottom - low

        # Check hammer criteria
        # 1. Body size < body_ratio of total range
        if body / total_range > body_ratio:
            return None

        # 2. Lower shadow >= lower_shadow_ratio * body
        if body > 0 and lower_shadow / body < lower_shadow_ratio:
            return None
        elif body == 0 and lower_shadow < total_range * 0.5:
            # Doji-like: need significant lower shadow
            return None

        # 3. Upper shadow < upper_shadow_max_pct of total range
        if upper_shadow / total_range > upper_shadow_max_pct:
            return None

        # Calculate quality score
        # Higher score = smaller body, longer lower wick, smaller upper wick
        body_score = 1.0 - (body / total_range)  # smaller body = better
        if body > 0:
            shadow_score = min(lower_shadow / body / 4.0, 1.0)  # longer wick = better
        else:
            shadow_score = 0.8
        upper_score = 1.0 - (upper_shadow / total_range)  # less upper = better

        score = (body_score * 0.3 + shadow_score * 0.5 + upper_score * 0.2)
        return round(min(max(score, 0.0), 1.0), 2)

    # ---- Proximity Check ----

    def _check_proximity(self, current_price: float, support_levels: List[Dict],
                         proximity_pct: float) -> Dict:
        """
        Check price proximity to support levels.
        Returns dict with nearest support info and status.
        """
        if not support_levels:
            return {"status": "no_support", "nearest": None, "dist_pct": None, "touches": 0}

        best = None
        for level in support_levels:
            dist_pct = ((current_price - level["level"]) / level["level"]) * 100

            if best is None or abs(dist_pct) < abs(best["dist_pct"]):
                best = {
                    "level": round(level["level"], 2),
                    "dist_pct": round(dist_pct, 2),
                    "touches": level["touches"],
                }

        if best["dist_pct"] < 0:
            best["status"] = "broken"
        elif best["dist_pct"] <= proximity_pct:
            best["status"] = "near"
        else:
            best["status"] = "far"

        return best

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
        Run the support & hammer screener.

        Returns dict with 4 categories:
            - hammer_at_support: Hammer candle + near support level
            - near_support: Price near support, no hammer
            - hammer_only: Hammer detected, not near support
            - broken_support: Price fell below prior support
        """
        if not force_refresh:
            cache = self._load_cache()
            if cache:
                return cache["data"]

        config = load_config()
        stock_universe = get_stock_universe()
        lookback = config.get("support_lookback_days", 120)
        proximity_pct = config.get("support_proximity_pct", 3.0)
        body_ratio = config.get("hammer_body_ratio", 0.33)
        lower_shadow_ratio = config.get("hammer_lower_shadow_ratio", 2.0)
        upper_shadow_max_pct = config.get("hammer_upper_shadow_max_pct", 0.10)

        tickers = NIFTY_500_TICKERS[:stock_universe]

        hammer_at_support = []
        near_support = []
        hammer_only = []
        broken_support = []
        failed_tickers = []

        total = len(tickers)

        for i, ticker in enumerate(tickers):
            nse_symbol = f"{ticker}.NS"

            if progress_callback:
                progress_callback(i + 1, total, ticker)

            try:
                prices = self._fetch_price_data(nse_symbol, days=lookback + 50)

                if len(prices) < 30:
                    failed_tickers.append(ticker)
                    continue

                # Detect support levels
                support_levels = self._get_support_levels(prices, lookback)

                # Get latest candle
                latest = prices.iloc[-1]
                current_price = float(latest["Close"])

                # Detect hammer on latest candle
                hammer_score = self._detect_hammer(
                    latest, body_ratio, lower_shadow_ratio, upper_shadow_max_pct
                )

                # Check proximity to support
                proximity = self._check_proximity(current_price, support_levels, proximity_pct)

                is_hammer = hammer_score is not None
                is_near = proximity.get("status") == "near"
                is_broken = proximity.get("status") == "broken"

                stock_info = {
                    "ticker": ticker,
                    "price": round(current_price, 2),
                }

                if is_hammer and is_near:
                    stock_info.update({
                        "support": proximity["level"],
                        "dist_pct": proximity["dist_pct"],
                        "score": hammer_score,
                        "touches": proximity["touches"],
                    })
                    hammer_at_support.append(stock_info)
                elif is_near:
                    stock_info.update({
                        "support": proximity["level"],
                        "dist_pct": proximity["dist_pct"],
                        "touches": proximity["touches"],
                    })
                    near_support.append(stock_info)
                elif is_hammer:
                    stock_info.update({
                        "score": hammer_score,
                        "nearest_support": proximity.get("level"),
                        "dist_pct": proximity.get("dist_pct"),
                    })
                    hammer_only.append(stock_info)
                elif is_broken:
                    stock_info.update({
                        "broken_level": proximity["level"],
                        "below_pct": round(abs(proximity["dist_pct"]), 2),
                    })
                    broken_support.append(stock_info)

            except Exception:
                failed_tickers.append(ticker)

        # Sort panels
        hammer_at_support.sort(key=lambda x: x.get("score", 0), reverse=True)
        near_support.sort(key=lambda x: x.get("dist_pct", 999))
        hammer_only.sort(key=lambda x: x.get("score", 0), reverse=True)
        broken_support.sort(key=lambda x: x.get("below_pct", 0), reverse=True)

        result = {
            "hammer_at_support": hammer_at_support,
            "near_support": near_support,
            "hammer_only": hammer_only,
            "broken_support": broken_support,
            "last_updated": datetime.now().isoformat(),
            "settings": {
                "stock_universe": stock_universe,
                "support_lookback_days": lookback,
                "support_proximity_pct": proximity_pct,
            },
            "stats": {
                "total_processed": total - len(failed_tickers),
                "failed": len(failed_tickers),
            },
        }

        self._save_cache(result)
        return result
