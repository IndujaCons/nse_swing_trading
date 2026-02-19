"""
Momentum Scanner Engine
Multi-timeframe momentum strategy: Daily RSI(2), 30-min RSI(14), EMA(50), EMA(200).
All 4 conditions must be TRUE for a buy signal.
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
    MOMENTUM_CACHE_FILE, MOMENTUM_STATE_FILE,
    get_cache_ttl, load_config
)

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


class MomentumEngine:
    """Engine for multi-timeframe momentum scanning."""

    def __init__(self):
        self.cache_file = MOMENTUM_CACHE_FILE
        self.state_file = MOMENTUM_STATE_FILE
        self._ensure_cache_dir()

    def _ensure_cache_dir(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

    # ---- Data Fetching ----

    def _is_session_complete(self) -> bool:
        """Check if today's session is complete (past 15:15 IST).
        At 15:15 the last full 30-min candle (14:45-15:15) closes
        and the daily candle is effectively final."""
        import pytz
        ist = pytz.timezone("Asia/Kolkata")
        now_ist = datetime.now(ist)
        return now_ist.hour > 15 or (now_ist.hour == 15 and now_ist.minute >= 15)

    def _fetch_daily_data(self, symbol: str) -> pd.DataFrame:
        """Fetch daily candles. Before 15:15 IST, drops today's
        incomplete candle so the system uses previous session data."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=500)
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        if df.empty:
            return df
        if not self._is_session_complete():
            import pytz
            ist = pytz.timezone("Asia/Kolkata")
            today = datetime.now(ist).date()
            last_date = df.index[-1].date()
            if last_date == today:
                df = df.iloc[:-1]
        return df

    def _fetch_intraday_data(self, symbol: str) -> pd.DataFrame:
        """Fetch 30-min candles. Before 15:15 IST, drops ALL of
        today's candles so the system uses previous session's
        last candle (14:45-15:15 of prior trading day)."""
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="5d", interval="30m")
        if df.empty:
            return df
        if not self._is_session_complete():
            # Drop all of today's intraday candles
            import pytz
            ist = pytz.timezone("Asia/Kolkata")
            today = pd.Timestamp(datetime.now(ist).date(), tz=ist)
            df = df[df.index.tz_convert(ist).normalize() < today]
        else:
            # Session complete — drop the 15:15-15:30 stub candle if present
            # Keep only candles that started at or before 14:45
            last_ts = df.index[-1]
            candle_close = last_ts + timedelta(minutes=30)
            if last_ts.tzinfo is not None:
                from datetime import timezone
                now = datetime.now(timezone.utc)
            else:
                now = datetime.now()
            if candle_close > now:
                df = df.iloc[:-1]
        return df

    # ---- Indicator Calculations ----

    def _calculate_rsi(self, closes: pd.Series, period: int) -> Optional[float]:
        """Calculate RSI using Wilder's smoothing (EWM with alpha=1/period)."""
        if len(closes) < period + 1:
            return None

        delta = closes.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        last_avg_loss = avg_loss.iloc[-1]
        if last_avg_loss == 0:
            return 100.0

        rs = avg_gain.iloc[-1] / last_avg_loss
        rsi = 100 - (100 / (1 + rs))
        return round(float(rsi), 2)

    def _calculate_ema(self, closes: pd.Series, period: int) -> Optional[float]:
        """Calculate EMA for the given period."""
        if len(closes) < period:
            return None
        ema = closes.ewm(span=period, adjust=False).mean()
        return round(float(ema.iloc[-1]), 2)

    # ---- State Tracking ----

    def _load_state(self) -> Dict:
        """Load previous scan state (which tickers had all 4 conditions TRUE)."""
        if not os.path.exists(self.state_file):
            return {"date": None, "passing_tickers": []}
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"date": None, "passing_tickers": []}

    def _save_state(self, passing_tickers: List[str]):
        """Save current scan state."""
        state = {
            "date": datetime.now().isoformat(),
            "passing_tickers": passing_tickers
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

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
        except Exception:
            return None

    def clear_cache(self):
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)

    # ---- Main Screener ----

    def run_screener(self, force_refresh: bool = False, progress_callback=None) -> Dict:
        """
        Run the momentum scanner.

        Returns dict with 4 categories:
            - momentum_entries: All 4 TRUE + new since last scan
            - active_momentum: All 4 TRUE + was TRUE last scan
            - full_scan: All 50 stocks with indicator values
            - exit_signals: Was all-TRUE, now failing
        """
        if not force_refresh:
            cache = self._load_cache()
            if cache:
                return cache["data"]

        config = load_config()
        rsi2_threshold = config.get("momentum_rsi2_threshold", 75)
        rsi14_threshold = 30  # Fixed: 30-min RSI(14) > 30

        tickers = NIFTY_50_TICKERS

        # Load previous state
        prev_state = self._load_state()
        prev_passing = set(prev_state.get("passing_tickers", []))

        all_stocks = []
        failed_tickers = []
        total = len(tickers)

        for i, ticker in enumerate(tickers):
            nse_symbol = f"{ticker}.NS"

            if progress_callback:
                progress_callback(i + 1, total, ticker)

            try:
                # Fetch daily data
                daily = self._fetch_daily_data(nse_symbol)
                if len(daily) < 200:
                    failed_tickers.append(ticker)
                    continue

                daily_close = daily["Close"]
                current_price = round(float(daily_close.iloc[-1]), 2)

                # Daily RSI(2)
                daily_rsi2 = self._calculate_rsi(daily_close, 2)

                # Daily EMA(50) and EMA(200)
                ema50 = self._calculate_ema(daily_close, 50)
                ema200 = self._calculate_ema(daily_close, 200)

                # Fetch 30-min data for intraday RSI(14)
                intraday = self._fetch_intraday_data(nse_symbol)
                intraday_rsi14 = None
                if len(intraday) >= 15:
                    intraday_rsi14 = self._calculate_rsi(intraday["Close"], 14)

                # Evaluate 4 conditions (cast to native bool for JSON)
                cond1 = bool(daily_rsi2 is not None and daily_rsi2 >= rsi2_threshold)
                cond2 = bool(intraday_rsi14 is not None and intraday_rsi14 > rsi14_threshold)
                cond3 = bool(ema50 is not None and current_price > ema50)
                cond4 = bool(ema200 is not None and current_price > ema200)

                all_pass = cond1 and cond2 and cond3 and cond4

                stock_info = {
                    "ticker": ticker,
                    "close": current_price,
                    "daily_rsi2": daily_rsi2,
                    "intraday_rsi14": intraday_rsi14,
                    "ema50": ema50,
                    "ema200": ema200,
                    "cond1_pass": cond1,
                    "cond2_pass": cond2,
                    "cond3_pass": cond3,
                    "cond4_pass": cond4,
                    "all_pass": all_pass,
                }

                all_stocks.append(stock_info)

            except Exception:
                failed_tickers.append(ticker)

        # Classify stocks into panels
        currently_passing = set(s["ticker"] for s in all_stocks if s["all_pass"])

        momentum_entries = []
        active_momentum = []
        exit_signals = []

        for stock in all_stocks:
            t = stock["ticker"]
            if stock["all_pass"]:
                if t in prev_passing:
                    active_momentum.append(stock)
                else:
                    momentum_entries.append(stock)
            elif t in prev_passing:
                exit_signals.append(stock)

        # Sort panels
        momentum_entries.sort(key=lambda x: x.get("daily_rsi2") or 0, reverse=True)
        active_momentum.sort(key=lambda x: x.get("daily_rsi2") or 0, reverse=True)
        exit_signals.sort(key=lambda x: x.get("daily_rsi2") or 0)  # ascending

        # Full scan: sort by conditions passed desc, then RSI(2) desc
        def sort_full_scan(s):
            conds = sum([s["cond1_pass"], s["cond2_pass"], s["cond3_pass"], s["cond4_pass"]])
            rsi2 = s.get("daily_rsi2") or 0
            return (-conds, -rsi2)

        full_scan = sorted(all_stocks, key=sort_full_scan)

        # Save state for next scan
        self._save_state(list(currently_passing))

        result = {
            "momentum_entries": momentum_entries,
            "active_momentum": active_momentum,
            "full_scan": full_scan,
            "exit_signals": exit_signals,
            "last_updated": datetime.now().isoformat(),
            "settings": {
                "rsi2_threshold": rsi2_threshold,
                "rsi14_threshold": rsi14_threshold,
                "universe": len(NIFTY_50_TICKERS),
            },
            "stats": {
                "total_processed": total - len(failed_tickers),
                "failed": len(failed_tickers),
            },
        }

        self._save_cache(result)
        return result
