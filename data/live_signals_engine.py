"""
Live Signals Engine — scans for Strategy J and K entry signals in real-time,
manages positions, and checks exit conditions.
"""

import csv
import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import yfinance as yf

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import (
    LIVE_SIGNALS_CACHE_FILE, LIVE_POSITIONS_FILE, LIVE_SIGNALS_HISTORY_FILE,
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


def _calculate_rsi_series(closes: pd.Series, period: int) -> pd.Series:
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


class LiveSignalsEngine:
    """Scans J/K/N/O/R/S/T/U/V entry signals, manages positions, checks exit conditions."""

    def __init__(self, user_id=None):
        self.user_id = user_id
        self.cache_file = LIVE_SIGNALS_CACHE_FILE
        if user_id:
            data_dir = os.path.dirname(LIVE_POSITIONS_FILE)
            self.positions_file = os.path.join(data_dir, f"live_positions_{user_id}.json")
        else:
            self.positions_file = LIVE_POSITIONS_FILE
        self._ensure_dirs()

    def _ensure_dirs(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.positions_file), exist_ok=True)

    # ==================== Entry Signal Scanning ====================

    def scan_entry_signals(self, force_refresh=False, progress_callback=None):
        """Scan for J, K, N, O, R, S, T, U entry signals across Nifty 50/100."""
        if not force_refresh:
            cached = self._load_cache()
            if cached is not None:
                return cached

        config = load_config()
        universe = config.get("live_signals_universe", 50)

        if universe <= 50:
            tickers = NIFTY_50_TICKERS
        elif universe <= 100:
            tickers = NIFTY_50_TICKERS + NIFTY_NEXT50_TICKERS
        else:
            # Midcap 150: Nifty 500 tickers 101-250 (by market cap ranking)
            nifty100_set = set(NIFTY_50_TICKERS + NIFTY_NEXT50_TICKERS)
            midcap = [t for t in NIFTY_500_TICKERS if t not in nifty100_set][:150]
            tickers = midcap

        total = len(tickers)
        end_date = datetime.now()
        daily_start = end_date - timedelta(days=500)

        # Fetch Nifty index data once
        try:
            nifty_raw = yf.Ticker("^NSEI").history(start=daily_start, end=end_date)
        except Exception:
            nifty_raw = pd.DataFrame()

        # Nifty CCI(20) for Strategy N regime filter
        nifty_cci_val = 0.0
        if not nifty_raw.empty:
            n_tp = (nifty_raw["High"] + nifty_raw["Low"] + nifty_raw["Close"]) / 3
            n_sma = n_tp.rolling(window=20, min_periods=20).mean()
            n_md = n_tp.rolling(window=20, min_periods=20).apply(
                lambda x: np.mean(np.abs(x - x.mean())), raw=True)
            n_cci = (n_tp - n_sma) / (0.015 * n_md)
            nifty_cci_val = float(n_cci.iloc[-1]) if not pd.isna(n_cci.iloc[-1]) else 0.0

        j_signals = []
        k_signals = []
        n_signals = []
        o_signals = []
        r_signals = []
        s_signals = []
        t_signals = []
        u_signals = []
        v_signals = []

        for idx, ticker in enumerate(tickers):
            if progress_callback:
                progress_callback(idx + 1, total, ticker)

            try:
                daily = yf.Ticker(f"{ticker}.NS").history(
                    start=daily_start, end=end_date)
            except Exception:
                continue

            if daily.empty or len(daily) < 210:
                continue

            closes = daily["Close"]
            opens = daily["Open"]
            highs = daily["High"]
            lows = daily["Low"]
            volumes = daily["Volume"].astype(float)

            i = len(daily) - 1  # Latest bar
            price = float(closes.iloc[i])
            open_price = float(opens.iloc[i])
            low = float(lows.iloc[i])
            high = float(highs.iloc[i])
            hl_range = high - low
            ibs = (price - low) / hl_range if hl_range > 0 else 0.5
            is_green = price > open_price

            # Strategy J: Weekly Close Support Bounce
            try:
                weekly = daily.resample("W-FRI").agg({
                    "Open": "first", "High": "max", "Low": "min",
                    "Close": "last", "Volume": "sum"
                }).dropna()
                if len(weekly) >= 27:
                    w_support_series = weekly["Close"].rolling(
                        window=26, min_periods=26).min()
                    w_support_daily = w_support_series.reindex(
                        daily.index, method="ffill")
                    ws = float(w_support_daily.iloc[i]) if not pd.isna(
                        w_support_daily.iloc[i]) else None

                    # 26-week weekly low for stop-loss
                    w_low_stop_series = weekly["Low"].rolling(
                        window=26, min_periods=26).min()
                    w_low_stop_daily = w_low_stop_series.reindex(
                        daily.index, method="ffill")
                    wls = float(w_low_stop_daily.iloc[i]) if not pd.isna(
                        w_low_stop_daily.iloc[i]) else None

                    if ws is not None and ws > 0:
                        close_near_pct = ((price - ws) / ws) * 100
                        # CCI(20) confirmation
                        tp = (highs + lows + closes) / 3
                        sma_tp = tp.rolling(window=20, min_periods=20).mean()
                        mean_dev = tp.rolling(window=20, min_periods=20).apply(
                            lambda x: np.mean(np.abs(x - x.mean())), raw=True)
                        cci_series = (tp - sma_tp) / (0.015 * mean_dev)
                        cci_val = float(cci_series.iloc[i]) if not pd.isna(cci_series.iloc[i]) else 0.0
                        if (close_near_pct >= 0 and close_near_pct <= 3.0
                                and ibs > 0.5 and is_green
                                and cci_val > -100):
                            raw_stop = wls if wls else ws
                            j_signals.append({
                                "ticker": ticker,
                                "price": round(price, 2),
                                "support": round(ws, 2),
                                "stop": round(raw_stop, 2),
                                "close_near_pct": round(close_near_pct, 2),
                                "ibs": round(ibs, 2),
                                "low": round(low, 2),
                            })
            except Exception:
                pass

            # Strategy K: RS Trending Dip
            try:
                rsi2_series = _calculate_rsi_series(closes, 2)
                rsi2 = float(rsi2_series.iloc[i])
                rsi2_prev = float(rsi2_series.iloc[i - 1]) if i > 0 else 50.0
                ema50 = float(closes.ewm(span=50, adjust=False).mean().iloc[i])
                vol_today = float(volumes.iloc[i])
                vol_avg20 = float(volumes.rolling(
                    window=20, min_periods=20).mean().iloc[i]) if len(volumes) >= 20 else 0.0

                rs_spread = 0.0
                if not nifty_raw.empty:
                    nifty_close = nifty_raw["Close"].reindex(
                        daily.index, method="ffill")
                    stock_3m = closes.pct_change(periods=63)
                    nifty_3m = nifty_close.pct_change(periods=63)
                    rs_spread_series = stock_3m - nifty_3m
                    if not pd.isna(rs_spread_series.iloc[i]):
                        rs_spread = float(rs_spread_series.iloc[i])

                vol_ratio = round(vol_today / vol_avg20, 2) if vol_avg20 > 0 else 0.0

                if (rs_spread > 0 and rsi2 >= 20
                        and not np.isnan(rsi2_prev) and rsi2_prev < 20
                        and price > ema50
                        and ibs > 0.5
                        and vol_today > vol_avg20):
                    # Don't duplicate if already a J signal
                    already_j = any(s["ticker"] == ticker for s in j_signals)
                    if not already_j:
                        k_signals.append({
                            "ticker": ticker,
                            "price": round(price, 2),
                            "rs_pct": round(rs_spread * 100, 1),
                            "rsi2": round(rsi2, 1),
                            "vol_ratio": vol_ratio,
                            "ema50": round(ema50, 2),
                            "ibs": round(ibs, 2),
                        })
            except Exception:
                pass

            # Strategy N: CCI Momentum (crossover)
            try:
                tp = (highs + lows + closes) / 3
                sma_tp = tp.rolling(window=20, min_periods=20).mean()
                mean_dev = tp.rolling(window=20, min_periods=20).apply(
                    lambda x: np.mean(np.abs(x - x.mean())), raw=True)
                cci_series = (tp - sma_tp) / (0.015 * mean_dev)
                cci_val = float(cci_series.iloc[i]) if not pd.isna(cci_series.iloc[i]) else 0.0
                cci_prev = float(cci_series.iloc[i-1]) if i > 0 and not pd.isna(cci_series.iloc[i-1]) else 0.0

                if cci_val > 100 and cci_prev <= 100 and is_green:
                    # RS spread (3M) for ranking
                    rs_spread = 0.0
                    if not nifty_raw.empty:
                        nifty_close = nifty_raw["Close"].reindex(daily.index, method="ffill")
                        stock_3m = closes.pct_change(periods=63)
                        nifty_3m = nifty_close.pct_change(periods=63)
                        rs_s = stock_3m - nifty_3m
                        if not pd.isna(rs_s.iloc[i]):
                            rs_spread = float(rs_s.iloc[i])
                    n_signals.append({
                        "ticker": ticker,
                        "price": round(price, 2),
                        "cci": round(cci_val, 1),
                        "rs_pct": round(rs_spread * 100, 1),
                    })
            except Exception:
                pass

            # Strategy O: Connors RSI(2) Mean Reversion
            try:
                rsi2_series_o = _calculate_rsi_series(closes, 2)
                rsi2_val = float(rsi2_series_o.iloc[i])
                sma200 = float(closes.rolling(window=200, min_periods=200).mean().iloc[i])
                if not np.isnan(sma200) and rsi2_val < 5 and price > sma200:
                    o_signals.append({
                        "ticker": ticker,
                        "price": round(price, 2),
                        "rsi2": round(rsi2_val, 1),
                        "sma200": round(sma200, 2),
                    })
            except Exception:
                pass

            # Strategy R: RSI Momentum
            try:
                rsi14_series_r = _calculate_rsi_series(closes, 14)
                rsi14_val = float(rsi14_series_r.iloc[i])
                rsi14_prev = float(rsi14_series_r.iloc[i - 1]) if i > 0 else 50.0
                ema200_val = float(closes.ewm(span=200, adjust=False).mean().iloc[i])
                if rsi14_prev <= 60 and rsi14_val > 60 and price > ema200_val and is_green:
                    r_signals.append({
                        "ticker": ticker,
                        "price": round(price, 2),
                        "rsi14": round(rsi14_val, 1),
                    })
            except Exception:
                pass

            # Strategy S: MACD + RSI Combo
            try:
                rsi14_series_s = _calculate_rsi_series(closes, 14)
                rsi14_val = float(rsi14_series_s.iloc[i])
                ema12 = closes.ewm(span=12, adjust=False).mean()
                ema26 = closes.ewm(span=26, adjust=False).mean()
                macd_line = ema12 - ema26
                macd_signal = macd_line.ewm(span=9, adjust=False).mean()
                macd_val = float(macd_line.iloc[i])
                macd_sig = float(macd_signal.iloc[i])
                macd_prev = float(macd_line.iloc[i - 1]) if i > 0 else 0.0
                macd_sig_prev = float(macd_signal.iloc[i - 1]) if i > 0 else 0.0
                macd_cross_up = (macd_prev <= macd_sig_prev and macd_val > macd_sig)
                if macd_cross_up and macd_val > 0 and rsi14_val > 50 and is_green:
                    s_signals.append({
                        "ticker": ticker,
                        "price": round(price, 2),
                        "macd": round(macd_val, 2),
                        "rsi14": round(rsi14_val, 1),
                    })
            except Exception:
                pass

            # Strategy T: Keltner Channel Pullback
            try:
                ema20_val = float(closes.ewm(span=20, adjust=False).mean().iloc[i])
                prev_close_s = closes.shift(1)
                tr1 = highs - lows
                tr2 = (highs - prev_close_s).abs()
                tr3 = (lows - prev_close_s).abs()
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr14 = float(true_range.ewm(alpha=1/14, min_periods=14, adjust=False).mean().iloc[i])
                if atr14 > 0:
                    upper_keltner = ema20_val + 2 * atr14
                    near_ema20 = abs(price - ema20_val) / ema20_val <= 0.01
                    was_at_upper = False
                    ema20_s = closes.ewm(span=20, adjust=False).mean()
                    atr14_s = true_range.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
                    for lb_j in range(max(0, i - 10), i):
                        past_high = float(highs.iloc[lb_j])
                        past_ema20 = float(ema20_s.iloc[lb_j])
                        past_atr14 = float(atr14_s.iloc[lb_j]) if not pd.isna(atr14_s.iloc[lb_j]) else 0.0
                        if past_high >= past_ema20 + 2 * past_atr14:
                            was_at_upper = True
                            break
                    if near_ema20 and was_at_upper and is_green:
                        t_signals.append({
                            "ticker": ticker,
                            "price": round(price, 2),
                            "ema20": round(ema20_val, 2),
                            "upper_keltner": round(upper_keltner, 2),
                        })
            except Exception:
                pass

            # Strategy U: Stochastic RSI
            try:
                rsi14_series_u = _calculate_rsi_series(closes, 14)
                rsi14_min = rsi14_series_u.rolling(window=14, min_periods=14).min()
                rsi14_max = rsi14_series_u.rolling(window=14, min_periods=14).max()
                stoch_raw = (rsi14_series_u - rsi14_min) / (rsi14_max - rsi14_min)
                stoch_raw = stoch_raw.fillna(0.5)
                stoch_k = stoch_raw.rolling(window=3, min_periods=3).mean() * 100
                stoch_d = stoch_k.rolling(window=3, min_periods=3).mean()
                stk = float(stoch_k.iloc[i]) if not pd.isna(stoch_k.iloc[i]) else 50.0
                std = float(stoch_d.iloc[i]) if not pd.isna(stoch_d.iloc[i]) else 50.0
                stk_prev = float(stoch_k.iloc[i - 1]) if i > 0 and not pd.isna(stoch_k.iloc[i - 1]) else 50.0
                std_prev = float(stoch_d.iloc[i - 1]) if i > 0 and not pd.isna(stoch_d.iloc[i - 1]) else 50.0
                ema200_val = float(closes.ewm(span=200, adjust=False).mean().iloc[i])
                k_cross_up = (stk_prev <= std_prev and stk > std)
                if k_cross_up and stk < 30 and std < 30 and price > ema200_val:
                    u_signals.append({
                        "ticker": ticker,
                        "price": round(price, 2),
                        "stoch_k": round(stk, 1),
                        "stoch_d": round(std, 1),
                    })
            except Exception:
                pass

            # Strategy V: Donchian Breakout with Volume
            try:
                high_20d = float(highs.rolling(window=20, min_periods=20).max().shift(1).iloc[i])
                ema200_val = float(closes.ewm(span=200, adjust=False).mean().iloc[i])
                vol_today = float(volumes.iloc[i])
                vol_avg20 = float(volumes.rolling(
                    window=20, min_periods=20).mean().iloc[i]) if len(volumes) >= 20 else 0.0
                vol_ratio = round(vol_today / vol_avg20, 2) if vol_avg20 > 0 else 0.0
                if (not np.isnan(high_20d) and price > high_20d
                        and vol_avg20 > 0 and vol_today > 1.5 * vol_avg20
                        and price > ema200_val and is_green):
                    v_signals.append({
                        "ticker": ticker,
                        "price": round(price, 2),
                        "high_20d": round(high_20d, 2),
                        "vol_ratio": vol_ratio,
                    })
            except Exception:
                pass

        # Sort N signals by RS spread (strongest outperformer first)
        n_signals.sort(key=lambda s: s.get("rs_pct", 0), reverse=True)

        result = {
            "j_signals": j_signals,
            "k_signals": k_signals,
            "n_signals": n_signals,
            "o_signals": o_signals,
            "r_signals": r_signals,
            "s_signals": s_signals,
            "t_signals": t_signals,
            "u_signals": u_signals,
            "v_signals": v_signals,
            "last_updated": datetime.now().isoformat(),
            "universe": universe,
        }

        self._save_cache(result)
        self._append_signals_history(j_signals, k_signals, n_signals,
                                     o_signals, r_signals, s_signals,
                                     t_signals, u_signals, v_signals)
        return result

    def get_cache_age_minutes(self):
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
            last = datetime.fromisoformat(data["last_updated"])
            return (datetime.now() - last).total_seconds() / 60
        except Exception:
            return None

    def _load_cache(self):
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
            last = datetime.fromisoformat(data["last_updated"])
            ttl = get_cache_ttl()
            if (datetime.now() - last).total_seconds() < ttl * 60:
                return data
        except Exception:
            pass
        return None

    def _save_cache(self, data):
        with open(self.cache_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _append_signals_history(self, j_signals, k_signals, n_signals=None,
                               o_signals=None, r_signals=None, s_signals=None,
                               t_signals=None, u_signals=None, v_signals=None):
        """Append today's scan results to the history CSV."""
        history_file = LIVE_SIGNALS_HISTORY_FILE
        scan_date = datetime.now().strftime("%Y-%m-%d")
        scan_time = datetime.now().strftime("%H:%M")

        write_header = not os.path.exists(history_file)

        with open(history_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "date", "time", "strategy", "ticker", "price",
                    "support", "close_near_pct", "ibs",
                    "rs_pct", "rsi2", "vol_ratio",
                ])

            for s in j_signals:
                writer.writerow([
                    scan_date, scan_time, "J", s["ticker"], s["price"],
                    s["support"], s["close_near_pct"], s["ibs"],
                    "", "", "",
                ])

            for s in k_signals:
                writer.writerow([
                    scan_date, scan_time, "K", s["ticker"], s["price"],
                    "", "", s["ibs"],
                    s["rs_pct"], s["rsi2"], s["vol_ratio"],
                ])

            for s in (n_signals or []):
                writer.writerow([
                    scan_date, scan_time, "N", s["ticker"], s["price"],
                    "", "", "",
                    s.get("rs_pct", ""), "", s.get("vol_ratio", ""),
                ])

            for s in (o_signals or []):
                writer.writerow([
                    scan_date, scan_time, "O", s["ticker"], s["price"],
                    "", "", "",
                    "", s.get("rsi2", ""), "",
                ])

            for s in (r_signals or []):
                writer.writerow([
                    scan_date, scan_time, "R", s["ticker"], s["price"],
                    "", "", "",
                    "", "", "",
                ])

            for s in (s_signals or []):
                writer.writerow([
                    scan_date, scan_time, "S", s["ticker"], s["price"],
                    "", "", "",
                    "", "", "",
                ])

            for s in (t_signals or []):
                writer.writerow([
                    scan_date, scan_time, "T", s["ticker"], s["price"],
                    "", "", "",
                    "", "", "",
                ])

            for s in (u_signals or []):
                writer.writerow([
                    scan_date, scan_time, "U", s["ticker"], s["price"],
                    "", "", "",
                    "", "", "",
                ])

            for s in (v_signals or []):
                writer.writerow([
                    scan_date, scan_time, "V", s["ticker"], s["price"],
                    "", "", "",
                    "", "", s.get("vol_ratio", ""),
                ])

    # ==================== Position Management ====================

    def _load_positions_data(self):
        try:
            with open(self.positions_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"active": [], "closed": []}

    def _save_positions_data(self, data):
        with open(self.positions_file, 'w') as f:
            json.dump(data, f, indent=2)

    def get_positions(self):
        return self._load_positions_data()["active"]

    def add_position(self, ticker, strategy, entry_price, amount,
                     support_level=0, ibs=0, metadata=None):
        """Create a new position from a Buy signal."""
        shares = int(amount // entry_price)
        if shares <= 0:
            return {"error": "Amount too small for even 1 share"}

        pos = {
            "id": str(uuid.uuid4())[:8],
            "ticker": ticker,
            "strategy": strategy,
            "entry_date": datetime.now().strftime("%Y-%m-%d"),
            "entry_price": round(entry_price, 2),
            "shares": shares,
            "amount_invested": round(shares * entry_price, 2),
            "support_level": round(support_level, 2),
            "partial_exit_done": False,
            "remaining_shares": shares,
            "metadata": metadata or {},
        }
        if self.user_id:
            pos["user_id"] = self.user_id

        data = self._load_positions_data()
        data["active"].append(pos)
        self._save_positions_data(data)
        return pos

    def delete_position(self, position_id):
        """Remove a position without placing any order (e.g. was never filled)."""
        data = self._load_positions_data()
        before = len(data["active"])
        data["active"] = [p for p in data["active"] if p["id"] != position_id]
        if len(data["active"]) == before:
            return {"error": "Position not found"}
        self._save_positions_data(data)
        return {"deleted": position_id}

    def close_position(self, position_id, exit_price, reason, shares_to_exit):
        """Close (or partially close) a position."""
        data = self._load_positions_data()
        active = data["active"]
        pos = None
        pos_idx = None

        for idx, p in enumerate(active):
            if p["id"] == position_id:
                pos = p
                pos_idx = idx
                break

        if pos is None:
            return {"error": "Position not found"}

        current_shares = pos["remaining_shares"]
        shares_to_exit = min(shares_to_exit, current_shares)

        pnl_pct = round(((exit_price - pos["entry_price"]) / pos["entry_price"]) * 100, 2)
        pnl_amount = round((exit_price - pos["entry_price"]) * shares_to_exit, 2)

        closed_record = {
            "id": pos["id"],
            "ticker": pos["ticker"],
            "strategy": pos["strategy"],
            "entry_date": pos["entry_date"],
            "entry_price": pos["entry_price"],
            "exit_date": datetime.now().strftime("%Y-%m-%d"),
            "exit_price": round(exit_price, 2),
            "shares_exited": shares_to_exit,
            "reason": reason,
            "pnl_pct": pnl_pct,
            "pnl_amount": pnl_amount,
        }

        remaining = current_shares - shares_to_exit
        if remaining <= 0:
            # Fully closed
            active.pop(pos_idx)
        else:
            # Partial close
            pos["remaining_shares"] = remaining
            pos["partial_exit_done"] = True

        data["closed"].append(closed_record)
        self._save_positions_data(data)
        return closed_record

    # ==================== Exit Signal Checking ====================

    def check_exit_signals(self):
        """Check exit conditions for all active positions."""
        positions = self.get_positions()
        if not positions:
            return []

        # Fetch Nifty data for 3-day low check
        end_date = datetime.now()
        try:
            nifty_data = yf.Ticker("^NSEI").history(
                start=end_date - timedelta(days=30), end=end_date)
            nifty_closes = nifty_data["Close"]
        except Exception:
            nifty_closes = pd.Series(dtype=float)

        nifty_weak = False
        if len(nifty_closes) >= 4:
            nifty_today = float(nifty_closes.iloc[-1])
            nifty_3day_low = float(nifty_closes.iloc[-4:-1].min())
            nifty_weak = nifty_today < nifty_3day_low

        nifty_close_today = float(nifty_closes.iloc[-1]) if len(nifty_closes) > 0 else 0.0

        exit_signals = []

        for pos in positions:
            ticker = pos["ticker"]
            try:
                daily = yf.Ticker(f"{ticker}.NS").history(
                    start=end_date - timedelta(days=30), end=end_date)
            except Exception:
                continue

            if daily.empty or len(daily) < 2:
                continue

            current_price = float(daily["Close"].iloc[-1])
            current_low = float(daily["Low"].iloc[-1])
            lows = daily["Low"]

            entry_price = pos["entry_price"]
            pnl_pct = round(((current_price - entry_price) / entry_price) * 100, 2)

            if pos["strategy"] == "J":
                support = pos["support_level"]
                t1_price = entry_price * 1.05
                t2_price = entry_price * 1.10

                # Nifty drop shield: skip support break if Nifty fell same or more
                j_nifty_entry = pos.get("metadata", {}).get("nifty_at_entry", 0)
                j_nifty_shields = False
                if j_nifty_entry > 0 and nifty_close_today > 0:
                    nifty_pct = (nifty_close_today - j_nifty_entry) / j_nifty_entry
                    stock_pct = (current_price - entry_price) / entry_price
                    if nifty_pct <= stock_pct and nifty_pct < 0:
                        j_nifty_shields = True

                if not pos["partial_exit_done"]:
                    # Support break — full exit (skip if Nifty shielded)
                    if not j_nifty_shields and current_price < support:
                        exit_signals.append(self._make_exit_signal(
                            pos, current_price, pnl_pct, "SUPPORT_BREAK",
                            pos["remaining_shares"]))
                    # +5% partial
                    elif current_price >= t1_price:
                        half = pos["shares"] // 2
                        exit_signals.append(self._make_exit_signal(
                            pos, current_price, pnl_pct, "5PCT_PARTIAL", half))
                else:
                    # Remaining shares after partial
                    if not j_nifty_shields and current_price < support:
                        exit_signals.append(self._make_exit_signal(
                            pos, current_price, pnl_pct, "SUPPORT_BREAK",
                            pos["remaining_shares"]))
                    elif current_price >= t2_price:
                        exit_signals.append(self._make_exit_signal(
                            pos, current_price, pnl_pct, "10PCT_TARGET",
                            pos["remaining_shares"]))
                    elif not nifty_weak and len(lows) >= 4:
                        three_day_low = float(lows.iloc[-4:-1].min())
                        if current_price < three_day_low:
                            exit_signals.append(self._make_exit_signal(
                                pos, current_price, pnl_pct, "BELOW_3DAY_LOW",
                                pos["remaining_shares"]))

            elif pos["strategy"] == "K":
                stop_3pct = entry_price * 0.97
                t1_price = entry_price * 1.05

                # Check Nifty drop shield
                nifty_at_entry = pos.get("metadata", {}).get("nifty_at_entry", 0)
                nifty_shields = False
                if nifty_at_entry > 0 and nifty_close_today > 0:
                    nifty_pct = (nifty_close_today - nifty_at_entry) / nifty_at_entry
                    stock_pct = (current_price - entry_price) / entry_price
                    if nifty_pct <= stock_pct and nifty_pct < 0:
                        nifty_shields = True

                if not pos["partial_exit_done"]:
                    # 3% stop (unless Nifty shields)
                    if not nifty_shields and current_price < stop_3pct:
                        exit_signals.append(self._make_exit_signal(
                            pos, current_price, pnl_pct, "STOP_3PCT",
                            pos["remaining_shares"]))
                    # +5% partial
                    elif current_price >= t1_price:
                        half = pos["shares"] // 2
                        exit_signals.append(self._make_exit_signal(
                            pos, current_price, pnl_pct, "5PCT_PARTIAL", half))
                else:
                    # 3% stop on remaining
                    if not nifty_shields and current_price < stop_3pct:
                        exit_signals.append(self._make_exit_signal(
                            pos, current_price, pnl_pct, "STOP_3PCT",
                            pos["remaining_shares"]))
                    # 3-day low exit (skip if Nifty weak)
                    elif not nifty_weak and len(lows) >= 4:
                        three_day_low = float(lows.iloc[-4:-1].min())
                        if current_price < three_day_low:
                            exit_signals.append(self._make_exit_signal(
                                pos, current_price, pnl_pct, "BELOW_3DAY_LOW",
                                pos["remaining_shares"]))

            elif pos["strategy"] == "N":
                # Hard stop-loss: 3%
                if current_price <= entry_price * 0.97:
                    sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "HARD_SL_3PCT", sz))
                    continue
                closes_s = daily["Close"]
                highs_s = daily["High"]
                lows_s = daily["Low"]
                tp = (highs_s + lows_s + closes_s) / 3
                sma_tp = tp.rolling(window=20, min_periods=20).mean()
                mean_dev = tp.rolling(window=20, min_periods=20).apply(
                    lambda x: np.mean(np.abs(x - x.mean())), raw=True)
                cci_series = (tp - sma_tp) / (0.015 * mean_dev)
                cci_now = float(cci_series.iloc[-1]) if not pd.isna(cci_series.iloc[-1]) else 0.0
                # Partial: +5% → sell 50%
                if not pos["partial_exit_done"] and current_price >= entry_price * 1.05:
                    half = pos["shares"] // 2
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "5PCT_PARTIAL", half))
                # CCI exit: CCI < -100 AND red candle
                is_red = current_price < float(daily["Open"].iloc[-1])
                if cci_now < -100 and is_red:
                    sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "CCI_RED_EXIT", sz))

            elif pos["strategy"] == "O":
                # O: 3% hard SL, Close > EMA(5) exit
                if current_price <= entry_price * 0.97:
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "HARD_SL_3PCT", pos["shares"]))
                    continue
                closes_o = daily["Close"]
                ema5_o = float(closes_o.ewm(span=5, adjust=False).mean().iloc[-1])
                if not np.isnan(ema5_o) and current_price > ema5_o:
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "EMA5_EXIT", pos["shares"]))

            elif pos["strategy"] == "R":
                # R: 5% hard SL, +5% partial, RSI(14) < 50 exit
                if current_price <= entry_price * 0.95:
                    sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "HARD_SL_3PCT", sz))
                    continue
                rsi14_s = _calculate_rsi_series(daily["Close"], 14)
                rsi14_now = float(rsi14_s.iloc[-1]) if not pd.isna(rsi14_s.iloc[-1]) else 50.0
                if not pos["partial_exit_done"] and current_price >= entry_price * 1.05:
                    half = pos["shares"] // 2
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "5PCT_PARTIAL", half))
                elif rsi14_now < 50:
                    sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "RSI14_EXIT", sz))

            elif pos["strategy"] == "S":
                # S: 5% hard SL, +5% partial, MACD cross down OR RSI(14) < 40
                if current_price <= entry_price * 0.95:
                    sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "HARD_SL_3PCT", sz))
                    continue
                closes_s = daily["Close"]
                rsi14_s = _calculate_rsi_series(closes_s, 14)
                rsi14_now = float(rsi14_s.iloc[-1]) if not pd.isna(rsi14_s.iloc[-1]) else 50.0
                ema12 = closes_s.ewm(span=12, adjust=False).mean()
                ema26 = closes_s.ewm(span=26, adjust=False).mean()
                macd_line = ema12 - ema26
                macd_signal = macd_line.ewm(span=9, adjust=False).mean()
                macd_now = float(macd_line.iloc[-1])
                macd_sig_now = float(macd_signal.iloc[-1])
                macd_prev = float(macd_line.iloc[-2]) if len(macd_line) >= 2 else 0.0
                macd_sig_prev = float(macd_signal.iloc[-2]) if len(macd_signal) >= 2 else 0.0
                macd_cross_dn = (macd_prev >= macd_sig_prev and macd_now < macd_sig_now)
                if not pos["partial_exit_done"] and current_price >= entry_price * 1.05:
                    half = pos["shares"] // 2
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "5PCT_PARTIAL", half))
                elif macd_cross_dn or rsi14_now < 40:
                    reason = "MACD_CROSS_DOWN" if macd_cross_dn else "RSI14_EXIT"
                    sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, reason, sz))

            elif pos["strategy"] == "T":
                # T: 5% hard SL, +5% partial, price >= upper Keltner exit
                if current_price <= entry_price * 0.95:
                    sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "HARD_SL_3PCT", sz))
                    continue
                closes_t = daily["Close"]
                highs_t = daily["High"]
                lows_t = daily["Low"]
                ema20_t = float(closes_t.ewm(span=20, adjust=False).mean().iloc[-1])
                prev_close_t = closes_t.shift(1)
                tr1 = highs_t - lows_t
                tr2 = (highs_t - prev_close_t).abs()
                tr3 = (lows_t - prev_close_t).abs()
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr14_t = float(true_range.ewm(alpha=1/14, min_periods=14, adjust=False).mean().iloc[-1])
                upper_keltner = ema20_t + 2 * atr14_t
                if not pos["partial_exit_done"] and current_price >= entry_price * 1.05:
                    half = pos["shares"] // 2
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "5PCT_PARTIAL", half))
                elif current_price >= upper_keltner:
                    sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "KELTNER_UPPER_EXIT", sz))

            elif pos["strategy"] == "U":
                # U: 5% hard SL, +5% partial, StochRSI %K crosses below %D both > 70
                if current_price <= entry_price * 0.95:
                    sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "HARD_SL_3PCT", sz))
                    continue
                closes_u = daily["Close"]
                rsi14_u = _calculate_rsi_series(closes_u, 14)
                rsi14_min = rsi14_u.rolling(window=14, min_periods=14).min()
                rsi14_max = rsi14_u.rolling(window=14, min_periods=14).max()
                stoch_raw = (rsi14_u - rsi14_min) / (rsi14_max - rsi14_min)
                stoch_raw = stoch_raw.fillna(0.5)
                stoch_k = stoch_raw.rolling(window=3, min_periods=3).mean() * 100
                stoch_d = stoch_k.rolling(window=3, min_periods=3).mean()
                stk = float(stoch_k.iloc[-1]) if not pd.isna(stoch_k.iloc[-1]) else 50.0
                std_val = float(stoch_d.iloc[-1]) if not pd.isna(stoch_d.iloc[-1]) else 50.0
                stk_prev = float(stoch_k.iloc[-2]) if len(stoch_k) >= 2 and not pd.isna(stoch_k.iloc[-2]) else 50.0
                std_prev = float(stoch_d.iloc[-2]) if len(stoch_d) >= 2 and not pd.isna(stoch_d.iloc[-2]) else 50.0
                k_cross_dn = (stk_prev >= std_prev and stk < std_val)
                if not pos["partial_exit_done"] and current_price >= entry_price * 1.05:
                    half = pos["shares"] // 2
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "5PCT_PARTIAL", half))
                elif k_cross_dn and stk > 70 and std_val > 70:
                    sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "STOCHRSI_EXIT", sz))

            elif pos["strategy"] == "V":
                # V: 5% hard SL, +5% partial, Close < 10-day low trailing exit
                if current_price <= entry_price * 0.95:
                    sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "HARD_SL_5PCT", sz))
                    continue
                low_10d = float(lows.rolling(window=10, min_periods=10).min().shift(1).iloc[-1])
                if not pos["partial_exit_done"] and current_price >= entry_price * 1.05:
                    half = pos["shares"] // 2
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "5PCT_PARTIAL", half))
                elif not np.isnan(low_10d) and low_10d > 0 and current_price < low_10d:
                    sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "DONCHIAN_LOW_EXIT", sz))

        return exit_signals

    def _make_exit_signal(self, pos, current_price, pnl_pct, reason, shares):
        return {
            "position_id": pos["id"],
            "ticker": pos["ticker"],
            "strategy": pos["strategy"],
            "entry_price": pos["entry_price"],
            "current_price": round(current_price, 2),
            "pnl_pct": pnl_pct,
            "reason": reason,
            "shares_to_exit": shares,
            "entry_date": pos["entry_date"],
        }
