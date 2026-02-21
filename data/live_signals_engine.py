"""
Live Signals Engine — scans for Strategy J and T entry signals in real-time,
manages positions (with 3-stage exits for T), and checks exit conditions.
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
    """Scans J/T entry signals, manages positions with 3-stage exits, checks exit conditions."""

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
        """Scan for J and T entry signals across Nifty 50/100."""
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

        j_signals = []
        t_signals = []

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

        result = {
            "j_signals": j_signals,
            "t_signals": t_signals,
            "last_updated": datetime.now().isoformat(),
            "universe": universe,
        }

        self._save_cache(result)
        self._append_signals_history(j_signals, t_signals)
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

    def _append_signals_history(self, j_signals, t_signals=None):
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
                ])

            for s in j_signals:
                writer.writerow([
                    scan_date, scan_time, "J", s["ticker"], s["price"],
                    s["support"], s["close_near_pct"], s["ibs"],
                ])

            for s in (t_signals or []):
                writer.writerow([
                    scan_date, scan_time, "T", s["ticker"], s["price"],
                    "", "", "",
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
            "partial_stage": 0,
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

        # Fetch Nifty data for Nifty shield
        end_date = datetime.now()
        try:
            nifty_data = yf.Ticker("^NSEI").history(
                start=end_date - timedelta(days=30), end=end_date)
            nifty_closes = nifty_data["Close"]
        except Exception:
            nifty_closes = pd.Series(dtype=float)

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
            highs = daily["High"]

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
                    else:
                        # Chandelier exit: highest high since entry - 3x ATR(14)
                        entry_date_str = pos.get("entry_date", "")
                        entry_dt = pd.Timestamp(entry_date_str) if entry_date_str else daily.index[0]
                        since_entry = daily[daily.index >= entry_dt]
                        if len(since_entry) >= 2:
                            highest_high = float(since_entry["High"].max())
                            # ATR(14) from daily data
                            tr = pd.concat([
                                daily["High"] - daily["Low"],
                                (daily["High"] - daily["Close"].shift()).abs(),
                                (daily["Low"] - daily["Close"].shift()).abs()
                            ], axis=1).max(axis=1)
                            atr14 = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else 0.0
                            chandelier_stop = highest_high - 3.0 * atr14
                            if atr14 > 0 and current_price < chandelier_stop:
                                exit_signals.append(self._make_exit_signal(
                                    pos, current_price, pnl_pct, "CHANDELIER_EXIT",
                                    pos["remaining_shares"]))

            elif pos["strategy"] == "T":
                # T: 3-stage exit logic
                # Stage 0: 5% SL → exit all. +5% → sell 1/3, set stage=1.
                # Stage 1: 5% SL → exit remaining. +8% → sell 1/3, set stage=2 (partial_exit_done=True).
                # Stage 2: 5% SL → exit remaining. Upper Keltner → exit remaining.
                stage = pos.get("partial_stage", 0)
                third = pos["shares"] // 3

                if current_price <= entry_price * 0.95:
                    sz = pos["remaining_shares"]
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "HARD_SL_5PCT", sz))
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

                if stage == 0 and current_price >= entry_price * 1.05 and third > 0:
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "PARTIAL_5PCT_1of3", third))
                elif stage == 1 and current_price >= entry_price * 1.08 and third > 0:
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "PARTIAL_8PCT_2of3", third))
                elif (stage == 2 or pos["partial_exit_done"]) and current_price >= upper_keltner:
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "KELTNER_UPPER_EXIT",
                        pos["remaining_shares"]))

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
