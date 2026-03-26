"""
GOLDM Live Signal Engine — MCX Gold Mini ORB Paper Trading
==========================================================
Detects ORB breakout signals, manages paper positions, tracks P&L.

Frozen Config:
  OR: 30-min (09:00-09:30)
  Entry: 15-min candle close above/below OR high/low → next bar open
  Target: 1.5× OR range | SL: 1.0× OR range
  Range filter: skip if OR > 0.3% of price
  Session: 09:30-23:00, squareoff 23:20
  Max 1 trade/day, both sides
  Charges: ₹220/lot round-trip
"""

import json
import os
import uuid
from datetime import datetime, timedelta, time as dtime, date
from pathlib import Path

import pandas as pd
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
GOLDM_CFG = {
    "lot_size": 100,
    "charges_per_lot": 220,
    "tick_size": 1,
    "slippage_ticks": 2,
    "or_minutes": 30,
    "or_target_mult": 1.5,
    "or_sl_mult": 1.0,
    "or_max_range_pct": 0.3,
    "session_start": dtime(9, 0),
    "trade_start": dtime(9, 30),
    "trade_end": dtime(23, 0),
    "squareoff_time": dtime(23, 20),
}

DATA_STORE = Path(__file__).parent.parent / "data_store"
DATA_STORE.mkdir(exist_ok=True)


class GoldmEngine:
    """Live GOLDM ORB signal engine with paper position management."""

    def __init__(self, user_id="user1"):
        self.user_id = user_id
        self.cfg = dict(GOLDM_CFG)
        self.positions_file = DATA_STORE / f"goldm_positions_{user_id}.json"
        self._ensure_file()

    def _ensure_file(self):
        if not self.positions_file.exists():
            self.positions_file.write_text(json.dumps({
                "active": None,
                "closed": [],
                "today": {
                    "date": None,
                    "or_high": None,
                    "or_low": None,
                    "or_range": None,
                    "or_range_pct": None,
                    "signal": None,
                    "traded": False,
                }
            }, indent=2))

    def _load(self):
        with open(self.positions_file) as f:
            return json.load(f)

    def _save(self, data):
        with open(self.positions_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

    # ── Kite Data Fetching ────────────────────────────────────────────────────

    def fetch_today_candles(self, kite):
        """Fetch today's 5-min GOLDM candles from Kite, resample to 15-min."""
        # Find GOLDM nearest contract
        instruments = kite.instruments("MCX")
        goldm_futs = [i for i in instruments
                      if "GOLDM" in i["tradingsymbol"] and i["instrument_type"] == "FUT"]
        if not goldm_futs:
            return None, "No GOLDM futures found"

        goldm_futs.sort(key=lambda i: i["expiry"])
        nearest = goldm_futs[0]
        token = nearest["instrument_token"]
        symbol = nearest["tradingsymbol"]

        # Fetch today's 5-min data
        today = datetime.now().date()
        try:
            candles = kite.historical_data(
                token, today, today + timedelta(days=1),
                interval="5minute"
            )
        except Exception as e:
            return None, f"Kite API error: {e}"

        if not candles:
            return None, "No candles today (market closed?)"

        df = pd.DataFrame(candles)
        df['date'] = pd.to_datetime(df['date'])
        if df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_convert('Asia/Kolkata')
        else:
            df['date'] = df['date'].dt.tz_localize('Asia/Kolkata')
        df = df.set_index('date').sort_index()

        # Resample to 15-min
        df_15 = df.resample('15min').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()

        return df_15, symbol

    # ── Opening Range ─────────────────────────────────────────────────────────

    def compute_or(self, df_15):
        """Compute today's 30-min opening range from 15-min data."""
        if df_15 is None or df_15.empty:
            return None

        # OR = first 30 min (2 × 15-min bars starting from 09:00)
        or_start = df_15.index[0]
        or_end = or_start + pd.Timedelta(minutes=30)
        or_bars = df_15[df_15.index <= or_end]

        if len(or_bars) < 2:
            return None

        or_high = float(or_bars['high'].max())
        or_low = float(or_bars['low'].min())
        or_range = or_high - or_low
        or_close = float(or_bars['close'].iloc[-1])
        or_range_pct = (or_range / or_close * 100) if or_close > 0 else 99

        return {
            "or_high": or_high,
            "or_low": or_low,
            "or_range": or_range,
            "or_range_pct": round(or_range_pct, 3),
            "or_close": or_close,
            "skip": or_range_pct > self.cfg["or_max_range_pct"] or or_range <= 0,
        }

    # ── Signal Detection ──────────────────────────────────────────────────────

    def check_entry_signal(self, df_15, or_data):
        """Check if a breakout signal has fired. Returns signal dict or None."""
        if or_data is None or or_data.get("skip"):
            return None

        or_high = or_data["or_high"]
        or_low = or_data["or_low"]
        or_range = or_data["or_range"]

        # Get bars after OR period
        or_end = df_15.index[0] + pd.Timedelta(minutes=30)
        trade_bars = df_15[
            (df_15.index > or_end) &
            (df_15.index.time <= self.cfg["trade_end"])
        ]

        if trade_bars.empty:
            return None

        # Check each bar for breakout (close above OR high / below OR low)
        for ts, row in trade_bars.iterrows():
            if row['close'] > or_high:
                slippage = self.cfg["slippage_ticks"] * self.cfg["tick_size"]
                # Entry would be next bar's open — for live, we use current price + slippage
                entry = float(row['close']) + slippage
                target = entry + or_range * self.cfg["or_target_mult"]
                sl = entry - or_range * self.cfg["or_sl_mult"]
                return {
                    "side": "LONG",
                    "entry": round(entry, 1),
                    "target": round(target, 1),
                    "sl": round(sl, 1),
                    "or_high": or_high,
                    "or_low": or_low,
                    "or_range": or_range,
                    "signal_time": str(ts),
                    "signal_bar_close": float(row['close']),
                }

            elif row['close'] < or_low:
                slippage = self.cfg["slippage_ticks"] * self.cfg["tick_size"]
                entry = float(row['close']) - slippage
                target = entry - or_range * self.cfg["or_target_mult"]
                sl = entry + or_range * self.cfg["or_sl_mult"]
                return {
                    "side": "SHORT",
                    "entry": round(entry, 1),
                    "target": round(target, 1),
                    "sl": round(sl, 1),
                    "or_high": or_high,
                    "or_low": or_low,
                    "or_range": or_range,
                    "signal_time": str(ts),
                    "signal_bar_close": float(row['close']),
                }

        return None  # no breakout yet

    # ── Exit Signal Check ─────────────────────────────────────────────────────

    def check_exit_signal(self, current_price, current_time=None):
        """Check if active position should exit. Returns reason or None."""
        data = self._load()
        pos = data.get("active")
        if not pos:
            return None

        price = float(current_price)
        now = current_time or datetime.now().time()

        # Squareoff time
        if now >= self.cfg["squareoff_time"]:
            return "SQUAREOFF"

        if pos["side"] == "LONG":
            # Target (intra-bar — GTT)
            if price >= pos["target"]:
                return "TARGET"
            # SL
            if price <= pos["sl"]:
                return "SL"
        elif pos["side"] == "SHORT":
            if price <= pos["target"]:
                return "TARGET"
            if price >= pos["sl"]:
                return "SL"

        return None

    # ── Position Management ───────────────────────────────────────────────────

    def add_position(self, signal, lots=1):
        """Enter a paper position from a signal."""
        data = self._load()
        if data.get("active"):
            return None, "Already in a position"

        pos = {
            "id": uuid.uuid4().hex[:8],
            "ticker": "GOLDM",
            "strategy": "GOLDM_ORB",
            "side": signal["side"],
            "entry_date": str(date.today()),
            "entry_time": datetime.now().strftime("%H:%M"),
            "entry_price": signal["entry"],
            "lots": lots,
            "lot_size": self.cfg["lot_size"],
            "or_high": signal["or_high"],
            "or_low": signal["or_low"],
            "or_range": signal["or_range"],
            "target": signal["target"],
            "sl": signal["sl"],
        }

        data["active"] = pos
        data["today"]["traded"] = True
        self._save(data)
        return pos, None

    def close_position(self, exit_price, reason="MANUAL"):
        """Close active position."""
        data = self._load()
        pos = data.get("active")
        if not pos:
            return None, "No active position"

        price = float(exit_price)
        lots = pos["lots"]
        lot_size = pos["lot_size"]
        charges = self.cfg["charges_per_lot"] * lots

        if pos["side"] == "LONG":
            pnl = (price - pos["entry_price"]) * lot_size * lots - charges
        else:
            pnl = (pos["entry_price"] - price) * lot_size * lots - charges

        closed_trade = {
            **pos,
            "exit_date": str(date.today()),
            "exit_time": datetime.now().strftime("%H:%M"),
            "exit_price": price,
            "pnl": round(pnl, 2),
            "pnl_per_gram": round((price - pos["entry_price"]) if pos["side"] == "LONG"
                                  else (pos["entry_price"] - price), 1),
            "charges": charges,
            "reason": reason,
        }

        data["active"] = None
        data["closed"].append(closed_trade)
        self._save(data)
        return closed_trade, None

    def get_status(self):
        """Get full status: today's OR, signal, position, closed trades."""
        data = self._load()
        return {
            "active": data.get("active"),
            "closed": data.get("closed", []),
            "today": data.get("today", {}),
            "config": {
                "or_minutes": self.cfg["or_minutes"],
                "target_mult": self.cfg["or_target_mult"],
                "sl_mult": self.cfg["or_sl_mult"],
                "range_filter": self.cfg["or_max_range_pct"],
                "lot_size": self.cfg["lot_size"],
                "charges": self.cfg["charges_per_lot"],
            }
        }

    def scan(self, kite):
        """Full scan: fetch candles, compute OR, check signals."""
        df_15, symbol = self.fetch_today_candles(kite)
        if df_15 is None:
            return {"error": symbol, "or": None, "signal": None}

        or_data = self.compute_or(df_15)

        data = self._load()
        today_str = str(date.today())

        # Reset today if new day
        if data.get("today", {}).get("date") != today_str:
            data["today"] = {
                "date": today_str,
                "or_high": None, "or_low": None,
                "or_range": None, "or_range_pct": None,
                "signal": None, "traded": False,
            }
            data["active"] = None  # auto-close stale positions from yesterday

        if or_data:
            data["today"]["or_high"] = or_data["or_high"]
            data["today"]["or_low"] = or_data["or_low"]
            data["today"]["or_range"] = or_data["or_range"]
            data["today"]["or_range_pct"] = or_data["or_range_pct"]

        # Check signal (only if not already traded today)
        signal = None
        if or_data and not or_data["skip"] and not data["today"]["traded"]:
            signal = self.check_entry_signal(df_15, or_data)
            if signal:
                data["today"]["signal"] = signal["side"]

        # Check exit for active position
        exit_reason = None
        if data.get("active") and not df_15.empty:
            current_price = float(df_15['close'].iloc[-1])
            current_time = df_15.index[-1].time()
            exit_reason = self.check_exit_signal(current_price, current_time)

        self._save(data)

        # Build candle data for chart
        candles = []
        for ts, row in df_15.iterrows():
            candles.append({
                "time": str(ts),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": int(row['volume']),
            })

        return {
            "symbol": symbol,
            "or": or_data,
            "signal": signal,
            "exit_signal": exit_reason,
            "active": data.get("active"),
            "today": data["today"],
            "candles": candles,
            "current_price": float(df_15['close'].iloc[-1]) if not df_15.empty else None,
            "current_time": str(df_15.index[-1]) if not df_15.empty else None,
        }
