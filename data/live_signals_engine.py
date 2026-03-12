"""
Live Signals Engine — scans for Strategy J, T, R, and MW entry signals in real-time,
manages positions (with 3-stage exits for T and R), and checks exit conditions.
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
from sector_mapping import STOCK_SECTOR_MAP

# Sectoral Indices (Yahoo Finance symbols) for RS momentum
SECTORAL_INDICES = {
    "NIFTY PVT BANK": "NIFTY_PVT_BANK.NS",
    "NIFTY PSU BANK": "^CNXPSUBANK",
    "NIFTY IT": "^CNXIT",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY METAL": "^CNXMETAL",
    "NIFTY REALTY": "^CNXREALTY",
    "NIFTY ENERGY": "^CNXENERGY",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY MEDIA": "^CNXMEDIA",
    "NIFTY INFRA": "^CNXINFRA",
    "NIFTY FIN SERVICE": "NIFTY_FIN_SERVICE.NS",
    "NIFTY COMMODITIES": "^CNXCMDT",
    "NIFTY CONSUMPTION": "^CNXCONSUM",
    "NIFTY HEALTHCARE": "^CNXPHARMA",
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

# Nifty 200 constituents (next 100 beyond Nifty 100)
NIFTY_200_NEXT100_TICKERS = [
    "360ONE", "ABCAPITAL", "ALKEM", "APLAPOLLO", "ASHOKLEY",
    "ASTRAL", "AUBANK", "BAJAJHFL", "BANKINDIA", "BDL",
    "BHARATFORG", "BHARTIHEXA", "BIOCON", "BLUESTARCO", "BSE",
    "CGPOWER", "COCHINSHIP", "COFORGE", "CONCOR", "COROMANDEL",
    "CUMMINSIND", "DIVISLAB", "DIXON", "DMART", "ENRIN",
    "EXIDEIND", "FEDERALBNK", "FORTIS", "GLENMARK", "GMRAIRPORT",
    "GODFRYPHLP", "GODREJPROP", "HDFCAMC", "HINDZINC", "HUDCO",
    "HYUNDAI", "IDEA", "IDFCFIRSTB", "IGL", "INDHOTEL",
    "INDIANB", "IRCTC", "IREDA", "ITCHOTELS", "JINDALSTEL",
    "JUBLFOOD", "KALYANKJIL", "KEI", "KPITTECH", "LICHSGFIN",
    "LTF", "LTM", "M&MFIN", "MAXHEALTH", "MAZDOCK",
    "MFSL", "MOTILALOFS", "MPHASIS", "MRF", "MUTHOOTFIN",
    "NATIONALUM", "NMDC", "NTPCGREEN", "NYKAA", "OBEROIRLTY",
    "OFSS", "OIL", "PAGEIND", "PATANJALI", "PAYTM",
    "PERSISTENT", "PHOENIXLTD", "PIIND", "POLICYBZR", "POWERINDIA",
    "PREMIERENE", "PRESTIGE", "HINDPETRO", "IRB", "RVNL",
    "SAIL", "SBICARD", "SHRIRAMFIN", "SOLARINDS", "SONACOMS",
    "SUPREMEIND", "SUZLON", "SWIGGY", "TATACOMM", "TATAELXSI",
    "TATATECH", "TIINDIA", "TMPV", "TORNTPOWER", "UNIONBANK",
    "UPL", "VBL", "VMM", "VOLTAS", "WAAREEENER",
    "YESBANK", "ZYDUSLIFE",
]


def _calculate_rsi_series(closes: pd.Series, period: int) -> pd.Series:
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _find_swing_lows(lows: pd.Series, left: int = 5, right: int = 3) -> list:
    """Find swing low positions: low[i] is min of surrounding window."""
    result = []
    vals = lows.values
    n = len(vals)
    for i in range(left, n - right):
        window = vals[i - left: i + right + 1]
        if vals[i] == np.min(window):
            result.append((i, float(vals[i])))
    return result


def _rsi_near_swing(rsi14_vals, idx, window=3):
    """Get min RSI within ±window bars of a swing low index.

    Price low and RSI low often don't land on the same bar —
    use the lowest RSI in the zone to match how traders read charts.
    """
    start = max(0, idx - window)
    end = min(len(rsi14_vals), idx + window + 1)
    chunk = rsi14_vals[start:end]
    valid = [v for v in chunk if not np.isnan(v)]
    return min(valid) if valid else rsi14_vals[idx]


def _detect_bullish_divergence(lows_vals, rsi14_vals, i, swing_lows,
                               max_lookback=50, min_sep=5,
                               rsi_threshold=40, min_rsi_divergence=3,
                               min_price_drop=0.0, max_curr_age=None):
    """Check for bullish RSI divergence at bar i.

    Uses min RSI within ±3 bars of each swing low to match visual chart reading.
    If max_curr_age set, current swing low must be within that many bars of i.
    """
    recent = [(idx, val) for idx, val in swing_lows
              if idx <= i and i - idx <= max_lookback]
    if len(recent) < 2:
        return False, None, None
    for k in range(len(recent) - 1, 0, -1):
        curr_idx, curr_low = recent[k]
        prev_idx, prev_low = recent[k - 1]
        if max_curr_age is not None and i - curr_idx > max_curr_age:
            continue
        if curr_idx - prev_idx < min_sep:
            continue
        if curr_low >= prev_low * (1 - min_price_drop):
            continue
        curr_rsi = _rsi_near_swing(rsi14_vals, curr_idx)
        prev_rsi = _rsi_near_swing(rsi14_vals, prev_idx)
        if np.isnan(curr_rsi) or np.isnan(prev_rsi):
            continue
        if curr_rsi - prev_rsi < min_rsi_divergence:
            continue
        if curr_rsi >= rsi_threshold:
            continue
        return True, curr_low, curr_rsi
    return False, None, None


def _detect_hidden_bullish_divergence(lows_vals, rsi14_vals, i, swing_lows,
                                       max_lookback=50, min_sep=5,
                                       rsi_threshold=60, min_rsi_divergence=5,
                                       max_curr_age=None):
    """Check for hidden bullish RSI divergence at bar i.

    Hidden bullish divergence (uptrend continuation):
    - Price: current swing low > previous swing low (higher low)
    - RSI(14): current < previous - min_rsi_divergence (lower low in RSI)
    - RSI(14) < rsi_threshold at current swing low
    If max_curr_age set, current swing low must be within that many bars of i.

    Returns (True, swing_low_val, rsi_at_low) or (False, None, None).
    """
    recent = [(idx, val) for idx, val in swing_lows
              if idx <= i and i - idx <= max_lookback]
    if len(recent) < 2:
        return False, None, None
    for k in range(len(recent) - 1, 0, -1):
        curr_idx, curr_low = recent[k]
        prev_idx, prev_low = recent[k - 1]
        if max_curr_age is not None and i - curr_idx > max_curr_age:
            continue
        if curr_idx - prev_idx < min_sep:
            continue
        if curr_low <= prev_low:
            continue
        curr_rsi = _rsi_near_swing(rsi14_vals, curr_idx)
        prev_rsi = _rsi_near_swing(rsi14_vals, prev_idx)
        if np.isnan(curr_rsi) or np.isnan(prev_rsi):
            continue
        if prev_rsi - curr_rsi < min_rsi_divergence:
            continue
        if curr_rsi >= rsi_threshold:
            continue
        return True, curr_low, curr_rsi
    return False, None, None


def _calculate_adx_series(highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14):
    """Calculate ADX, +DI, and -DI series using Wilder's smoothing."""
    prev_high = highs.shift(1)
    prev_low = lows.shift(1)
    prev_close = closes.shift(1)

    tr1 = highs - lows
    tr2 = (highs - prev_close).abs()
    tr3 = (lows - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = highs - prev_high
    down_move = prev_low - lows
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=highs.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=highs.index)

    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    smooth_plus_dm = plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    smooth_minus_dm = minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    plus_di = 100 * smooth_plus_dm / atr
    minus_di = 100 * smooth_minus_dm / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(window=period, min_periods=period).mean()

    return adx, plus_di, minus_di


class LiveSignalsEngine:
    """Scans J/T/R/MW entry signals, manages positions with 3-stage exits, checks exit conditions."""

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

    # ==================== Sector RS Momentum ====================

    def _compute_sector_momentum(self, end_date):
        """Compute sector RS vs Nifty 50 with multi-day momentum.

        Returns dict: sector_name -> {rs_today, rs_5d_ago, rs_10d_ago, rs_20d_ago,
                                       delta_5d, delta_10d, delta_20d, momentum_score}
        momentum_score: weighted sum of deltas — captures sectors gaining RS fast.
        """
        try:
            nifty = yf.Ticker("^NSEI").history(
                start=end_date - timedelta(days=120), end=end_date)
            if nifty.empty or len(nifty) < 30:
                return {}
            nifty_closes = nifty["Close"]
        except Exception:
            return {}

        sector_momentum = {}
        rs_period = 20  # 20-day return for RS calculation

        for sector_name, symbol in SECTORAL_INDICES.items():
            try:
                sec_data = yf.Ticker(symbol).history(
                    start=end_date - timedelta(days=120), end=end_date)
                if sec_data.empty or len(sec_data) < 30:
                    continue
                sec_closes = sec_data["Close"]

                # Align dates
                common = nifty_closes.index.intersection(sec_closes.index)
                if len(common) < rs_period + 20:
                    continue
                n_aligned = nifty_closes.reindex(common)
                s_aligned = sec_closes.reindex(common)

                # Compute rolling RS (sector return - nifty return) at multiple points
                def rs_at(idx):
                    if idx < rs_period:
                        return None
                    n_ret = (float(n_aligned.iloc[idx]) / float(n_aligned.iloc[idx - rs_period]) - 1) * 100
                    s_ret = (float(s_aligned.iloc[idx]) / float(s_aligned.iloc[idx - rs_period]) - 1) * 100
                    return round(s_ret - n_ret, 2)

                last = len(common) - 1
                rs_today = rs_at(last)
                rs_5d = rs_at(max(rs_period, last - 5))
                rs_10d = rs_at(max(rs_period, last - 10))
                rs_20d = rs_at(max(rs_period, last - 20))

                if rs_today is None:
                    continue

                delta_5d = round(rs_today - rs_5d, 2) if rs_5d is not None else 0
                delta_10d = round(rs_today - rs_10d, 2) if rs_10d is not None else 0
                delta_20d = round(rs_today - rs_20d, 2) if rs_20d is not None else 0

                # Momentum score: weighted — recent change matters more
                # Positive = sector RS is rising (from -10 to -3 = +7 = good)
                momentum_score = round(delta_5d * 3 + delta_10d * 2 + delta_20d * 1, 2)

                sector_momentum[sector_name] = {
                    "rs": rs_today,
                    "rs_5d_ago": rs_5d,
                    "rs_10d_ago": rs_10d,
                    "rs_20d_ago": rs_20d,
                    "delta_5d": delta_5d,
                    "delta_10d": delta_10d,
                    "delta_20d": delta_20d,
                    "momentum": momentum_score,
                }
            except Exception:
                continue

        return sector_momentum

    # ==================== Entry Signal Scanning ====================

    def scan_entry_signals(self, force_refresh=False, progress_callback=None, scan_date=None):
        """Scan for J, T, and R entry signals across Nifty 50/100.

        scan_date: optional "YYYY-MM-DD" string to scan a historical date.
        """
        is_historical = scan_date is not None

        if not force_refresh and not is_historical:
            cached = self._load_cache()
            if cached is not None:
                return cached

        config = load_config()
        universe = config.get("live_signals_universe", 50)

        if universe <= 50:
            tickers = NIFTY_50_TICKERS
        elif universe <= 100:
            tickers = NIFTY_50_TICKERS + NIFTY_NEXT50_TICKERS
        elif universe <= 200:
            tickers = NIFTY_50_TICKERS + NIFTY_NEXT50_TICKERS + NIFTY_200_NEXT100_TICKERS
        else:
            # Midcap 150: Nifty 500 tickers 101-250 (by market cap ranking)
            nifty100_set = set(NIFTY_50_TICKERS + NIFTY_NEXT50_TICKERS)
            midcap = [t for t in NIFTY_500_TICKERS if t not in nifty100_set][:150]
            tickers = midcap

        total = len(tickers)
        if is_historical:
            end_date = datetime.strptime(scan_date, "%Y-%m-%d") + timedelta(days=1)
        else:
            end_date = datetime.now() + timedelta(days=1)
        daily_start = end_date - timedelta(days=500)

        # Fetch Nifty index data once
        try:
            nifty_raw = yf.Ticker("^NSEI").history(start=daily_start, end=end_date)
        except Exception:
            nifty_raw = pd.DataFrame()

        # Fetch Nifty 200 benchmark for RS calculation
        try:
            bench_raw = yf.Ticker("^CNX200").history(start=daily_start, end=end_date)
        except Exception:
            bench_raw = nifty_raw  # fallback to Nifty 50

        # Compute benchmark 123-day return for RS
        bench_ret123_val = 0.0
        bench_ret123_28ago = 0.0
        if not bench_raw.empty and len(bench_raw) > 160:
            bc = bench_raw["Close"]
            bi = len(bc) - 1
            if bi >= 123 and float(bc.iloc[bi - 123]) > 0:
                bench_ret123_val = (float(bc.iloc[bi]) / float(bc.iloc[bi - 123]) - 1) * 100
            if bi >= 123 + 28 and float(bc.iloc[bi - 28 - 123]) > 0:
                bench_ret123_28ago = (float(bc.iloc[bi - 28]) / float(bc.iloc[bi - 28 - 123]) - 1) * 100

        # Nifty regime: close > 20-week EMA (simple mode)
        nifty_regime_on = False
        if not nifty_raw.empty:
            nifty_weekly = nifty_raw["Close"].resample("W-FRI").last().dropna()
            if len(nifty_weekly) >= 20:
                nifty_20w_ema = nifty_weekly.ewm(span=20, adjust=False).mean()
                nifty_20w_daily = nifty_20w_ema.reindex(nifty_raw.index, method="ffill")
                last_nifty_close = float(nifty_raw["Close"].iloc[-1])
                last_nifty_20w = float(nifty_20w_daily.iloc[-1])
                nifty_regime_on = last_nifty_close >= last_nifty_20w

        j_signals = []
        t_signals = []
        r_signals = []
        rw_signals = []
        mw_signals = []
        rs_signals = []
        actual_date = None  # Track actual last trading date from data

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
            bar_date = daily.index[i]
            if hasattr(bar_date, 'tz') and bar_date.tz is not None:
                bar_date = bar_date.tz_localize(None)
            if actual_date is None or bar_date > actual_date:
                actual_date = bar_date
            price = float(closes.iloc[i])
            open_price = float(opens.iloc[i])
            low = float(lows.iloc[i])
            high = float(highs.iloc[i])
            hl_range = high - low
            ibs = (price - low) / hl_range if hl_range > 0 else 0.5
            is_green = price > open_price

            # Gap-down detection
            no_gap_down = True
            if i > 0:
                prev_close = float(closes.iloc[i - 1])
                if open_price < prev_close:
                    no_gap_down = False

            # Strategy RS: Relative Strength Rotation watchlist (scans ALL stocks, even gap-down)
            try:
                if is_green and no_gap_down and ibs > 0.5 and nifty_regime_on and i >= 160:
                    stock_ret123 = (price / float(closes.iloc[i - 123]) - 1) * 100
                    rs_val = stock_ret123 - bench_ret123_val
                    if rs_val > 5.0:
                        # RS rising: compare vs 28 trading days ago
                        stock_ret123_28ago = (float(closes.iloc[i - 28]) / float(closes.iloc[i - 28 - 123]) - 1) * 100
                        rs_28ago = stock_ret123_28ago - bench_ret123_28ago
                        if rs_val > rs_28ago:
                            # Price > 30-week EMA
                            weekly_rs = daily.resample("W-FRI").agg({"Close": "last"}).dropna()
                            if len(weekly_rs) >= 30:
                                ema30w = weekly_rs["Close"].ewm(span=30, adjust=False).mean()
                                ema30w_daily = ema30w.reindex(daily.index, method="ffill")
                                ema30w_val = float(ema30w_daily.iloc[i])
                                if not pd.isna(ema30w_val) and price > ema30w_val:
                                    # dist_high filter: price >= 3% below 20d high
                                    skip_rs = False
                                    high_20d = float(highs.iloc[max(0, i - 20):i].max()) if i >= 20 else price
                                    dist_high_pct = round((high_20d - price) / high_20d * 100, 1) if high_20d > 0 else 0.0
                                    if i >= 20 and high_20d > 0 and (high_20d - price) / high_20d < 0.03:
                                        skip_rs = True
                                    if not skip_rs:
                                        # ATR% for display
                                        prev_cl_rs = closes.shift(1)
                                        tr_rs = pd.concat([highs - lows, (highs - prev_cl_rs).abs(), (lows - prev_cl_rs).abs()], axis=1).max(axis=1)
                                        atr14_rs = float(tr_rs.ewm(alpha=1/14, min_periods=14, adjust=False).mean().iloc[i])
                                        atr_pct_rs = round(atr14_rs / price * 100, 2) if price > 0 else 99.0
                                        rs_signals.append({
                                            "ticker": ticker,
                                            "price": round(price, 2),
                                            "rs_pct": round(rs_val, 1),
                                            "dist_high_pct": dist_high_pct,
                                            "atr_pct": atr_pct_rs,
                                            "stop_pct": 8.0,
                                            "sector": STOCK_SECTOR_MAP.get(ticker, "OTHER"),
                                        })
            except Exception:
                pass

            # Skip gap-down stocks for J/T/R/MW
            if not no_gap_down:
                continue

            # Strategy J: Weekly Close Support Bounce
            try:
                weekly = daily.resample("W-FRI").agg({
                    "Open": "first", "High": "max", "Low": "min",
                    "Close": "last", "Volume": "sum"
                }).dropna()
                if len(weekly) >= 27:
                    # Skip last 2 weeks (use proven support, not recent noise)
                    w_support_series = weekly["Close"].rolling(
                        window=26, min_periods=26).min().shift(2)
                    w_support_daily = w_support_series.reindex(
                        daily.index, method="ffill")
                    ws = float(w_support_daily.iloc[i]) if not pd.isna(
                        w_support_daily.iloc[i]) else None

                    # 26-week weekly low for stop-loss, also skip last 2 weeks
                    w_low_stop_series = weekly["Low"].rolling(
                        window=26, min_periods=26).min().shift(2)
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
                            j_stop_pct = round((price - raw_stop) / price * 100, 2) if price > 0 else 99.0
                            # ATR14 for volatility ranking
                            prev_close_j = closes.shift(1)
                            tr1_j = highs - lows
                            tr2_j = (highs - prev_close_j).abs()
                            tr3_j = (lows - prev_close_j).abs()
                            tr_j = pd.concat([tr1_j, tr2_j, tr3_j], axis=1).max(axis=1)
                            atr14_j = float(tr_j.ewm(alpha=1/14, min_periods=14, adjust=False).mean().iloc[i])
                            atr_norm_j = round(atr14_j / price * 100, 2) if price > 0 else 99.0
                            j_signals.append({
                                "ticker": ticker,
                                "price": round(price, 2),
                                "support": round(ws, 2),
                                "stop": round(raw_stop, 2),
                                "stop_pct": j_stop_pct,
                                "close_near_pct": round(close_near_pct, 2),
                                "ibs": round(ibs, 2),
                                "low": round(low, 2),
                                "atr_pct": atr_norm_j,
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
                    # Skip T if stock already has a J signal (match backtest dedup)
                    already_j = any(s["ticker"] == ticker for s in j_signals)
                    if near_ema20 and was_at_upper and is_green and not already_j and ibs > 0.5:
                        atr_norm_t = round(atr14 / price * 100, 2) if price > 0 else 99.0
                        t_signals.append({
                            "ticker": ticker,
                            "price": round(price, 2),
                            "ema20": round(ema20_val, 2),
                            "upper_keltner": round(upper_keltner, 2),
                            "stop_pct": 5.0,
                            "atr_pct": atr_norm_t,
                        })
            except Exception:
                pass

            # Strategy R: Bullish RSI Divergence (Regular + Hidden)
            try:
                # Skip R if stock already has J or T signal (dedup)
                already_jt = any(s["ticker"] == ticker for s in j_signals + t_signals)
                if is_green and ibs > 0.5 and not already_jt:
                    rsi14_series = _calculate_rsi_series(closes, 14)
                    swing_lows = _find_swing_lows(lows)
                    rsi14_vals = rsi14_series.values
                    lows_vals = lows.values
                    divergence, swing_low_val, rsi_at_low = _detect_bullish_divergence(
                        lows_vals, rsi14_vals, i, swing_lows,
                        rsi_threshold=35)
                    r_div_type = "regular"
                    if not divergence:
                        # Try hidden bullish divergence if price > EMA50 (uptrend)
                        ema50_val = float(closes.ewm(span=50, adjust=False).mean().iloc[i])
                        if price > ema50_val:
                            divergence, swing_low_val, rsi_at_low = _detect_hidden_bullish_divergence(
                                lows_vals, rsi14_vals, i, swing_lows)
                            if divergence:
                                r_div_type = "hidden"
                    if divergence and swing_low_val is not None:
                        rsi14_at_bar = float(rsi14_series.iloc[i]) if not pd.isna(rsi14_series.iloc[i]) else 0.0
                        r_struct_stop = round(swing_low_val * 0.99, 2)
                        r_stop_pct = round((price - r_struct_stop) / price * 100, 2) if price > 0 else 99.0
                        r_min_stop = 2.0 if r_div_type == "hidden" else 0.0
                        if r_min_stop < r_stop_pct <= 6.0:
                            # ATR14 for volatility ranking
                            prev_close_r = closes.shift(1)
                            tr1_r = highs - lows
                            tr2_r = (highs - prev_close_r).abs()
                            tr3_r = (lows - prev_close_r).abs()
                            tr_r = pd.concat([tr1_r, tr2_r, tr3_r], axis=1).max(axis=1)
                            atr14_r = float(tr_r.ewm(alpha=1/14, min_periods=14, adjust=False).mean().iloc[i])
                            atr_norm_r = round(atr14_r / price * 100, 2) if price > 0 else 99.0
                            r_signals.append({
                                "ticker": ticker,
                                "price": round(price, 2),
                                "rsi14": round(rsi14_at_bar, 1),
                                "rsi_at_low": round(float(rsi_at_low), 1),
                                "swing_low": round(swing_low_val, 2),
                                "stop": r_struct_stop,
                                "stop_pct": r_stop_pct,
                                "atr_pct": atr_norm_r,
                                "div_type": r_div_type,
                            })
            except Exception:
                pass

            # Strategy RW: DISABLED — weekly RSI divergence (code preserved in momentum_backtest.py)

            # Strategy MW: Weekly ADX >= 20 and rising with DI+ > DI-
            try:
                already_any = any(s["ticker"] == ticker for s in j_signals + t_signals + r_signals)
                if is_green and ibs > 0.5 and not already_any:
                    weekly_mw = daily.resample("W-FRI").agg({
                        "Open": "first", "High": "max", "Low": "min",
                        "Close": "last", "Volume": "sum"
                    }).dropna()
                    if len(weekly_mw) >= 28:
                        mw_adx, mw_pdi, mw_mdi = _calculate_adx_series(
                            weekly_mw["High"], weekly_mw["Low"], weekly_mw["Close"])
                        w_dates = weekly_mw.index
                        day_ts = pd.Timestamp(actual_date or datetime.now().date())
                        if w_dates.tz:
                            day_ts = day_ts.tz_localize(w_dates.tz)
                        w_before = w_dates[w_dates < day_ts]
                        if len(w_before) >= 2:
                            w_idx = len(w_before) - 1
                            curr_adx = float(mw_adx.iloc[w_idx])
                            prev_adx = float(mw_adx.iloc[w_idx - 1])
                            plus_di = float(mw_pdi.iloc[w_idx])
                            minus_di = float(mw_mdi.iloc[w_idx])
                            if (not np.isnan(curr_adx) and not np.isnan(prev_adx)
                                    and curr_adx >= 25 and curr_adx > prev_adx
                                    and plus_di > minus_di):
                                prev_close_mw = closes.shift(1)
                                tr1_mw = highs - lows
                                tr2_mw = (highs - prev_close_mw).abs()
                                tr3_mw = (lows - prev_close_mw).abs()
                                tr_mw = pd.concat([tr1_mw, tr2_mw, tr3_mw], axis=1).max(axis=1)
                                atr14_mw = float(tr_mw.ewm(alpha=1/14, min_periods=14, adjust=False).mean().iloc[i])
                                atr_norm_mw = round(atr14_mw / price * 100, 2) if price > 0 else 99.0
                                mw_signals.append({
                                    "ticker": ticker,
                                    "price": round(price, 2),
                                    "adx": round(curr_adx, 1),
                                    "plus_di": round(plus_di, 1),
                                    "minus_di": round(minus_di, 1),
                                    "stop_pct": 5.0,
                                    "atr_pct": atr_norm_mw,
                                })
            except Exception:
                pass

        # Strategy WT: Weekly Trend Breakout — scans full Nifty 500
        wt_signals = []
        wt_tickers = NIFTY_500_TICKERS  # Full Nifty 500
        # Exclude tickers already scanned that have other signals
        already_signaled = set(s["ticker"] for s in j_signals + t_signals + r_signals + mw_signals)
        wt_total = len(wt_tickers)
        for wt_idx, ticker in enumerate(wt_tickers):
            if progress_callback:
                progress_callback(total + wt_idx + 1, total + wt_total, f"WT: {ticker}")
            try:
                daily_wt = yf.Ticker(f"{ticker}.NS").history(
                    start=daily_start, end=end_date)
            except Exception:
                continue
            if daily_wt.empty or len(daily_wt) < 210:
                continue

            wt_closes = daily_wt["Close"]
            wt_opens = daily_wt["Open"]
            wt_highs = daily_wt["High"]
            wt_lows = daily_wt["Low"]
            wt_volumes = daily_wt["Volume"].astype(float)
            wi = len(daily_wt) - 1
            wt_price = float(wt_closes.iloc[wi])
            wt_open = float(wt_opens.iloc[wi])
            wt_is_green = wt_price > wt_open
            wt_low = float(wt_lows.iloc[wi])
            wt_high = float(wt_highs.iloc[wi])
            wt_hl = wt_high - wt_low
            wt_ibs = (wt_price - wt_low) / wt_hl if wt_hl > 0 else 0.5

            # Gap-down filter
            if wi > 0:
                wt_prev_close = float(wt_closes.iloc[wi - 1])
                if wt_open < wt_prev_close:
                    continue

            if not wt_is_green or wt_ibs <= 0.5:
                continue

            try:
                weekly_wt = daily_wt.resample("W-FRI").agg({
                    "Open": "first", "High": "max", "Low": "min",
                    "Close": "last", "Volume": "sum"
                }).dropna()
                if len(weekly_wt) < 25:
                    continue

                w_closes_wt = weekly_wt["Close"]
                w_ema20 = w_closes_wt.ewm(span=20, adjust=False).mean()
                w_ema50 = w_closes_wt.ewm(span=50, adjust=False).mean()
                w_high20 = w_closes_wt.rolling(20).max()
                w_slope = (w_ema50 / w_ema50.shift(4) - 1) * 100 / 4
                w_gap = w_ema20 - w_ema50
                w_gap_pct = (w_ema20 - w_ema50) / w_ema50 * 100
                w_vol = weekly_wt["Volume"]
                w_vol_avg20 = w_vol.rolling(20).mean()
                w_return = (w_closes_wt / w_closes_wt.shift(1) - 1) * 100

                # Use previous completed week (shift by 1)
                w_dates = weekly_wt.index
                day_ts = pd.Timestamp(actual_date or datetime.now().date())
                if w_dates.tz:
                    day_ts = day_ts.tz_localize(w_dates.tz)
                w_before = w_dates[w_dates < day_ts]
                if len(w_before) < 2:
                    continue
                w_idx = len(w_before) - 1

                def _wt_v(series, idx):
                    v = series.iloc[idx] if idx < len(series) else np.nan
                    return float(v) if not pd.isna(v) else 0.0

                # All shifted by 1 for confirmed candle
                cl = _wt_v(w_closes_wt.shift(1), w_idx)
                hn = _wt_v(w_high20.shift(1), w_idx)
                e20 = _wt_v(w_ema20.shift(1), w_idx)
                e50 = _wt_v(w_ema50.shift(1), w_idx)
                sl = _wt_v(w_slope.shift(1), w_idx)
                gp = _wt_v(w_gap.shift(1), w_idx)
                gp2 = _wt_v(w_gap.shift(2), w_idx)
                gpct = _wt_v(w_gap_pct.shift(1), w_idx)
                vol = _wt_v(w_vol.shift(1), w_idx)
                vavg = _wt_v(w_vol_avg20.shift(1), w_idx)
                ret = _wt_v(w_return.shift(1), w_idx)

                if (e20 > 0 and e50 > 0 and hn > 0
                        and cl >= hn          # 20-week breakout
                        and e20 > e50         # EMA trend
                        and gp > gp2 > 0      # gap widening
                        and sl >= 0.4          # slope steep enough
                        and gpct >= 2.0        # gap >= 2%
                        and vavg > 0 and vol >= vavg * 1.3  # volume
                        and ret <= 15.0):      # no parabolic spike
                    # ATR14
                    pc_wt = wt_closes.shift(1)
                    tr1_wt = wt_highs - wt_lows
                    tr2_wt = (wt_highs - pc_wt).abs()
                    tr3_wt = (wt_lows - pc_wt).abs()
                    tr_wt = pd.concat([tr1_wt, tr2_wt, tr3_wt], axis=1).max(axis=1)
                    atr14_wt = float(tr_wt.ewm(alpha=1/14, min_periods=14, adjust=False).mean().iloc[wi])
                    atr_pct_wt = round(atr14_wt / wt_price * 100, 2) if wt_price > 0 else 99.0
                    wt_signals.append({
                        "ticker": ticker,
                        "price": round(wt_price, 2),
                        "slope": round(sl, 2),
                        "gap_pct": round(gpct, 1),
                        "vol_ratio": round(vol / vavg, 1) if vavg > 0 else 0,
                        "week_return": round(ret, 1),
                        "stop_pct": 12.0,
                        "atr_pct": atr_pct_wt,
                    })
            except Exception:
                pass

        # Compute sector momentum
        sector_momentum = self._compute_sector_momentum(end_date)

        # Attach sector info to each signal
        for sig_list in [j_signals, t_signals, r_signals, mw_signals, wt_signals, rs_signals]:
            for s in sig_list:
                sector = STOCK_SECTOR_MAP.get(s["ticker"], "OTHER")
                s["sector"] = sector
                sec_data = sector_momentum.get(sector, {})
                s["sector_rs"] = sec_data.get("rs", 0)
                s["sector_momentum"] = sec_data.get("momentum", 0)
                s["sector_delta_5d"] = sec_data.get("delta_5d", 0)

        # Sort by volatility (lowest ATR% first = calmest stocks)
        # R gets priority in combined ranking (best WR + P&L historically)
        j_signals.sort(key=lambda s: s.get("atr_pct", 99.0))
        t_signals.sort(key=lambda s: s.get("atr_pct", 99.0))
        r_signals.sort(key=lambda s: s.get("atr_pct", 99.0))
        mw_signals.sort(key=lambda s: s.get("atr_pct", 99.0))
        wt_signals.sort(key=lambda s: s.get("atr_pct", 99.0))
        rs_signals.sort(key=lambda s: -s.get("rs_pct", 0))
        result = {
            "j_signals": j_signals,
            "t_signals": t_signals,
            "r_signals": r_signals,
            "rw_signals": [],  # RW disabled
            "mw_signals": mw_signals,
            "wt_signals": wt_signals,
            "rs_signals": rs_signals,
            "nifty_regime": "ON" if nifty_regime_on else "OFF",
            "sector_momentum": sector_momentum,
            "last_updated": datetime.now().isoformat(),
            "universe": universe,
        }

        if actual_date is not None:
            result["actual_date"] = actual_date.strftime("%Y-%m-%d")
        if is_historical:
            result["scan_date"] = scan_date
        else:
            self._save_cache(result)
            self._append_signals_history(j_signals, t_signals, r_signals, rw_signals, mw_signals, wt_signals)

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

    def _append_signals_history(self, j_signals, t_signals=None, r_signals=None, rw_signals=None, mw_signals=None, wt_signals=None):
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

            for s in (r_signals or []):
                writer.writerow([
                    scan_date, scan_time, "R", s["ticker"], s["price"],
                    s.get("swing_low", ""), "", "",
                ])

            for s in (rw_signals or []):
                writer.writerow([
                    scan_date, scan_time, "RW", s["ticker"], s["price"],
                    s.get("swing_low", ""), "", "",
                ])

            for s in (mw_signals or []):
                writer.writerow([
                    scan_date, scan_time, "MW", s["ticker"], s["price"],
                    "", "", "",
                ])

            for s in (wt_signals or []):
                writer.writerow([
                    scan_date, scan_time, "WT", s["ticker"], s["price"],
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

        exit_date_str = datetime.now().strftime("%Y-%m-%d")
        try:
            holding_days = (datetime.strptime(exit_date_str, "%Y-%m-%d")
                            - datetime.strptime(pos["entry_date"], "%Y-%m-%d")).days
        except Exception:
            holding_days = 0

        closed_record = {
            "id": pos["id"],
            "ticker": pos["ticker"],
            "strategy": pos["strategy"],
            "entry_date": pos["entry_date"],
            "entry_price": pos["entry_price"],
            "exit_date": exit_date_str,
            "exit_price": round(exit_price, 2),
            "shares_exited": shares_to_exit,
            "reason": reason,
            "pnl_pct": pnl_pct,
            "pnl_amount": pnl_amount,
            "holding_days": holding_days,
        }

        remaining = current_shares - shares_to_exit
        if remaining <= 0:
            # Fully closed
            active.pop(pos_idx)
        else:
            # Partial close — advance stage so exit signals move to next level
            pos["remaining_shares"] = remaining
            current_stage = pos.get("partial_stage", 0)
            pos["partial_stage"] = current_stage + 1
            # Mark partial_exit_done after stage 1 (2nd partial) for strategies with 3-stage exits
            if pos["partial_stage"] >= 2:
                pos["partial_exit_done"] = True

        data["closed"].append(closed_record)
        self._save_positions_data(data)
        return closed_record

    # ==================== Exit Signal Checking ====================

    def check_exit_signals(self, ltps=None):
        """Check exit conditions for all active positions.
        ltps: optional dict {ticker: live_price} from Zerodha. Falls back to yfinance close."""
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

            current_price = ltps.get(ticker) if ltps and ticker in ltps else float(daily["Close"].iloc[-1])
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
                        if entry_dt.tzinfo is None and daily.index.tz is not None:
                            entry_dt = entry_dt.tz_localize(daily.index.tz)
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
                # Stage 0: 5% SL → exit all. +6% → sell 1/3, set stage=1.
                # Stage 1: 3% SL → exit remaining. +10% → sell 1/3, set stage=2 (partial_exit_done=True).
                # Stage 2: 3% SL → exit remaining. Upper Keltner → exit remaining.
                stage = pos.get("partial_stage", 0)
                third = pos["shares"] // 3

                # Tighter SL after first partial exit
                sl_pct = 0.03 if stage >= 1 else 0.05
                if current_price <= entry_price * (1 - sl_pct):
                    sz = pos["remaining_shares"]
                    sl_label = f"HARD_SL_{int(sl_pct*100)}PCT"
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, sl_label, sz))
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

                if stage == 0 and current_price >= entry_price * 1.06 and third > 0:
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "PARTIAL_6PCT_1of3", third))
                elif stage == 1 and current_price >= entry_price * 1.10 and third > 0:
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "PARTIAL_10PCT_2of3", third))
                elif (stage == 2 or pos["partial_exit_done"]) and current_price >= upper_keltner:
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "KELTNER_UPPER_EXIT",
                        pos["remaining_shares"]))

            elif pos["strategy"] == "R":
                # R: 3-stage exit logic (mirrors T with structural SL)
                stage = pos.get("partial_stage", 0)
                third = pos["shares"] // 3
                structural_stop = pos.get("metadata", {}).get("r_swing_low_stop", 0)

                # Nifty crash shield for R
                r_nifty_entry = pos.get("metadata", {}).get("nifty_at_entry", 0)
                r_nifty_shields = False
                if r_nifty_entry > 0 and nifty_close_today > 0:
                    nifty_pct = (nifty_close_today - r_nifty_entry) / r_nifty_entry
                    stock_pct = (current_price - entry_price) / entry_price
                    if nifty_pct <= stock_pct and nifty_pct < 0:
                        r_nifty_shields = True

                # Structural SL: 1% below divergence swing low — exit all
                if not r_nifty_shields and structural_stop > 0 and current_price <= structural_stop:
                    sz = pos["remaining_shares"]
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "STRUCTURAL_SL", sz))
                    continue

                # Tight SL after first partial (3%) — also shielded
                if not r_nifty_shields and stage >= 1 and current_price <= entry_price * 0.97:
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "HARD_SL_3PCT",
                        pos["remaining_shares"]))
                    continue

                # Compute upper Keltner for final exit
                closes_r = daily["Close"]
                highs_r = daily["High"]
                lows_r = daily["Low"]
                ema20_r = float(closes_r.ewm(span=20, adjust=False).mean().iloc[-1])
                prev_close_r = closes_r.shift(1)
                tr1 = highs_r - lows_r
                tr2 = (highs_r - prev_close_r).abs()
                tr3 = (lows_r - prev_close_r).abs()
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr14_r = float(true_range.ewm(alpha=1/14, min_periods=14, adjust=False).mean().iloc[-1])
                upper_keltner = ema20_r + 2 * atr14_r

                if stage == 0 and current_price >= entry_price * 1.06 and third > 0:
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "PARTIAL_6PCT_1of3", third))
                elif stage == 1 and current_price >= entry_price * 1.10 and third > 0:
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "PARTIAL_10PCT_2of3", third))
                elif (stage == 2 or pos["partial_exit_done"]) and current_price >= upper_keltner:
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "KELTNER_UPPER_EXIT",
                        pos["remaining_shares"]))

            elif pos["strategy"] == "MW":
                # MW exits: 6% SL initial, 3% after P1, breakeven after P2, partials, weekly Keltner
                stage = pos.get("partial_stage", 0)
                third = pos["shares"] // 3

                # Nifty crash shield for MW
                mw_nifty_entry = pos.get("metadata", {}).get("nifty_at_entry", 0)
                mw_nifty_shields = False
                if mw_nifty_entry > 0 and nifty_close_today > 0:
                    nifty_pct = (nifty_close_today - mw_nifty_entry) / mw_nifty_entry
                    stock_pct = (current_price - entry_price) / entry_price
                    if nifty_pct <= stock_pct and nifty_pct < 0:
                        mw_nifty_shields = True

                # Hard SL: 6% initial, 3% after P1, breakeven after P2
                if stage >= 2:
                    mw_sl_price = entry_price
                    sl_label = "BREAKEVEN_SL"
                elif stage >= 1:
                    mw_sl_price = entry_price * 0.97
                    sl_label = "HARD_SL_3PCT"
                else:
                    mw_sl_price = entry_price * 0.94
                    sl_label = "HARD_SL_6PCT"

                if not mw_nifty_shields and current_price <= mw_sl_price:
                    sz = pos["remaining_shares"]
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, sl_label, sz))
                    continue

                # 2-stage partial exits (+6% sell 1/3, +10% sell 1/3)
                if stage == 0 and current_price >= entry_price * 1.06 and third > 0:
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "PARTIAL_6PCT_1of3", third))
                elif stage == 1 and current_price >= entry_price * 1.10 and third > 0:
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "PARTIAL_10PCT_2of3", third))

                # Weekly upper Keltner exit on remaining (after first partial)
                if stage >= 1:
                    try:
                        weekly = yf.Ticker(f"{ticker}.NS").history(
                            start=end_date - timedelta(days=180), end=end_date, interval="1wk")
                        if not weekly.empty and len(weekly) >= 14:
                            w_closes = weekly["Close"]
                            w_ema20 = float(w_closes.ewm(span=20, adjust=False).mean().iloc[-1])
                            w_highs = weekly["High"]
                            w_lows = weekly["Low"]
                            w_prev = w_closes.shift(1)
                            w_tr = pd.concat([w_highs - w_lows, (w_highs - w_prev).abs(), (w_lows - w_prev).abs()], axis=1).max(axis=1)
                            w_atr14 = float(w_tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean().iloc[-1])
                            mw_upper_keltner = w_ema20 + 2 * w_atr14
                            if current_price >= mw_upper_keltner:
                                exit_signals.append(self._make_exit_signal(
                                    pos, current_price, pnl_pct, "KELTNER_UPPER_EXIT",
                                    pos["remaining_shares"]))
                    except Exception:
                        pass

        # Underwater exit: if held >= 10 trading days and still below entry, cut it
        for pos in positions:
            # Skip if already flagged for exit above
            already_flagged = any(e["position_id"] == pos["id"] for e in exit_signals)
            if already_flagged:
                continue

            entry_date_str = pos.get("entry_date", "")
            if not entry_date_str:
                continue

            try:
                daily = yf.Ticker(f"{pos['ticker']}.NS").history(
                    start=end_date - timedelta(days=30), end=end_date)
            except Exception:
                continue

            if daily.empty or len(daily) < 2:
                continue

            current_price = ltps.get(pos["ticker"]) if ltps and pos["ticker"] in ltps else float(daily["Close"].iloc[-1])
            entry_price = pos["entry_price"]

            if current_price >= entry_price:
                continue  # Not underwater

            entry_dt = pd.Timestamp(entry_date_str)
            if entry_dt.tzinfo is None and daily.index.tz is not None:
                entry_dt = entry_dt.tz_localize(daily.index.tz)
            trading_days_held = len(daily[daily.index >= entry_dt])
            uw_days = 25 if pos["strategy"] == "MW" else 10
            if trading_days_held >= uw_days:
                pnl_pct = round(((current_price - entry_price) / entry_price) * 100, 2)
                sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                exit_signals.append(self._make_exit_signal(
                    pos, current_price, pnl_pct, "UNDERWATER_EXIT", sz))

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
