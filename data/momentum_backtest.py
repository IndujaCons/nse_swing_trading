"""
Momentum Scanner Backtest Engine
EOD backtest engine for swing trading strategies.

Strategy J — Weekly Close Support Bounce:
  Entry support: Lowest weekly CLOSE of last 26 weeks (~6 months)
  Stop support: Lowest daily LOW of 120 days
  Entry: Daily low within 1% of weekly close support AND close > support AND IBS > 0.5
         AND green candle AND CCI(20) > -100
  Exit 1: Close >= Entry+5% → sell 50%
  Exit 2: Close >= Entry+10% → sell remaining
  Stop: Close < daily low support (6-month low)

Strategy K — RS Trending Dip (scale-out):
  RS = stock 3-month return minus Nifty 3-month return
  Entry: RS > 0 (outperforming Nifty) AND RSI(2) < 20 (short-term oversold)
  Exit 1: Close >= Entry+5% → sell 50%
  Exit 2: RS drops below 0 → sell remaining (stock stops outperforming)
  Stop: Close < Entry-3% → sell all
  No max hold — stay in as long as RS > 0

Strategy N — CCI Momentum:
  Entry: CCI(20) crosses above 100 (prev day < 100, today > 100)
  Exit: CCI(20) < -100

Strategy O — Connors RSI(2) Mean Reversion:
  Entry: Price > SMA(200) AND RSI(2) < 5
  Exit: Close > EMA(5)
  Stop: 3% hard SL

Strategy R — RSI Momentum:
  Entry: RSI(14) crosses above 60 AND Price > EMA(200) AND green candle
  Exit 1: Close >= Entry+5% → sell 50%
  Exit 2: RSI(14) < 50 → sell remaining
  Stop: 5% hard SL

Strategy S — MACD + RSI Combo:
  Entry: MACD crosses signal above 0 AND RSI(14) > 50 AND green candle
  Exit 1: Close >= Entry+5% → sell 50%
  Exit 2: MACD crosses below signal OR RSI(14) < 40 → sell remaining
  Stop: 5% hard SL

Strategy T — Keltner Channel Pullback:
  Entry: Price pulls back to EMA(20) midline (was at upper Keltner in last 10 days) AND green candle
  Exit 1: Close >= Entry+5% → sell 50%
  Exit 2: Price >= upper Keltner (EMA20 + 2*ATR14) → sell remaining
  Stop: 5% hard SL

Strategy U — Stochastic RSI:
  Entry: StochRSI %K crosses above %D, both < 30 AND Price > EMA(200)
  Exit 1: Close >= Entry+5% → sell 50%
  Exit 2: StochRSI %K crosses below %D, both > 70 → sell remaining
  Stop: 5% hard SL

Strategy V — Donchian Breakout with Volume:
  Entry: Close > 20-day high (shifted) AND Volume > 1.5x avg(20) AND Price > EMA(200) AND green candle
  Exit 1: Close >= Entry+5% → sell 50%
  Exit 2: Close < 10-day low → sell remaining (trailing channel exit)
  Stop: 5% hard SL
"""

import os
import sys
import random
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import numpy as np
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import load_config


# Actual Nifty Next 50 constituents (Nifty 100 = Nifty 50 + Next 50)
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

BATCH_VARIANTS = [
    ("J", None, "J: Weekly Support"),
    ("K", None, "K: RS Trending"),
    ("N", None, "N: CCI Momentum"),
    ("O", None, "O: Connors RSI(2)"),
    ("R", None, "R: RSI Momentum"),
    ("S", None, "S: MACD+RSI"),
    ("T", None, "T: Keltner Pullback"),
    ("U", None, "U: StochRSI"),
    ("V", None, "V: Donchian Breakout"),
]


class MomentumBacktester:
    """Single-stock daily backtest for swing trading strategies (J, K, N, O, R, S, T, U, V)."""

    def _calculate_rsi_series(self, closes: pd.Series, period: int) -> pd.Series:
        """Calculate full RSI series using Wilder's smoothing."""
        delta = closes.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _calculate_cci_series(highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index (CCI) series."""
        tp = (highs + lows + closes) / 3
        sma_tp = tp.rolling(window=period, min_periods=period).mean()
        mean_dev = tp.rolling(window=period, min_periods=period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        cci = (tp - sma_tp) / (0.015 * mean_dev)
        return cci

    def run(self, symbol: str, period_days: int, strategy: str = "B",
            capital: int = 100000, exit_target: str = None,
            _daily_data=None, end_date=None) -> Dict:
        """
        Run daily-only backtest for a single symbol.

        Args:
            symbol: NSE ticker (e.g. "ICICIBANK")
            period_days: backtest lookback in calendar days (30, 90, 180, 365)
            strategy: "J","K","N","O","R","S","T","U"
            capital: starting capital in INR
            exit_target: profit target for Strategy J/K variants

        Returns:
            Dict with trades list, summary stats, and metadata.
        """
        config = load_config()
        rsi2_entry = config.get("momentum_rsi2_threshold", 75)

        end_date = end_date or datetime.now()

        if _daily_data is not None:
            daily = _daily_data
        else:
            nse_symbol = f"{symbol}.NS"
            # Fetch daily data (need 500 days warmup for EMA200 to stabilize)
            daily_start = end_date - timedelta(days=period_days + 500)
            try:
                daily = yf.Ticker(nse_symbol).history(start=daily_start, end=end_date)
            except Exception:
                daily = pd.DataFrame()

        if daily.empty or len(daily) < 210:
            return {"error": f"Insufficient daily data for {symbol}",
                    "trades": [], "summary": self._empty_summary()}

        closes = daily["Close"]
        opens = daily["Open"]
        highs = daily["High"]
        lows = daily["Low"]

        # IBS = (Close - Low) / (High - Low), clamped to [0, 1]
        hl_range = highs - lows
        ibs_series = ((closes - lows) / hl_range).where(hl_range > 0, 0.5)

        # NR7: True if today's range is the narrowest of the last 7 days
        rolling_min_range = hl_range.rolling(window=7, min_periods=7).min()
        nr7_series = (hl_range == rolling_min_range) & (hl_range > 0)

        # Support levels (exclude current bar via shift)
        support_6m = lows.rolling(window=120, min_periods=120).min().shift(1)   # 6-month
        support_1y = lows.rolling(window=252, min_periods=252).min().shift(1)   # 1-year

        # Volume for daily use
        volumes = daily["Volume"].astype(float)

        # RS Rating filter for Strategy J and K
        # For K: simple 3-month return vs Nifty (RS > 0 = outperforming)
        rs_filter_series = None  # True/False: outperforming Nifty (for J)
        rs_spread_series = None  # Stock 3M return minus Nifty 3M return (for K)
        nifty_cci_series = None  # Nifty CCI(20) for Strategy N regime filter
        if strategy in ("J", "K", "N"):
            try:
                nifty_data = yf.Ticker("^NSEI").history(
                    start=daily.index[0], end=end_date)
                if not nifty_data.empty:
                    nifty_close = nifty_data["Close"].reindex(daily.index, method="ffill")

                    if strategy == "N":
                        n_tp = (nifty_data["High"] + nifty_data["Low"] + nifty_data["Close"]) / 3
                        n_sma = n_tp.rolling(window=20, min_periods=20).mean()
                        n_md = n_tp.rolling(window=20, min_periods=20).apply(
                            lambda x: np.mean(np.abs(x - x.mean())), raw=True)
                        nifty_cci_series = ((n_tp - n_sma) / (0.015 * n_md)).reindex(daily.index, method="ffill")

                    if strategy in ("J",):
                        def _weighted_rs_score(stock_closes):
                            r3m = stock_closes.pct_change(periods=63)
                            r6m = stock_closes.pct_change(periods=126)
                            r9m = stock_closes.pct_change(periods=189)
                            r12m = stock_closes.pct_change(periods=252)
                            return 0.4 * r3m + 0.3 * r6m + 0.2 * r9m + 0.1 * r12m
                        stock_ws = _weighted_rs_score(closes)
                        nifty_ws = _weighted_rs_score(nifty_close)
                        rs_filter_series = stock_ws > nifty_ws

                    if strategy in ("K", "N"):
                        # Simple 3-month return spread: stock vs Nifty
                        stock_3m = closes.pct_change(periods=63)
                        nifty_3m = nifty_close.pct_change(periods=63)
                        rs_spread_series = stock_3m - nifty_3m  # >0 = outperforming

            except Exception:
                pass

        # Weekly support for Strategy J
        # Entry: lowest weekly CLOSE of last 26 weeks
        # Stop: lowest weekly LOW of last 26 weeks
        weekly_support_series = None
        weekly_low_stop_series = None
        if strategy == "J":
            weekly = daily.resample("W-FRI").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum"
            }).dropna()
            # 26-week rolling min of weekly CLOSE (W-FRI: ffill naturally excludes current incomplete week)
            w_support = weekly["Close"].rolling(window=26, min_periods=26).min()
            weekly_support_series = w_support.reindex(daily.index, method="ffill")
            # 26-week rolling min of weekly LOW (for stop-loss)
            w_low_stop = weekly["Low"].rolling(window=26, min_periods=26).min()
            weekly_low_stop_series = w_low_stop.reindex(daily.index, method="ffill")


        # Pre-compute indicator series over entire dataset
        rsi2_series = self._calculate_rsi_series(closes, 2)
        rsi3_series = self._calculate_rsi_series(closes, 3)
        ema5_series = closes.ewm(span=5, adjust=False).mean()
        ema8_series = closes.ewm(span=8, adjust=False).mean()
        ema10_series = closes.ewm(span=10, adjust=False).mean()
        ema20_series = closes.ewm(span=20, adjust=False).mean()
        ema50_series = closes.ewm(span=50, adjust=False).mean()
        ema200_series = closes.ewm(span=200, adjust=False).mean()
        vol_avg20_series = volumes.rolling(window=20, min_periods=20).mean()
        cci20_series = self._calculate_cci_series(highs, lows, closes, 20)

        # --- New indicators for strategies O, R, S, T, U ---
        # RSI(14) for strategies O, R, S, U
        rsi14_series = self._calculate_rsi_series(closes, 14)

        # SMA(200) and SMA(5) for Strategy O
        sma200_series = closes.rolling(window=200, min_periods=200).mean()
        sma5_series = closes.rolling(window=5, min_periods=5).mean()

        # MACD(12,26,9) for Strategy S
        ema12_series = closes.ewm(span=12, adjust=False).mean()
        ema26_series = closes.ewm(span=26, adjust=False).mean()
        macd_line_series = ema12_series - ema26_series
        macd_signal_series = macd_line_series.ewm(span=9, adjust=False).mean()

        # ATR(14) for Strategy T (Keltner Channel)
        prev_close_series = closes.shift(1)
        tr1 = highs - lows
        tr2 = (highs - prev_close_series).abs()
        tr3 = (lows - prev_close_series).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14_series = true_range.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

        # Stochastic RSI(14,14,3,3) for Strategy U
        rsi14_min = rsi14_series.rolling(window=14, min_periods=14).min()
        rsi14_max = rsi14_series.rolling(window=14, min_periods=14).max()
        stoch_rsi_raw = (rsi14_series - rsi14_min) / (rsi14_max - rsi14_min)
        stoch_rsi_raw = stoch_rsi_raw.fillna(0.5)
        stoch_k_series = stoch_rsi_raw.rolling(window=3, min_periods=3).mean() * 100
        stoch_d_series = stoch_k_series.rolling(window=3, min_periods=3).mean()

        # Donchian channels for Strategy V
        high_20d_v = highs.rolling(window=20, min_periods=20).max().shift(1)
        low_10d_v = lows.rolling(window=10, min_periods=10).min().shift(1)
        volumes = daily["Volume"].astype(float)
        vol_avg20_v = volumes.rolling(window=20, min_periods=20).mean()

        # Tracking variables for new strategies
        o_partial_done = False
        o_remaining = 0
        r_partial_done = False
        r_remaining = 0
        s_partial_done = False
        s_remaining = 0
        t_partial_done = False
        t_remaining = 0
        u_partial_done = False
        u_remaining = 0
        v_partial_done = False
        v_remaining = 0

        # Determine backtest start index (need 200+ bars warmup)
        bt_start_date = (end_date - timedelta(days=period_days)).date()
        bt_indices = [i for i, ts in enumerate(daily.index)
                      if ts.date() >= bt_start_date and i >= 200]

        if not bt_indices:
            return {"error": "No trading days in selected period",
                    "trades": [], "summary": self._empty_summary()}

        # Backtest loop
        in_position = False
        entry_price = 0.0
        entry_date = None
        shares = 0
        remaining_shares = 0  # For scale-out: shares left after partial exit
        partial_exit_done = False  # For scale-out: has first leg exited?
        n_partial_done = False  # Strategy N: partial exit tracking
        n_remaining = 0  # Strategy N: remaining shares after partial
        trades = []
        entry_support_j = 0.0    # For Strategy J: weekly open support (entry level)
        entry_stop_j = 0.0       # For Strategy J: weekly open support (stop level)
        entry_bar = 0            # For Strategy J: bar index at entry
        nifty_at_entry = 0.0     # For Strategy K: Nifty close at entry (for SL filter)

        for i in bt_indices:
            price = float(closes.iloc[i])
            rsi2 = float(rsi2_series.iloc[i])
            rsi2_prev = float(rsi2_series.iloc[i - 1]) if i > 0 else np.nan
            rsi3 = float(rsi3_series.iloc[i])
            rsi3_prev = float(rsi3_series.iloc[i - 1]) if i > 0 else np.nan
            ema5 = float(ema5_series.iloc[i])
            ema8 = float(ema8_series.iloc[i])
            ema10 = float(ema10_series.iloc[i])
            ema20 = float(ema20_series.iloc[i])
            ema50 = float(ema50_series.iloc[i])
            ema200 = float(ema200_series.iloc[i])
            vol_today = float(volumes.iloc[i])
            vol_avg20 = float(vol_avg20_series.iloc[i]) if not pd.isna(vol_avg20_series.iloc[i]) else 0.0
            ibs = float(ibs_series.iloc[i])
            nr7 = bool(nr7_series.iloc[i])
            open_price = float(opens.iloc[i])
            high = float(highs.iloc[i])
            low = float(lows.iloc[i])
            prior_sup_6m = float(support_6m.iloc[i]) if not pd.isna(support_6m.iloc[i]) else None
            prior_sup_1y = float(support_1y.iloc[i]) if not pd.isna(support_1y.iloc[i]) else None
            # Today's low can update support only if candle is green
            is_green_candle = price > open_price
            sup_6m = min(prior_sup_6m, low) if (is_green_candle and prior_sup_6m is not None) else prior_sup_6m
            sup_1y = min(prior_sup_1y, low) if (is_green_candle and prior_sup_1y is not None) else prior_sup_1y
            day = daily.index[i].date()

            if pd.isna(rsi2):
                continue

            # Previous bar values for crossover detection
            prev_close = float(closes.iloc[i - 1]) if i > 0 else None
            prev_ema20 = float(ema20_series.iloc[i - 1]) if i > 0 else None

            if strategy == "J":
                # Strategy J: Weekly Close Support Bounce (scale-out)
                # Entry: daily low within 1% of 26-week min weekly CLOSE
                # Stop: close < 26-week min weekly LOW
                # Exit 1: +5% → sell 50%   Exit 2: +10% → sell remaining

                w_support = float(weekly_support_series.iloc[i]) if weekly_support_series is not None and not pd.isna(weekly_support_series.iloc[i]) else None
                w_low_stop = float(weekly_low_stop_series.iloc[i]) if weekly_low_stop_series is not None and not pd.isna(weekly_low_stop_series.iloc[i]) else None

                entry_signal = False
                if w_support is not None and not in_position:
                    close_near = ((price - w_support) / w_support) * 100 if w_support > 0 else 999
                    cci_val = float(cci20_series.iloc[i]) if not pd.isna(cci20_series.iloc[i]) else 0.0
                    entry_signal = (close_near >= 0        # close above support
                                    and close_near <= 3.0  # close within +3% of support
                                    and ibs > 0.5          # bounce
                                    and price > open_price # green candle
                                    and cci_val > -100)    # CCI(20) not deeply oversold

                # Exits (scale-out: 50% at +5%, remaining at +8%)
                t1_price = entry_price * 1.05 if in_position else 0
                t2_price = entry_price * 1.10 if in_position else 0
                days_since_entry = (i - entry_bar) if in_position else 0

                def _j_trade(ep, shares_, exit_day, exit_px, reason):
                    t = self._make_trade(entry_date, ep, shares_, exit_day, exit_px, reason)
                    t["zone_bottom"] = round(entry_support_j, 2)
                    t["zone_top"] = None
                    t["zone_formed"] = None
                    t["rally_pct"] = None
                    t["vol_ratio"] = None
                    t["entry_dist_pct"] = round(((ep - entry_support_j) / entry_support_j) * 100, 2) if entry_support_j > 0 else 0
                    trades.append(t)

                def _j_stop(sz):
                    """Check stop, return True if stopped.
                    Skip support break if Nifty fell same or more since entry."""
                    nifty_shields = False
                    if nifty_at_entry > 0 and nifty_close is not None:
                        nifty_now = float(nifty_close.iloc[i])
                        nifty_pct = (nifty_now - nifty_at_entry) / nifty_at_entry
                        stock_pct = (price - entry_price) / entry_price
                        if nifty_pct <= stock_pct and nifty_pct < 0:
                            nifty_shields = True

                    if nifty_shields:
                        return False  # Ignore stop — market-wide fall

                    if price < entry_stop_j:
                        _j_trade(entry_price, sz, day, price, "SUPPORT_BREAK")
                        return True
                    return False

                if in_position and not partial_exit_done:
                    if _j_stop(shares):
                        in_position = False
                        partial_exit_done = False
                        continue
                    if price >= t1_price:
                        half = shares // 2
                        _j_trade(entry_price, half, day, price, "5PCT_PARTIAL")
                        remaining_shares = shares - half
                        partial_exit_done = True
                        if price >= t2_price:
                            _j_trade(entry_price, remaining_shares, day, price, "10PCT")
                            in_position = False
                            partial_exit_done = False
                        continue

                if in_position and partial_exit_done:
                    if _j_stop(remaining_shares):
                        in_position = False
                        partial_exit_done = False
                        continue
                    # Exit remaining: Close < 3-day low (skip if Nifty weak)
                    if i >= 3:
                        three_day_low = float(lows.iloc[max(0, i-3):i].min())
                        nifty_weak = False
                        if nifty_close is not None:
                            nifty_today = float(nifty_close.iloc[i])
                            nifty_3day_low_close = float(nifty_close.iloc[max(0, i-3):i].min())
                            nifty_weak = nifty_today < nifty_3day_low_close
                        if not nifty_weak and price < three_day_low:
                            _j_trade(entry_price, remaining_shares, day, price, "BELOW_3DAY_LOW")
                            in_position = False
                            partial_exit_done = False
                            continue
                    if price >= t2_price:
                        _j_trade(entry_price, remaining_shares, day, price, "10PCT")
                        in_position = False
                        partial_exit_done = False
                        continue

                # Entry (only when fully out)
                if entry_signal and not in_position:
                    shares = int(capital // price)
                    if shares > 0:
                        entry_price = price
                        entry_date = day
                        entry_bar = i
                        entry_support_j = w_support  # Weekly close support (for display)
                        entry_stop_j = w_low_stop if w_low_stop is not None else w_support
                        nifty_at_entry = float(nifty_close.iloc[i]) if nifty_close is not None else 0.0
                        in_position = True
                        partial_exit_done = False
                continue

            elif strategy == "K":
                # Strategy K: RS Trending Dip (scale-out)
                # Entry: RS > 0 AND prev RSI(2) < 20 AND today RSI(2) >= 20 AND Close > EMA(50)
                # Exit 1: +5% sell 50%
                # Exit 2: Close < 3-day low → sell remaining (skip if Nifty weak)
                # Stop: 3% below entry

                rs_val = float(rs_spread_series.iloc[i]) if (
                    rs_spread_series is not None
                    and not pd.isna(rs_spread_series.iloc[i])) else 0.0

                entry_signal = (rs_val > 0 and rsi2 >= 20
                               and not np.isnan(rsi2_prev) and rsi2_prev < 20
                               and price > ema50
                               and ibs > 0.5
                               and vol_today > vol_avg20)  # TEST: volume filter

                t1_price = entry_price * 1.05 if in_position else 0
                stop_3pct = entry_price * 0.97 if in_position else 0

                def _k_stop(sz):
                    """Check stop, return True if stopped.
                    Skip 3% SL if Nifty has fallen same or more since entry."""
                    nifty_drop_shields = False
                    if nifty_at_entry > 0 and nifty_close is not None:
                        nifty_now = float(nifty_close.iloc[i])
                        nifty_pct = (nifty_now - nifty_at_entry) / nifty_at_entry
                        stock_pct = (price - entry_price) / entry_price
                        if nifty_pct <= stock_pct and nifty_pct < 0:
                            nifty_drop_shields = True

                    if nifty_drop_shields:
                        return False  # Ignore SL — market-wide fall

                    if price < stop_3pct:
                        trades.append(self._make_trade(
                            entry_date, entry_price, sz, day,
                            price, "STOP_3PCT"))
                        return True
                    return False

                if in_position and not partial_exit_done:
                    if _k_stop(shares):
                        in_position = False
                        partial_exit_done = False
                        continue
                    if price >= t1_price:
                        half = shares // 2
                        trades.append(self._make_trade(
                            entry_date, entry_price, half, day,
                            price, "5PCT_PARTIAL"))
                        remaining_shares = shares - half
                        partial_exit_done = True
                        continue

                if in_position and partial_exit_done:
                    if _k_stop(remaining_shares):
                        in_position = False
                        partial_exit_done = False
                        continue
                    # Exit remaining: Close < 3-day low (skip if Nifty weak)
                    # if rs_val < 0:  # OLD: RS drops below 0
                    three_day_low = float(lows.iloc[max(0, i-3):i].min()) if i >= 3 else float(lows.iloc[i])
                    nifty_weak = False
                    if nifty_close is not None and i >= 3:
                        nifty_today = float(nifty_close.iloc[i])
                        nifty_3day_low_close = float(nifty_close.iloc[max(0, i-3):i].min())
                        nifty_weak = nifty_today < nifty_3day_low_close
                    if not nifty_weak and price < three_day_low:
                        trades.append(self._make_trade(
                            entry_date, entry_price, remaining_shares,
                            day, price, "BELOW_3DAY_LOW"))
                        in_position = False
                        partial_exit_done = False
                        continue

                # Entry (only when fully out)
                if entry_signal and not in_position:
                    shares = int(capital // price)
                    if shares > 0:
                        entry_price = price
                        entry_date = day
                        in_position = True
                        partial_exit_done = False
                        nifty_at_entry = float(nifty_close.iloc[i]) if nifty_close is not None else 0.0
                continue

            elif strategy == "N":
                # Strategy N: CCI Momentum
                # Entry: CCI(20) crosses above 100 (prev < 100, today > 100)
                # Exit: CCI(20) < 100
                cci_val = float(cci20_series.iloc[i]) if not pd.isna(cci20_series.iloc[i]) else 0.0
                cci_prev = float(cci20_series.iloc[i-1]) if i > 0 and not pd.isna(cci20_series.iloc[i-1]) else 0.0

                if in_position and not n_partial_done:
                    # Hard stop-loss: 3%
                    if price <= entry_price * 0.97:
                        trades.append(self._make_trade(
                            entry_date, entry_price, shares, day,
                            price, "HARD_SL_3PCT"))
                        in_position = False
                        n_partial_done = False
                        continue
                    # Partial: +5% → sell 50%
                    if price >= entry_price * 1.05:
                        half = shares // 2
                        if half > 0:
                            trades.append(self._make_trade(
                                entry_date, entry_price, half, day,
                                price, "PARTIAL_5PCT"))
                            n_remaining = shares - half
                            n_partial_done = True
                    # CCI exit (full)
                    elif cci_val < -100 and price < open_price:
                        trades.append(self._make_trade(
                            entry_date, entry_price, shares, day,
                            price, "CCI_RED_EXIT"))
                        in_position = False
                        n_partial_done = False
                        continue

                if in_position and n_partial_done:
                    # Hard stop-loss: 3% on remaining
                    if price <= entry_price * 0.97:
                        trades.append(self._make_trade(
                            entry_date, entry_price, n_remaining, day,
                            price, "HARD_SL_3PCT"))
                        in_position = False
                        n_partial_done = False
                        continue
                    # CCI exit (remaining)
                    if cci_val < -100 and price < open_price:
                        trades.append(self._make_trade(
                            entry_date, entry_price, n_remaining, day,
                            price, "CCI_RED_EXIT"))
                        in_position = False
                        n_partial_done = False
                        continue

                if not in_position:
                    if cci_val > 100 and cci_prev <= 100 and price > open_price:
                        shares = int(capital // price)
                        if shares > 0:
                            entry_price = price
                            entry_date = day
                            in_position = True
                            n_partial_done = False
                            n_remaining = shares
                continue

            elif strategy == "O":
                # Strategy O: Connors RSI(2) Mean Reversion
                # Entry: Price > SMA(200) AND RSI(2) < 5
                # Exit: Close > EMA(5)
                # Stop: 3% hard SL
                sma200 = float(sma200_series.iloc[i]) if not pd.isna(sma200_series.iloc[i]) else None
                ema5 = float(ema5_series.iloc[i])

                if in_position:
                    if price <= entry_price * 0.97:
                        trades.append(self._make_trade(
                            entry_date, entry_price, shares, day,
                            price, "HARD_SL_3PCT"))
                        in_position = False
                        continue
                    if price > ema5:
                        trades.append(self._make_trade(
                            entry_date, entry_price, shares, day,
                            price, "EMA5_EXIT"))
                        in_position = False
                        continue

                if not in_position and sma200 is not None and rsi2 < 5 and price > sma200:
                    shares = int(capital // price)
                    if shares > 0:
                        entry_price = price
                        entry_date = day
                        in_position = True
                continue

            elif strategy == "R":
                # Strategy R: RSI Momentum
                # Entry: RSI(14) crosses above 60 AND Price > EMA(200) AND green candle
                # Partial: +5% → sell 50%
                # Exit: RSI(14) < 50 → sell remaining
                # Stop: 5% hard SL
                rsi14 = float(rsi14_series.iloc[i]) if not pd.isna(rsi14_series.iloc[i]) else 50.0
                rsi14_prev = float(rsi14_series.iloc[i - 1]) if i > 0 and not pd.isna(rsi14_series.iloc[i - 1]) else 50.0

                if in_position and not r_partial_done:
                    if price <= entry_price * 0.95:
                        trades.append(self._make_trade(
                            entry_date, entry_price, shares, day,
                            price, "HARD_SL_5PCT"))
                        in_position = False
                        r_partial_done = False
                        continue
                    if price >= entry_price * 1.05:
                        half = shares // 2
                        if half > 0:
                            trades.append(self._make_trade(
                                entry_date, entry_price, half, day,
                                price, "PARTIAL_5PCT"))
                            r_remaining = shares - half
                            r_partial_done = True
                    elif rsi14 < 50:
                        trades.append(self._make_trade(
                            entry_date, entry_price, shares, day,
                            price, "RSI14_EXIT"))
                        in_position = False
                        r_partial_done = False
                        continue

                if in_position and r_partial_done:
                    if price <= entry_price * 0.95:
                        trades.append(self._make_trade(
                            entry_date, entry_price, r_remaining, day,
                            price, "HARD_SL_5PCT"))
                        in_position = False
                        r_partial_done = False
                        continue
                    if rsi14 < 50:
                        trades.append(self._make_trade(
                            entry_date, entry_price, r_remaining, day,
                            price, "RSI14_EXIT"))
                        in_position = False
                        r_partial_done = False
                        continue

                if not in_position:
                    if (rsi14_prev <= 60 and rsi14 > 60
                            and price > ema200 and price > open_price):
                        shares = int(capital // price)
                        if shares > 0:
                            entry_price = price
                            entry_date = day
                            in_position = True
                            r_partial_done = False
                            r_remaining = shares
                continue

            elif strategy == "S":
                # Strategy S: MACD + RSI Combo
                # Entry: MACD crosses above signal AND MACD > 0 AND RSI(14) > 50 AND green candle
                # Partial: +5% → sell 50%
                # Exit: MACD crosses below signal OR RSI(14) < 40 → sell remaining
                # Stop: 5% hard SL
                rsi14 = float(rsi14_series.iloc[i]) if not pd.isna(rsi14_series.iloc[i]) else 50.0
                macd_val = float(macd_line_series.iloc[i]) if not pd.isna(macd_line_series.iloc[i]) else 0.0
                macd_sig = float(macd_signal_series.iloc[i]) if not pd.isna(macd_signal_series.iloc[i]) else 0.0
                macd_prev = float(macd_line_series.iloc[i - 1]) if i > 0 and not pd.isna(macd_line_series.iloc[i - 1]) else 0.0
                macd_sig_prev = float(macd_signal_series.iloc[i - 1]) if i > 0 and not pd.isna(macd_signal_series.iloc[i - 1]) else 0.0

                macd_cross_up = (macd_prev <= macd_sig_prev and macd_val > macd_sig)
                macd_cross_down = (macd_prev >= macd_sig_prev and macd_val < macd_sig)

                if in_position and not s_partial_done:
                    if price <= entry_price * 0.95:
                        trades.append(self._make_trade(
                            entry_date, entry_price, shares, day,
                            price, "HARD_SL_5PCT"))
                        in_position = False
                        s_partial_done = False
                        continue
                    if price >= entry_price * 1.05:
                        half = shares // 2
                        if half > 0:
                            trades.append(self._make_trade(
                                entry_date, entry_price, half, day,
                                price, "PARTIAL_5PCT"))
                            s_remaining = shares - half
                            s_partial_done = True
                    elif macd_cross_down or rsi14 < 40:
                        reason = "MACD_CROSS_DOWN" if macd_cross_down else "RSI14_EXIT"
                        trades.append(self._make_trade(
                            entry_date, entry_price, shares, day,
                            price, reason))
                        in_position = False
                        s_partial_done = False
                        continue

                if in_position and s_partial_done:
                    if price <= entry_price * 0.95:
                        trades.append(self._make_trade(
                            entry_date, entry_price, s_remaining, day,
                            price, "HARD_SL_5PCT"))
                        in_position = False
                        s_partial_done = False
                        continue
                    if macd_cross_down or rsi14 < 40:
                        reason = "MACD_CROSS_DOWN" if macd_cross_down else "RSI14_EXIT"
                        trades.append(self._make_trade(
                            entry_date, entry_price, s_remaining, day,
                            price, reason))
                        in_position = False
                        s_partial_done = False
                        continue

                if not in_position:
                    if (macd_cross_up and macd_val > 0
                            and rsi14 > 50 and price > open_price):
                        shares = int(capital // price)
                        if shares > 0:
                            entry_price = price
                            entry_date = day
                            in_position = True
                            s_partial_done = False
                            s_remaining = shares
                continue

            elif strategy == "T":
                # Strategy T: Keltner Channel Pullback
                # Entry: Price near EMA(20) (within 1%) AND was above upper Keltner in last 10 bars AND green candle
                # Partial: +5% → sell 50%
                # Exit: Price >= upper Keltner → sell remaining
                # Stop: 5% hard SL
                atr14 = float(atr14_series.iloc[i]) if not pd.isna(atr14_series.iloc[i]) else 0.0
                upper_keltner = ema20 + 2 * atr14
                t_entry_upper_keltner = 0.0  # will be set at entry

                if in_position and not t_partial_done:
                    current_upper = ema20 + 2 * atr14
                    if price <= entry_price * 0.95:
                        trades.append(self._make_trade(
                            entry_date, entry_price, shares, day,
                            price, "HARD_SL_5PCT"))
                        in_position = False
                        t_partial_done = False
                        continue
                    if price >= entry_price * 1.05:
                        half = shares // 2
                        if half > 0:
                            trades.append(self._make_trade(
                                entry_date, entry_price, half, day,
                                price, "PARTIAL_5PCT"))
                            t_remaining = shares - half
                            t_partial_done = True
                    elif price >= current_upper:
                        trades.append(self._make_trade(
                            entry_date, entry_price, shares, day,
                            price, "KELTNER_UPPER_EXIT"))
                        in_position = False
                        t_partial_done = False
                        continue

                if in_position and t_partial_done:
                    current_upper = ema20 + 2 * atr14
                    if price <= entry_price * 0.95:
                        trades.append(self._make_trade(
                            entry_date, entry_price, t_remaining, day,
                            price, "HARD_SL_5PCT"))
                        in_position = False
                        t_partial_done = False
                        continue
                    if price >= current_upper:
                        trades.append(self._make_trade(
                            entry_date, entry_price, t_remaining, day,
                            price, "KELTNER_UPPER_EXIT"))
                        in_position = False
                        t_partial_done = False
                        continue

                if not in_position and atr14 > 0:
                    near_ema20 = abs(price - ema20) / ema20 <= 0.01
                    was_at_upper = False
                    for lookback_j in range(max(0, i - 10), i):
                        past_high = float(highs.iloc[lookback_j])
                        past_ema20 = float(ema20_series.iloc[lookback_j])
                        past_atr14 = float(atr14_series.iloc[lookback_j]) if not pd.isna(atr14_series.iloc[lookback_j]) else 0.0
                        past_upper = past_ema20 + 2 * past_atr14
                        if past_high >= past_upper:
                            was_at_upper = True
                            break
                    if near_ema20 and was_at_upper and price > open_price:
                        shares = int(capital // price)
                        if shares > 0:
                            entry_price = price
                            entry_date = day
                            in_position = True
                            t_partial_done = False
                            t_remaining = shares
                continue

            elif strategy == "U":
                # Strategy U: Stochastic RSI
                # Entry: StochRSI %K crosses above %D, both < 30 AND Price > EMA(200)
                # Partial: +5% → sell 50%
                # Exit: StochRSI %K crosses below %D, both > 70 → sell remaining
                # Stop: 5% hard SL
                stoch_k = float(stoch_k_series.iloc[i]) if not pd.isna(stoch_k_series.iloc[i]) else 50.0
                stoch_d = float(stoch_d_series.iloc[i]) if not pd.isna(stoch_d_series.iloc[i]) else 50.0
                stoch_k_prev = float(stoch_k_series.iloc[i - 1]) if i > 0 and not pd.isna(stoch_k_series.iloc[i - 1]) else 50.0
                stoch_d_prev = float(stoch_d_series.iloc[i - 1]) if i > 0 and not pd.isna(stoch_d_series.iloc[i - 1]) else 50.0

                k_cross_up = (stoch_k_prev <= stoch_d_prev and stoch_k > stoch_d)
                k_cross_down = (stoch_k_prev >= stoch_d_prev and stoch_k < stoch_d)

                if in_position and not u_partial_done:
                    if price <= entry_price * 0.95:
                        trades.append(self._make_trade(
                            entry_date, entry_price, shares, day,
                            price, "HARD_SL_5PCT"))
                        in_position = False
                        u_partial_done = False
                        continue
                    if price >= entry_price * 1.05:
                        half = shares // 2
                        if half > 0:
                            trades.append(self._make_trade(
                                entry_date, entry_price, half, day,
                                price, "PARTIAL_5PCT"))
                            u_remaining = shares - half
                            u_partial_done = True
                    elif k_cross_down and stoch_k > 70 and stoch_d > 70:
                        trades.append(self._make_trade(
                            entry_date, entry_price, shares, day,
                            price, "STOCHRSI_EXIT"))
                        in_position = False
                        u_partial_done = False
                        continue

                if in_position and u_partial_done:
                    if price <= entry_price * 0.95:
                        trades.append(self._make_trade(
                            entry_date, entry_price, u_remaining, day,
                            price, "HARD_SL_5PCT"))
                        in_position = False
                        u_partial_done = False
                        continue
                    if k_cross_down and stoch_k > 70 and stoch_d > 70:
                        trades.append(self._make_trade(
                            entry_date, entry_price, u_remaining, day,
                            price, "STOCHRSI_EXIT"))
                        in_position = False
                        u_partial_done = False
                        continue

                if not in_position:
                    if (k_cross_up and stoch_k < 30 and stoch_d < 30
                            and price > ema200):
                        shares = int(capital // price)
                        if shares > 0:
                            entry_price = price
                            entry_date = day
                            in_position = True
                            u_partial_done = False
                            u_remaining = shares
                continue

            elif strategy == "V":
                # Strategy V: Donchian Breakout with Volume
                # Entry: Close > 20-day high AND Vol > 1.5x avg(20) AND Price > EMA(200) AND green
                # Partial: +5% → sell 50%
                # Exit: Close < 10-day low → sell remaining
                # Stop: 5% hard SL
                h20 = float(high_20d_v.iloc[i]) if not pd.isna(high_20d_v.iloc[i]) else None
                l10 = float(low_10d_v.iloc[i]) if not pd.isna(low_10d_v.iloc[i]) else None
                vol_val = float(volumes.iloc[i])
                vol_avg = float(vol_avg20_v.iloc[i]) if not pd.isna(vol_avg20_v.iloc[i]) else 0.0

                if in_position and not v_partial_done:
                    if price <= entry_price * 0.95:
                        trades.append(self._make_trade(
                            entry_date, entry_price, shares, day,
                            price, "HARD_SL_5PCT"))
                        in_position = False
                        v_partial_done = False
                        continue
                    if price >= entry_price * 1.05:
                        half = shares // 2
                        if half > 0:
                            trades.append(self._make_trade(
                                entry_date, entry_price, half, day,
                                price, "PARTIAL_5PCT"))
                            v_remaining = shares - half
                            v_partial_done = True
                    elif l10 is not None and l10 > 0 and price < l10:
                        trades.append(self._make_trade(
                            entry_date, entry_price, shares, day,
                            price, "DONCHIAN_LOW_EXIT"))
                        in_position = False
                        v_partial_done = False
                        continue

                if in_position and v_partial_done:
                    if price <= entry_price * 0.95:
                        trades.append(self._make_trade(
                            entry_date, entry_price, v_remaining, day,
                            price, "HARD_SL_5PCT"))
                        in_position = False
                        v_partial_done = False
                        continue
                    if l10 is not None and l10 > 0 and price < l10:
                        trades.append(self._make_trade(
                            entry_date, entry_price, v_remaining, day,
                            price, "DONCHIAN_LOW_EXIT"))
                        in_position = False
                        v_partial_done = False
                        continue

                if not in_position:
                    if (h20 is not None and price > h20
                            and vol_avg > 0 and vol_val > 1.5 * vol_avg
                            and price > ema200 and is_green):
                        shares = int(capital // price)
                        if shares > 0:
                            entry_price = price
                            entry_date = day
                            in_position = True
                            v_partial_done = False
                            v_remaining = shares
                continue

        # Close open position at end of backtest
        if in_position:
            last_day = daily.index[bt_indices[-1]].date()
            last_price = float(closes.iloc[bt_indices[-1]])
            if partial_exit_done:
                trades.append(self._make_trade(
                    entry_date, entry_price, remaining_shares, last_day,
                    last_price, "BACKTEST_END"))
            else:
                trades.append(self._make_trade(
                    entry_date, entry_price, shares, last_day, last_price,
                    "BACKTEST_END"))

        trading_days_dates = [daily.index[i].date() for i in bt_indices]
        summary = self._calculate_summary(trades, capital)

        return {
            "symbol": symbol,
            "strategy": strategy,
            "exit_target": str(exit_target) if strategy in ("J", "K") and exit_target else None,
            "start_date": trading_days_dates[0].isoformat(),
            "end_date": trading_days_dates[-1].isoformat(),
            "trading_days": len(bt_indices),
            "capital": capital,
            "trades": trades,
            "summary": summary,
        }

    def _make_trade(self, entry_date, entry_price, shares, exit_date,
                    exit_price, reason):
        pnl = round((exit_price - entry_price) * shares, 2)
        pnl_pct = round(((exit_price - entry_price) / entry_price) * 100, 2)
        holding = (exit_date - entry_date).days
        return {
            "entry_date": entry_date.isoformat(),
            "entry_time": "15:15",
            "entry_price": round(entry_price, 2),
            "shares": shares,
            "exit_date": exit_date.isoformat(),
            "exit_time": "15:15",
            "exit_price": round(exit_price, 2),
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "holding_days": holding,
            "exit_reason": reason,
        }

    def _empty_summary(self):
        return {
            "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
            "win_rate": 0, "total_pnl": 0, "total_return_pct": 0,
            "avg_win": 0, "avg_loss": 0, "largest_win": 0, "largest_loss": 0,
            "profit_factor": 0, "avg_holding_days": 0,
        }

    def _calculate_summary(self, trades, capital):
        if not trades:
            return self._empty_summary()

        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in trades)

        avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0

        gross_wins = sum(t["pnl"] for t in wins)
        gross_losses = abs(sum(t["pnl"] for t in losses))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else 999.99

        return {
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(len(wins) / len(trades) * 100, 1),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_pnl / capital * 100, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "largest_win": round(max((t["pnl"] for t in trades), default=0), 2),
            "largest_loss": round(min((t["pnl"] for t in trades), default=0), 2),
            "profit_factor": round(min(profit_factor, 999.99), 2),
            "avg_holding_days": round(
                sum(t["holding_days"] for t in trades) / len(trades), 1),
        }

    def run_portfolio_backtest(self, period_days, universe=50,
                              capital_lakhs=10, per_stock=50000,
                              strategies=None, entries_per_day=1,
                              progress_callback=None, end_date=None):
        """
        Portfolio-level backtest with configurable capital and strategies.
        capital_lakhs: 10 or 20 (total capital in lakhs)
        strategies: list of strategy codes, e.g. ["J"], ["K"], ["J","K"]
        ₹50K per stock, max positions = capital / 50K.
        K signals prioritized over J when slots are limited.
        """
        from data.momentum_engine import NIFTY_50_TICKERS

        if strategies is None:
            strategies = ["J", "K"]

        TOTAL_CAPITAL = capital_lakhs * 100000
        PER_STOCK = per_stock
        MAX_POSITIONS = TOTAL_CAPITAL // PER_STOCK

        if universe <= 50:
            tickers = NIFTY_50_TICKERS
        else:
            tickers = NIFTY_50_TICKERS + NIFTY_NEXT50_TICKERS

        end_date = end_date or datetime.now()
        daily_start = end_date - timedelta(days=period_days + 500)
        bt_start_date = (end_date - timedelta(days=period_days)).date()

        # --- Phase 1: Fetch all data ---
        stock_data = {}  # ticker -> DataFrame
        total = len(tickers)

        # Fetch Nifty index data once
        try:
            nifty_raw = yf.Ticker("^NSEI").history(start=daily_start, end=end_date)
        except Exception:
            nifty_raw = pd.DataFrame()

        for idx, ticker in enumerate(tickers):
            if progress_callback:
                progress_callback(idx + 1, total, f"Fetching {ticker}")

            nse_symbol = f"{ticker}.NS"
            try:
                daily = yf.Ticker(nse_symbol).history(start=daily_start, end=end_date)
            except Exception:
                daily = pd.DataFrame()

            if daily.empty or len(daily) < 210:
                continue

            stock_data[ticker] = daily

        if not stock_data:
            return {"error": "No valid stock data", "trades": [],
                    "summary": self._empty_summary()}

        # --- Phase 2: Pre-compute indicators per stock ---
        indicators = {}  # ticker -> dict of series

        for ticker, daily in stock_data.items():
            closes = daily["Close"]
            opens = daily["Open"]
            highs = daily["High"]
            lows = daily["Low"]

            hl_range = highs - lows
            ibs_series = ((closes - lows) / hl_range).where(hl_range > 0, 0.5)

            # Support levels
            support_6m = lows.rolling(window=120, min_periods=120).min().shift(1)
            support_1y = lows.rolling(window=252, min_periods=252).min().shift(1)
            # Weekly support for Strategy J
            weekly = daily.resample("W-FRI").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum"
            }).dropna()
            w_support = weekly["Close"].rolling(window=26, min_periods=26).min()
            weekly_support_series = w_support.reindex(daily.index, method="ffill")
            w_low_stop = weekly["Low"].rolling(window=26, min_periods=26).min()
            weekly_low_stop_series = w_low_stop.reindex(daily.index, method="ffill")

            # Volume
            vol_series = daily["Volume"].astype(float)
            vol_avg20 = vol_series.rolling(window=20, min_periods=20).mean()

            # EMAs
            ema5_series = closes.ewm(span=5, adjust=False).mean()
            ema8_series = closes.ewm(span=8, adjust=False).mean()
            ema10_series = closes.ewm(span=10, adjust=False).mean()
            ema20_series = closes.ewm(span=20, adjust=False).mean()
            ema50_series = closes.ewm(span=50, adjust=False).mean()
            ema200_series = closes.ewm(span=200, adjust=False).mean()

            # RSI(2) and RSI(3)
            rsi2_series = self._calculate_rsi_series(closes, 2)
            rsi3_series = self._calculate_rsi_series(closes, 3)

            # RS spread for Strategy K (3M) and L (6M)
            rs_spread_series = None
            if not nifty_raw.empty:
                nifty_close = nifty_raw["Close"].reindex(daily.index, method="ffill")
                stock_3m = closes.pct_change(periods=63)
                nifty_3m = nifty_close.pct_change(periods=63)
                rs_spread_series = stock_3m - nifty_3m

            # CCI(20) for Strategy J entry confirmation
            cci20_series = self._calculate_cci_series(highs, lows, closes, 20)

            # 20-day rolling high of Highs for Strategy N (excluding current bar)
            high_20d = highs.rolling(window=20, min_periods=20).max().shift(1)

            # --- New indicators for O, R, S, T, U ---
            # RSI(14) for strategies O, R, S, U
            rsi14_series = self._calculate_rsi_series(closes, 14)

            # SMA(200) and SMA(5) for Strategy O
            sma200_series = closes.rolling(window=200, min_periods=200).mean()
            sma5_series = closes.rolling(window=5, min_periods=5).mean()

            # MACD(12,26,9) for Strategy S
            ema12_series = closes.ewm(span=12, adjust=False).mean()
            ema26_series = closes.ewm(span=26, adjust=False).mean()
            macd_line_series = ema12_series - ema26_series
            macd_signal_series = macd_line_series.ewm(span=9, adjust=False).mean()

            # ATR(14) for Strategy T (Keltner Channel)
            prev_close_s = closes.shift(1)
            tr1 = highs - lows
            tr2 = (highs - prev_close_s).abs()
            tr3 = (lows - prev_close_s).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr14_series = true_range.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

            # Stochastic RSI(14,14,3,3) for Strategy U
            rsi14_min = rsi14_series.rolling(window=14, min_periods=14).min()
            rsi14_max = rsi14_series.rolling(window=14, min_periods=14).max()
            stoch_rsi_raw = (rsi14_series - rsi14_min) / (rsi14_max - rsi14_min)
            stoch_rsi_raw = stoch_rsi_raw.fillna(0.5)
            stoch_k_series = stoch_rsi_raw.rolling(window=3, min_periods=3).mean() * 100
            stoch_d_series = stoch_k_series.rolling(window=3, min_periods=3).mean()

            # Donchian channels for Strategy V
            high_20d_v = highs.rolling(window=20, min_periods=20).max().shift(1)  # prev 20-day high
            low_10d_v = lows.rolling(window=10, min_periods=10).min().shift(1)    # prev 10-day low

            # Backtest indices (need 200+ bars warmup)
            bt_indices = [i for i, ts in enumerate(daily.index)
                          if ts.date() >= bt_start_date and i >= 200]

            indicators[ticker] = {
                "daily": daily,
                "closes": closes,
                "opens": opens,
                "highs": highs,
                "lows": lows,
                "ibs": ibs_series,
                "support_6m": support_6m,
                "support_1y": support_1y,
                "weekly_support": weekly_support_series,
                "weekly_low_stop": weekly_low_stop_series,
                "rsi2": rsi2_series,
                "rsi3": rsi3_series,
                "rsi14": rsi14_series,
                "cci20": cci20_series,
                "volume": vol_series,
                "vol_avg20": vol_avg20,
                "high_20d": high_20d,
                "sma200": sma200_series,
                "sma5": sma5_series,
                "macd_line": macd_line_series,
                "macd_signal": macd_signal_series,
                "atr14": atr14_series,
                "stoch_k": stoch_k_series,
                "stoch_d": stoch_d_series,
                "high_20d_v": high_20d_v,
                "low_10d_v": low_10d_v,
                "ema5": ema5_series,
                "ema8": ema8_series,
                "ema10": ema10_series,
                "ema20": ema20_series,
                "ema50": ema50_series,
                "ema200": ema200_series,
                "rs_spread": rs_spread_series,
                "bt_indices": bt_indices,
            }

        # --- Phase 3: Build union of all trading dates ---
        all_dates = set()
        for ticker, ind in indicators.items():
            for i in ind["bt_indices"]:
                all_dates.add(ind["daily"].index[i].date())
        all_dates = sorted(all_dates)

        if not all_dates:
            return {"error": "No trading days in selected period",
                    "trades": [], "summary": self._empty_summary()}

        # Build Nifty close lookup by date (for SL filter & 3-day low weakness)
        nifty_close_by_date = {}  # date -> close price
        nifty_above_ema200_by_date = {}  # date -> bool (Nifty > EMA200)
        nifty_cci_by_date = {}  # date -> CCI(20) value (for Strategy N regime filter)
        if not nifty_raw.empty:
            nifty_ema200 = nifty_raw["Close"].ewm(span=200, adjust=False).mean()
            # Nifty CCI(20)
            nifty_tp = (nifty_raw["High"] + nifty_raw["Low"] + nifty_raw["Close"]) / 3
            nifty_sma_tp = nifty_tp.rolling(window=20, min_periods=20).mean()
            nifty_mean_dev = nifty_tp.rolling(window=20, min_periods=20).apply(
                lambda x: np.mean(np.abs(x - x.mean())), raw=True)
            nifty_cci_series = (nifty_tp - nifty_sma_tp) / (0.015 * nifty_mean_dev)
            for ts, row in nifty_raw.iterrows():
                nifty_close_by_date[ts.date()] = float(row["Close"])
                nifty_above_ema200_by_date[ts.date()] = float(row["Close"]) > float(nifty_ema200.loc[ts])
                nifty_cci_by_date[ts.date()] = float(nifty_cci_series.loc[ts]) if not pd.isna(nifty_cci_series.loc[ts]) else 0.0

        # Build per-stock date->index mapping
        date_to_idx = {}  # ticker -> {date: iloc_index}
        for ticker, ind in indicators.items():
            mapping = {}
            for i in ind["bt_indices"]:
                d = ind["daily"].index[i].date()
                mapping[d] = i
            date_to_idx[ticker] = mapping

        # --- Phase 4: Day-by-day simulation ---
        positions: List[Dict] = []  # open positions
        trades: List[Dict] = []     # completed trades
        total_signals = 0
        missed_signals = 0
        max_positions_used = 0
        positions_over_time = []

        for day_idx, day in enumerate(all_dates):
            if progress_callback and day_idx % 20 == 0:
                progress_callback(
                    total + day_idx, total + len(all_dates),
                    f"Simulating {day}")

            # Check if Nifty is weak today (close < 3-day low close)
            nifty_weak_today = False
            nifty_today_close = nifty_close_by_date.get(day, 0.0)
            if day in nifty_close_by_date and day_idx >= 3:
                past_nifty_closes = [nifty_close_by_date[d]
                                     for d in all_dates[max(0, day_idx-3):day_idx]
                                     if d in nifty_close_by_date]
                if past_nifty_closes:
                    nifty_3day_low_close = min(past_nifty_closes)
                    nifty_weak_today = nifty_today_close < nifty_3day_low_close

            # === 1. Process exits first (frees capital) ===
            still_open = []
            for pos in positions:
                ticker = pos["symbol"]
                if day not in date_to_idx.get(ticker, {}):
                    still_open.append(pos)
                    continue

                i = date_to_idx[ticker][day]
                ind = indicators[ticker]
                price = float(ind["closes"].iloc[i])
                high = float(ind["highs"].iloc[i])
                low = float(ind["lows"].iloc[i])
                open_price = float(ind["opens"].iloc[i])

                exited = False

                if pos["strategy"] == "J":
                    entry_stop = pos["entry_stop_j"]

                    # Nifty drop shield: skip support break if Nifty fell same or more
                    j_nifty_shields = False
                    j_nifty_entry = pos.get("nifty_at_entry", 0.0)
                    if j_nifty_entry > 0 and nifty_today_close > 0:
                        nifty_pct = (nifty_today_close - j_nifty_entry) / j_nifty_entry
                        stock_pct = (price - pos["entry_price"]) / pos["entry_price"]
                        if nifty_pct <= stock_pct and nifty_pct < 0:
                            j_nifty_shields = True

                    # J exits: +5% partial, +10% full, support break stop
                    t1 = pos["entry_price"] * 1.05
                    t2 = pos["entry_price"] * 1.10
                    if not pos["partial_exit_done"]:
                        if not j_nifty_shields and price < entry_stop:
                            trades.append(self._make_portfolio_trade(
                                pos, pos["shares"], day, price, "SUPPORT_BREAK"))
                            exited = True
                        elif price >= t1:
                            half = pos["shares"] // 2
                            trades.append(self._make_portfolio_trade(
                                pos, half, day, price, "5PCT_PARTIAL"))
                            pos["remaining_shares"] = pos["shares"] - half
                            pos["partial_exit_done"] = True
                            if price >= t2:
                                trades.append(self._make_portfolio_trade(
                                    pos, pos["remaining_shares"], day, price, "10PCT"))
                                exited = True
                    else:
                        if not j_nifty_shields and price < entry_stop:
                            trades.append(self._make_portfolio_trade(
                                pos, pos["remaining_shares"], day, price, "SUPPORT_BREAK"))
                            exited = True
                        elif not nifty_weak_today and day_idx >= 3:
                            three_day_low = float(ind["lows"].iloc[max(0, i-3):i].min())
                            if price < three_day_low:
                                    trades.append(self._make_portfolio_trade(
                                        pos, pos["remaining_shares"], day, price, "BELOW_3DAY_LOW"))
                                    exited = True
                            if not exited and price >= t2:
                                trades.append(self._make_portfolio_trade(
                                    pos, pos["remaining_shares"], day, price, "10PCT"))
                                exited = True

                elif pos["strategy"] == "K":
                    # K exits (close mode): 3% stop, +5% partial, 3-day low remaining
                    stop_3pct = pos["entry_price"] * 0.97
                    t1 = pos["entry_price"] * 1.05
                    rs_val = 0.0
                    if ind["rs_spread"] is not None and not pd.isna(ind["rs_spread"].iloc[i]):
                        rs_val = float(ind["rs_spread"].iloc[i])

                    # Skip 3% SL if Nifty fallen same/more OR RS still positive
                    nifty_entry = pos.get("nifty_at_entry", 0.0)
                    nifty_now = nifty_close_by_date.get(day, 0.0)
                    k_nifty_shields = False
                    if nifty_entry > 0 and nifty_now > 0:
                        nifty_pct = (nifty_now - nifty_entry) / nifty_entry
                        stock_pct = (price - pos["entry_price"]) / pos["entry_price"]
                        if nifty_pct <= stock_pct and nifty_pct < 0:
                            k_nifty_shields = True

                    if not pos["partial_exit_done"]:
                        if not k_nifty_shields and price < stop_3pct:
                            trades.append(self._make_portfolio_trade(
                                pos, pos["shares"], day, price, "STOP_3PCT"))
                            exited = True
                        elif price >= t1:
                            half = pos["shares"] // 2
                            trades.append(self._make_portfolio_trade(
                                pos, half, day, price, "5PCT_PARTIAL"))
                            pos["remaining_shares"] = pos["shares"] - half
                            pos["partial_exit_done"] = True
                    else:
                        if not k_nifty_shields and price < stop_3pct:
                            trades.append(self._make_portfolio_trade(
                                pos, pos["remaining_shares"], day, price,
                                "STOP_3PCT"))
                            exited = True
                        # Exit remaining: Close < 3-day low (skip if Nifty weak)
                        elif not nifty_weak_today:
                            three_day_low = float(ind["lows"].iloc[max(0, i-3):i].min()) if i >= 3 else low
                            if price < three_day_low:
                                trades.append(self._make_portfolio_trade(
                                    pos, pos["remaining_shares"], day, price,
                                    "BELOW_3DAY_LOW"))
                                exited = True

                elif pos["strategy"] == "N":
                    cci_val = float(ind["cci20"].iloc[i]) if not pd.isna(ind["cci20"].iloc[i]) else 0.0
                    # N hard stop-loss: 3%
                    if price <= pos["entry_price"] * 0.97:
                        sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                        trades.append(self._make_portfolio_trade(
                            pos, sz, day, price, "HARD_SL_3PCT"))
                        exited = True
                    # N partial: +5% → sell 50%
                    if not exited and not pos["partial_exit_done"] and price >= pos["entry_price"] * 1.05:
                        half = pos["shares"] // 2
                        if half > 0:
                            trades.append(self._make_portfolio_trade(
                                pos, half, day, price, "PARTIAL_5PCT"))
                            pos["remaining_shares"] = pos["shares"] - half
                            pos["partial_exit_done"] = True
                    # N CCI exit: CCI < -100 AND red candle
                    if not exited and cci_val < -100 and price < open_price:
                        sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                        trades.append(self._make_portfolio_trade(
                            pos, sz, day, price, "CCI_RED_EXIT"))
                        exited = True

                elif pos["strategy"] == "O":
                    # O exits: 3% hard SL, Close > EMA(5)
                    ema5_val = float(ind["ema5"].iloc[i]) if not pd.isna(ind["ema5"].iloc[i]) else None
                    if price <= pos["entry_price"] * 0.97:
                        trades.append(self._make_portfolio_trade(
                            pos, pos["shares"], day, price, "HARD_SL_3PCT"))
                        exited = True
                    if not exited and ema5_val is not None and price > ema5_val:
                        trades.append(self._make_portfolio_trade(
                            pos, pos["shares"], day, price, "EMA5_EXIT"))
                        exited = True

                elif pos["strategy"] == "R":
                    # R exits: 5% hard SL, +5% partial, RSI(14) < 50 remaining
                    rsi14_val = float(ind["rsi14"].iloc[i]) if not pd.isna(ind["rsi14"].iloc[i]) else 50.0
                    if price <= pos["entry_price"] * 0.95:
                        sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                        trades.append(self._make_portfolio_trade(
                            pos, sz, day, price, "HARD_SL_5PCT"))
                        exited = True
                    if not exited and not pos["partial_exit_done"] and price >= pos["entry_price"] * 1.05:
                        half = pos["shares"] // 2
                        if half > 0:
                            trades.append(self._make_portfolio_trade(
                                pos, half, day, price, "PARTIAL_5PCT"))
                            pos["remaining_shares"] = pos["shares"] - half
                            pos["partial_exit_done"] = True
                    if not exited and rsi14_val < 50:
                        sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                        trades.append(self._make_portfolio_trade(
                            pos, sz, day, price, "RSI14_EXIT"))
                        exited = True

                elif pos["strategy"] == "S":
                    # S exits: 5% hard SL, +5% partial, MACD cross down OR RSI(14) < 40
                    rsi14_val = float(ind["rsi14"].iloc[i]) if not pd.isna(ind["rsi14"].iloc[i]) else 50.0
                    macd_val = float(ind["macd_line"].iloc[i]) if not pd.isna(ind["macd_line"].iloc[i]) else 0.0
                    macd_sig = float(ind["macd_signal"].iloc[i]) if not pd.isna(ind["macd_signal"].iloc[i]) else 0.0
                    macd_prev_val = float(ind["macd_line"].iloc[i - 1]) if i > 0 and not pd.isna(ind["macd_line"].iloc[i - 1]) else 0.0
                    macd_sig_prev = float(ind["macd_signal"].iloc[i - 1]) if i > 0 and not pd.isna(ind["macd_signal"].iloc[i - 1]) else 0.0
                    macd_cross_dn = (macd_prev_val >= macd_sig_prev and macd_val < macd_sig)

                    if price <= pos["entry_price"] * 0.95:
                        sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                        trades.append(self._make_portfolio_trade(
                            pos, sz, day, price, "HARD_SL_5PCT"))
                        exited = True
                    if not exited and not pos["partial_exit_done"] and price >= pos["entry_price"] * 1.05:
                        half = pos["shares"] // 2
                        if half > 0:
                            trades.append(self._make_portfolio_trade(
                                pos, half, day, price, "PARTIAL_5PCT"))
                            pos["remaining_shares"] = pos["shares"] - half
                            pos["partial_exit_done"] = True
                    if not exited and (macd_cross_dn or rsi14_val < 40):
                        reason = "MACD_CROSS_DOWN" if macd_cross_dn else "RSI14_EXIT"
                        sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                        trades.append(self._make_portfolio_trade(
                            pos, sz, day, price, reason))
                        exited = True

                elif pos["strategy"] == "T":
                    # T exits: 5% SL, +5% partial, upper Keltner remaining
                    ema20_val = float(ind["ema20"].iloc[i])
                    atr14_val = float(ind["atr14"].iloc[i]) if not pd.isna(ind["atr14"].iloc[i]) else 0.0
                    upper_keltner = ema20_val + 2 * atr14_val
                    if price <= pos["entry_price"] * 0.95:
                        sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                        trades.append(self._make_portfolio_trade(
                            pos, sz, day, price, "HARD_SL_5PCT"))
                        exited = True
                    if not exited and not pos["partial_exit_done"] and price >= pos["entry_price"] * 1.05:
                        half = pos["shares"] // 2
                        if half > 0:
                            trades.append(self._make_portfolio_trade(
                                pos, half, day, price, "PARTIAL_5PCT"))
                            pos["remaining_shares"] = pos["shares"] - half
                            pos["partial_exit_done"] = True
                    if not exited and price >= upper_keltner:
                        sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                        trades.append(self._make_portfolio_trade(
                            pos, sz, day, price, "KELTNER_UPPER_EXIT"))
                        exited = True

                elif pos["strategy"] == "U":
                    # U exits: 5% SL, +5% partial, StochRSI overbought exit
                    stk = float(ind["stoch_k"].iloc[i]) if not pd.isna(ind["stoch_k"].iloc[i]) else 50.0
                    std = float(ind["stoch_d"].iloc[i]) if not pd.isna(ind["stoch_d"].iloc[i]) else 50.0
                    stk_prev = float(ind["stoch_k"].iloc[i - 1]) if i > 0 and not pd.isna(ind["stoch_k"].iloc[i - 1]) else 50.0
                    std_prev = float(ind["stoch_d"].iloc[i - 1]) if i > 0 and not pd.isna(ind["stoch_d"].iloc[i - 1]) else 50.0
                    k_cross_dn = (stk_prev >= std_prev and stk < std)
                    if price <= pos["entry_price"] * 0.95:
                        sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                        trades.append(self._make_portfolio_trade(
                            pos, sz, day, price, "HARD_SL_5PCT"))
                        exited = True
                    if not exited and not pos["partial_exit_done"] and price >= pos["entry_price"] * 1.05:
                        half = pos["shares"] // 2
                        if half > 0:
                            trades.append(self._make_portfolio_trade(
                                pos, half, day, price, "PARTIAL_5PCT"))
                            pos["remaining_shares"] = pos["shares"] - half
                            pos["partial_exit_done"] = True
                    if not exited and k_cross_dn and stk > 70 and std > 70:
                        sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                        trades.append(self._make_portfolio_trade(
                            pos, sz, day, price, "STOCHRSI_EXIT"))
                        exited = True

                elif pos["strategy"] == "V":
                    # V exits: 5% hard SL, +5% partial, Close < 10-day low (trailing channel)
                    low_10d = float(ind["low_10d_v"].iloc[i]) if not pd.isna(ind["low_10d_v"].iloc[i]) else 0.0

                    if price <= pos["entry_price"] * 0.95:
                        sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                        trades.append(self._make_portfolio_trade(
                            pos, sz, day, price, "HARD_SL_5PCT"))
                        exited = True
                    if not exited and not pos["partial_exit_done"] and price >= pos["entry_price"] * 1.05:
                        half = pos["shares"] // 2
                        if half > 0:
                            trades.append(self._make_portfolio_trade(
                                pos, half, day, price, "PARTIAL_5PCT"))
                            pos["remaining_shares"] = pos["shares"] - half
                            pos["partial_exit_done"] = True
                    if not exited and low_10d > 0 and price < low_10d:
                        sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                        trades.append(self._make_portfolio_trade(
                            pos, sz, day, price, "DONCHIAN_LOW_EXIT"))
                        exited = True

                if not exited:
                    still_open.append(pos)

            positions = still_open

            # === 2. Collect all entry signals ===
            signals = []
            held_symbols = {p["symbol"] for p in positions}

            for ticker, ind in indicators.items():
                if ticker in held_symbols:
                    continue
                if day not in date_to_idx.get(ticker, {}):
                    continue

                i = date_to_idx[ticker][day]
                price = float(ind["closes"].iloc[i])
                open_price = float(ind["opens"].iloc[i])
                low = float(ind["lows"].iloc[i])
                high = float(ind["highs"].iloc[i])
                rsi2 = float(ind["rsi2"].iloc[i])
                rsi3 = float(ind["rsi3"].iloc[i])
                ibs = float(ind["ibs"].iloc[i])
                is_green = price > open_price

                if pd.isna(rsi2):
                    continue

                # Strategy J entry: close within 0-3% above weekly support, IBS > 0.5, green
                if "J" in strategies:
                    w_support = ind["weekly_support"]
                    w_low_stop = ind["weekly_low_stop"]
                    if w_support is not None and not pd.isna(w_support.iloc[i]):
                        ws = float(w_support.iloc[i])
                        wls = float(w_low_stop.iloc[i]) if w_low_stop is not None and not pd.isna(w_low_stop.iloc[i]) else ws
                        if ws > 0:
                            close_near = ((price - ws) / ws) * 100
                            cci_val = float(ind["cci20"].iloc[i]) if not pd.isna(ind["cci20"].iloc[i]) else 0.0
                            if (close_near >= 0 and close_near <= 3.0
                                    and ibs > 0.5 and is_green
                                    and cci_val > -100):
                                signals.append({
                                    "symbol": ticker,
                                    "strategy": "J",
                                    "price": price,
                                    "ibs": ibs,
                                    "entry_support_j": ws,
                                    "entry_stop_j": wls,
                                })

                # Strategy K entry: RS > 0, prev RSI(2) < 20, today RSI(2) >= 20, Close > EMA(50)
                if "K" in strategies:
                    if ind["rs_spread"] is not None and not pd.isna(ind["rs_spread"].iloc[i]):
                        rs_val = float(ind["rs_spread"].iloc[i])
                        rsi2_prev_k = float(ind["rsi2"].iloc[i - 1]) if i > 0 and not pd.isna(ind["rsi2"].iloc[i - 1]) else 50.0
                        ema50_val = float(ind["ema50"].iloc[i]) if not pd.isna(ind["ema50"].iloc[i]) else 0.0
                        vol_val = float(ind["volume"].iloc[i])
                        vol_avg = float(ind["vol_avg20"].iloc[i]) if not pd.isna(ind["vol_avg20"].iloc[i]) else 0.0
                        if rs_val > 0 and rsi2 >= 20 and rsi2_prev_k < 20 and price > ema50_val and ibs > 0.5 and vol_val > vol_avg:  # TEST: volume filter
                            # Don't duplicate if already signaled by I or J
                            already = any(s["symbol"] == ticker for s in signals)
                            if not already:
                                signals.append({
                                    "symbol": ticker,
                                    "strategy": "K",
                                    "price": price,
                                })

                # Strategy N entry: CCI(20) crosses above 100, Nifty CCI > 0
                if "N" in strategies:
                    cci_val = float(ind["cci20"].iloc[i]) if not pd.isna(ind["cci20"].iloc[i]) else 0.0
                    cci_prev = float(ind["cci20"].iloc[i-1]) if i > 0 and not pd.isna(ind["cci20"].iloc[i-1]) else 0.0
                    nifty_cci = nifty_cci_by_date.get(day, 0.0)
                    if cci_val > 100 and cci_prev <= 100 and is_green:
                        rs_val = float(ind["rs_spread"].iloc[i]) if ind["rs_spread"] is not None and not pd.isna(ind["rs_spread"].iloc[i]) else 0.0
                        already_n = any(s["symbol"] == ticker and s["strategy"] == "N" for s in signals)
                        if not already_n:
                            signals.append({
                                "symbol": ticker,
                                "strategy": "N",
                                "price": price,
                                "rs_spread": rs_val,
                            })

                # Strategy O entry: Price > SMA(200) AND RSI(2) < 5
                if "O" in strategies:
                    sma200_val = float(ind["sma200"].iloc[i]) if not pd.isna(ind["sma200"].iloc[i]) else None
                    if sma200_val is not None and rsi2 < 5 and price > sma200_val:
                        already = any(s["symbol"] == ticker for s in signals)
                        if not already:
                            signals.append({
                                "symbol": ticker,
                                "strategy": "O",
                                "price": price,
                            })

                # Strategy R entry: RSI(14) crosses above 60 AND Price > EMA(200) AND green candle
                if "R" in strategies:
                    rsi14_val = float(ind["rsi14"].iloc[i]) if not pd.isna(ind["rsi14"].iloc[i]) else 50.0
                    rsi14_prev = float(ind["rsi14"].iloc[i - 1]) if i > 0 and not pd.isna(ind["rsi14"].iloc[i - 1]) else 50.0
                    ema200_val = float(ind["ema200"].iloc[i])
                    if rsi14_prev <= 60 and rsi14_val > 60 and price > ema200_val and is_green:
                        already = any(s["symbol"] == ticker for s in signals)
                        if not already:
                            signals.append({
                                "symbol": ticker,
                                "strategy": "R",
                                "price": price,
                            })

                # Strategy S entry: MACD crosses signal above 0 AND RSI(14) > 50 AND green candle
                if "S" in strategies:
                    rsi14_val = float(ind["rsi14"].iloc[i]) if not pd.isna(ind["rsi14"].iloc[i]) else 50.0
                    macd_val = float(ind["macd_line"].iloc[i]) if not pd.isna(ind["macd_line"].iloc[i]) else 0.0
                    macd_sig_val = float(ind["macd_signal"].iloc[i]) if not pd.isna(ind["macd_signal"].iloc[i]) else 0.0
                    macd_prev_val = float(ind["macd_line"].iloc[i - 1]) if i > 0 and not pd.isna(ind["macd_line"].iloc[i - 1]) else 0.0
                    macd_sig_prev = float(ind["macd_signal"].iloc[i - 1]) if i > 0 and not pd.isna(ind["macd_signal"].iloc[i - 1]) else 0.0
                    macd_cross_up = (macd_prev_val <= macd_sig_prev and macd_val > macd_sig_val)
                    if macd_cross_up and macd_val > 0 and rsi14_val > 50 and is_green:
                        already = any(s["symbol"] == ticker for s in signals)
                        if not already:
                            signals.append({
                                "symbol": ticker,
                                "strategy": "S",
                                "price": price,
                            })

                # Strategy T entry: Price near EMA(20) (within 1%) AND was at upper Keltner in last 10 bars AND green
                if "T" in strategies:
                    ema20_val = float(ind["ema20"].iloc[i])
                    atr14_val = float(ind["atr14"].iloc[i]) if not pd.isna(ind["atr14"].iloc[i]) else 0.0
                    if atr14_val > 0:
                        near_ema20 = abs(price - ema20_val) / ema20_val <= 0.01
                        was_at_upper = False
                        for lb_j in range(max(0, i - 10), i):
                            past_high = float(ind["highs"].iloc[lb_j])
                            past_ema20 = float(ind["ema20"].iloc[lb_j])
                            past_atr14 = float(ind["atr14"].iloc[lb_j]) if not pd.isna(ind["atr14"].iloc[lb_j]) else 0.0
                            if past_high >= past_ema20 + 2 * past_atr14:
                                was_at_upper = True
                                break
                        if near_ema20 and was_at_upper and is_green:
                            already = any(s["symbol"] == ticker for s in signals)
                            if not already:
                                signals.append({
                                    "symbol": ticker,
                                    "strategy": "T",
                                    "price": price,
                                })

                # Strategy U entry: StochRSI %K crosses above %D, both < 30 AND Price > EMA(200)
                if "U" in strategies:
                    stk = float(ind["stoch_k"].iloc[i]) if not pd.isna(ind["stoch_k"].iloc[i]) else 50.0
                    std = float(ind["stoch_d"].iloc[i]) if not pd.isna(ind["stoch_d"].iloc[i]) else 50.0
                    stk_prev = float(ind["stoch_k"].iloc[i - 1]) if i > 0 and not pd.isna(ind["stoch_k"].iloc[i - 1]) else 50.0
                    std_prev = float(ind["stoch_d"].iloc[i - 1]) if i > 0 and not pd.isna(ind["stoch_d"].iloc[i - 1]) else 50.0
                    ema200_val = float(ind["ema200"].iloc[i])
                    k_cross_up = (stk_prev <= std_prev and stk > std)
                    if k_cross_up and stk < 30 and std < 30 and price > ema200_val:
                        already = any(s["symbol"] == ticker for s in signals)
                        if not already:
                            signals.append({
                                "symbol": ticker,
                                "strategy": "U",
                                "price": price,
                            })

                # Strategy V entry: Close > 20-day high AND Vol > 1.5x avg(20) AND Price > EMA(200) AND green
                if "V" in strategies:
                    high_20d_val = float(ind["high_20d_v"].iloc[i]) if not pd.isna(ind["high_20d_v"].iloc[i]) else None
                    ema200_val = float(ind["ema200"].iloc[i])
                    vol_val = float(ind["volume"].iloc[i])
                    vol_avg = float(ind["vol_avg20"].iloc[i]) if not pd.isna(ind["vol_avg20"].iloc[i]) else 0.0
                    if (high_20d_val is not None and price > high_20d_val
                            and vol_avg > 0 and vol_val > 1.5 * vol_avg
                            and price > ema200_val and is_green):
                        already = any(s["symbol"] == ticker for s in signals)
                        if not already:
                            signals.append({
                                "symbol": ticker,
                                "strategy": "V",
                                "price": price,
                            })

            total_signals += len(signals)

            # === 3. Allocate capital (max entries_per_day) ===
            available_slots = MAX_POSITIONS - len(positions)
            max_today = min(entries_per_day, available_slots)
            if signals and max_today > 0:
                random.shuffle(signals)
                taken = min(len(signals), max_today)
                missed_signals += len(signals) - taken
                signals = signals[:taken]

                for sig in signals:
                    shares = int(PER_STOCK // sig["price"])
                    if shares > 0:
                        pos = {
                            "symbol": sig["symbol"],
                            "strategy": sig["strategy"],
                            "entry_date": day,
                            "entry_price": sig["price"],
                            "shares": shares,
                            "remaining_shares": shares,
                            "partial_exit_done": False,
                        }
                        if sig["strategy"] == "J":
                            pos["entry_support_j"] = sig["entry_support_j"]
                            pos["entry_stop_j"] = sig["entry_stop_j"]
                            pos["nifty_at_entry"] = nifty_close_by_date.get(day, 0.0)
                        elif sig["strategy"] == "K":
                            pos["entry_stop_j"] = 0
                            pos["entry_support_j"] = 0
                            pos["nifty_at_entry"] = nifty_close_by_date.get(day, 0.0)
                        elif sig["strategy"] == "N":
                            pos["nifty_at_entry"] = nifty_close_by_date.get(day, 0.0)
                        positions.append(pos)
            elif signals and available_slots <= 0:
                missed_signals += len(signals)

            if len(positions) > max_positions_used:
                max_positions_used = len(positions)
            positions_over_time.append({
                "date": day.isoformat(),
                "positions": len(positions)
            })

        # --- Close remaining positions at end ---
        last_day = all_dates[-1]
        for pos in positions:
            ticker = pos["symbol"]
            if last_day in date_to_idx.get(ticker, {}):
                i = date_to_idx[ticker][last_day]
                price = float(indicators[ticker]["closes"].iloc[i])
            else:
                # Fallback: use the most recent available close price
                ticker_dates = date_to_idx.get(ticker, {})
                recent = [d for d in ticker_dates if d <= last_day]
                if recent:
                    i = ticker_dates[max(recent)]
                    price = float(indicators[ticker]["closes"].iloc[i])
                else:
                    price = pos["entry_price"]
            sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
            trades.append(self._make_portfolio_trade(
                pos, sz, last_day, price, "BACKTEST_END"))

        # --- Build summary ---
        # Prorate return based on max capital actually deployed
        effective_capital = max_positions_used * PER_STOCK if max_positions_used > 0 else TOTAL_CAPITAL
        summary = self._calculate_summary(trades, effective_capital)

        strat_labels = {"J": "J(3:15)", "K": "K(3:15)", "N": "N(CCI)", "O": "O(Connors RSI2)", "R": "R(RSI Mom)", "S": "S(MACD+RSI)", "T": "T(Keltner)", "U": "U(StochRSI)", "V": "V(Donchian)"}
        strat_label = " + ".join(strat_labels.get(s, s) for s in strategies)

        return {
            "strategy": "Portfolio",
            "strategies": strategies,
            "strategies_label": strat_label,
            "exit_target": None,
            "start_date": all_dates[0].isoformat(),
            "end_date": all_dates[-1].isoformat(),
            "trading_days": len(all_dates),
            "capital": TOTAL_CAPITAL,
            "capital_lakhs": capital_lakhs,
            "effective_capital_lakhs": round(effective_capital / 100000, 1),
            "max_positions": MAX_POSITIONS,
            "trades": trades,
            "summary": summary,
            "total_signals": total_signals,
            "missed_signals": missed_signals,
            "max_positions_used": max_positions_used,
            "universe": universe,
        }

    def _make_portfolio_trade(self, pos, shares, exit_date, exit_price, reason):
        """Make a trade dict for portfolio backtest (includes symbol & strategy)."""
        trade = self._make_trade(
            pos["entry_date"], pos["entry_price"], shares,
            exit_date, exit_price, reason)
        trade["symbol"] = pos["symbol"]
        trade["strategy"] = pos["strategy"]
        return trade

    def run_all_stocks(self, period_days, capital=100000, progress_callback=None, universe=50, end_date=None):
        """Run all strategy variants across stocks (Nifty 50 or 100)."""
        from data.momentum_engine import NIFTY_50_TICKERS

        if universe <= 50:
            tickers = NIFTY_50_TICKERS
        else:
            tickers = NIFTY_50_TICKERS + NIFTY_NEXT50_TICKERS

        results = []
        total = len(tickers)

        for idx, ticker in enumerate(tickers):
            if progress_callback:
                progress_callback(idx + 1, total, ticker)

            # Fetch data ONCE per stock
            nse_symbol = f"{ticker}.NS"
            _end = end_date or datetime.now()
            daily_start = _end - timedelta(days=period_days + 500)
            try:
                daily = yf.Ticker(nse_symbol).history(start=daily_start, end=_end)
            except Exception:
                daily = pd.DataFrame()

            row = {"symbol": ticker, "results": {}}

            for strat, exit_tgt, label in BATCH_VARIANTS:
                result = self.run(ticker, period_days, strategy=strat,
                                  capital=capital, exit_target=exit_tgt,
                                  _daily_data=daily, end_date=_end)
                if "error" not in result:
                    row["results"][label] = {
                        "total_pnl": result["summary"]["total_pnl"],
                        "return_pct": result["summary"]["total_return_pct"],
                        "win_rate": result["summary"]["win_rate"],
                        "trades": result["summary"]["total_trades"],
                        "profit_factor": result["summary"]["profit_factor"],
                    }
                else:
                    row["results"][label] = None

            results.append(row)

        columns = [v[2] for v in BATCH_VARIANTS]
        return {"columns": columns, "rows": results}

