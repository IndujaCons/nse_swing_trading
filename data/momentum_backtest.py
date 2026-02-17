"""
Momentum Scanner Backtest Engine
Daily-only EOD backtest with three strategy variants (B, C, D).

Strategy B & C — Momentum (entry: RSI(2) >= 75):
  Exit B: Close < EMA(10) / EMA(50) / EMA(200)
  Exit C: RSI(2) < 30 / Close < EMA(50) / EMA(200)

Strategy D — RSI Reversal (mean reversion):
  Entry: Close > EMA(50) AND Close > EMA(200)
         AND RSI(2) was < 10 yesterday AND RSI(2) > 10 today
  Exit (priority): Close > EMA(5) / 3% stop-loss / Close < EMA(50) / Close < EMA(200)

Strategy E — EMA Pullback (scale-out):
  Entry: EMA(50) > EMA(200) (uptrend)
         AND Close was < EMA(20) yesterday AND Close > EMA(20) today
  Exit 1: Close >= Entry+5% → sell 50%
  Exit 2: Close < EMA(20) → sell all remaining
  Global: Close < EMA(50) → sell all remaining

Strategy F — NR7 Squeeze (scale-out):
  NR7 = today's High-Low is the smallest of last 7 days
  Entry: NR7 day AND RSI(2) < 30 AND Close > EMA(50) AND EMA(50) > EMA(200)
  Exit 1: Close >= Entry+5% → sell 50%
  Exit 2: Close > EMA(10) → sell remaining
  Stop: 3% from entry OR Close < EMA(50)

Strategy G — IBS Extreme (scale-out):
  IBS = (Close - Low) / (High - Low)
  Entry: IBS < 0.2 AND Close > EMA(50) AND EMA(50) > EMA(200)
  Exit 1: Close >= Entry+5% → sell 50%
  Exit 2: IBS > 0.8 → sell remaining
  Stop: 3% from entry OR Close < EMA(50)

Strategy H — Consecutive Down Days (scale-out):
  Entry: 4 consecutive lower closes AND Close > EMA(50) AND EMA(50) > EMA(200)
  Exit 1: Close >= Entry+5% → sell 50%
  Exit 2: First up close (Close > previous Close) → sell remaining
  Stop: 3% from entry OR Close < EMA(50)

Strategy I — 6-Month Support Bounce:
  Support = Lowest Low of last 120 trading days
  Entry: Today's Low within 1% of Support AND Close > Support AND RS Rating > 50
  Exit (default): Close >= Entry+5% → sell all
  Exit (5%+8%):   50% at +5%, remaining 50% at +8%
  Hard stop: Close < Support → sell all

Strategy J — Weekly Close Support Bounce:
  Entry support: Lowest weekly CLOSE of last 26 weeks (~6 months)
  Stop support: Lowest daily LOW of 120 days (same as Strategy I)
  Entry: Daily low within 1% of weekly close support AND close > support AND IBS > 0.3 AND RS > 50
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
    ("I", "gtt", "I: Support Bounce (GTT)"),
    ("I", "close", "I: Support Bounce (3:15)"),
    ("I", "2wlow", "I: Support Bounce (2W Low)"),
    ("J", "gtt", "J: Weekly Support (GTT)"),
    ("J", "close", "J: Weekly Support (3:15)"),
    ("J", "2wlow", "J: Weekly Support (2W Low)"),
    ("E", None, "E: EMA Pullback"),
    ("G", None, "G: IBS Extreme"),
    ("D", "pct5", "D: RSI Rev +5%"),
    ("K", "gtt", "K: RS Trending (GTT)"),
    ("K", "close", "K: RS Trending (3:15)"),
    ("K", "2wlow", "K: RS Trending (2W Low)"),
    ("L", None, "L: Pure RS"),
    ("M", None, "M: EMA 10/20"),
    ("P", None, "P: AVWAP"),
    ("Q", None, "Q: Fib 61.8%"),
]


class MomentumBacktester:
    """Single-stock daily backtest for momentum EOD strategy (B, C, D, E, F, G, H, M)."""

    def _calculate_avwap_series(self, highs: pd.Series, lows: pd.Series,
                                closes: pd.Series, volumes: pd.Series,
                                lookback: int = 120):
        """Calculate AVWAP anchored from rolling 6-month swing low date.
        Returns: (avwap, anchor_age) — anchor_age = bars since swing low."""
        n = len(closes)
        avwap = pd.Series(index=closes.index, dtype=float)
        anchor_age = pd.Series(index=closes.index, dtype=float)
        typical = (highs + lows + closes) / 3.0

        for i in range(lookback, n):
            # Find the index position of the minimum low in past `lookback` bars
            window = lows.iloc[i - lookback + 1:i + 1]
            anchor_pos = i - lookback + 1 + window.values.argmin()

            # Calculate AVWAP from anchor to current bar
            tp_slice = typical.values[anchor_pos:i + 1]
            vol_slice = volumes.values[anchor_pos:i + 1]
            cum_vol = vol_slice.sum()
            if cum_vol > 0:
                avwap.iloc[i] = (tp_slice * vol_slice).sum() / cum_vol
            else:
                avwap.iloc[i] = closes.iloc[i]

            anchor_age.iloc[i] = i - anchor_pos

        return avwap, anchor_age

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

    def run(self, symbol: str, period_days: int, strategy: str = "B",
            capital: int = 100000, exit_target: str = None,
            _daily_data=None) -> Dict:
        """
        Run daily-only backtest for a single symbol.

        Args:
            symbol: NSE ticker (e.g. "ICICIBANK")
            period_days: backtest lookback in calendar days (30, 90, 180, 365)
            strategy: "B" (EMA10 exit), "C" (RSI2 < 30 exit), or "D" (RSI reversal)
            capital: starting capital in INR
            exit_target: profit target for Strategy D — "5","8","10","20" (EMA)
                         or "pct5","pct6" (percentage from entry)

        Returns:
            Dict with trades list, summary stats, and metadata.
        """
        config = load_config()
        rsi2_entry = config.get("momentum_rsi2_threshold", 75)

        end_date = datetime.now()

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

        # Support levels for Strategy I (exclude current bar via shift)
        support_6m = lows.rolling(window=120, min_periods=120).min().shift(1)   # 6-month
        support_1y = lows.rolling(window=252, min_periods=252).min().shift(1)   # 1-year

        # 2-week low (10 trading days), excluding current bar
        two_week_low = lows.rolling(window=10, min_periods=10).min().shift(1)

        # Volume for daily use
        volumes = daily["Volume"].astype(float)

        # RS Rating filter for Strategy I, J, and K
        # For I/J: weighted RS outperformance (commented out in entry logic)
        # For K: simple 3-month return vs Nifty (RS > 0 = outperforming)
        rs_filter_series = None  # True/False: outperforming Nifty (for I, J)
        rs_spread_series = None  # Stock 3M return minus Nifty 3M return (for K)
        rs_spread_6m_series = None  # Stock 6M return minus Nifty 6M return (for L)
        if strategy in ("I", "J", "K", "L", "P", "Q"):
            try:
                nifty_data = yf.Ticker("^NSEI").history(
                    start=daily.index[0], end=end_date)
                if not nifty_data.empty:
                    nifty_close = nifty_data["Close"].reindex(daily.index, method="ffill")

                    if strategy in ("I", "J"):
                        def _weighted_rs_score(stock_closes):
                            r3m = stock_closes.pct_change(periods=63)
                            r6m = stock_closes.pct_change(periods=126)
                            r9m = stock_closes.pct_change(periods=189)
                            r12m = stock_closes.pct_change(periods=252)
                            return 0.4 * r3m + 0.3 * r6m + 0.2 * r9m + 0.1 * r12m
                        stock_ws = _weighted_rs_score(closes)
                        nifty_ws = _weighted_rs_score(nifty_close)
                        rs_filter_series = stock_ws > nifty_ws

                    if strategy == "K":
                        # Simple 3-month return spread: stock vs Nifty
                        stock_3m = closes.pct_change(periods=63)
                        nifty_3m = nifty_close.pct_change(periods=63)
                        rs_spread_series = stock_3m - nifty_3m  # >0 = outperforming

                    if strategy == "L":
                        # 6-month return spread: stock vs Nifty
                        stock_6m = closes.pct_change(periods=126)
                        nifty_6m = nifty_close.pct_change(periods=126)
                        rs_spread_6m_series = stock_6m - nifty_6m
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


        # Resolve Strategy D exit target
        target_str = str(exit_target) if exit_target else "5"
        use_pct_target = target_str.startswith("pct")
        if use_pct_target:
            pct_target = int(target_str.replace("pct", "")) / 100.0
            d_exit_period = None
            target_label = f"+{int(pct_target*100)}%"
        else:
            d_exit_period = int(target_str) if target_str in ("5","8","10","20") else 5
            pct_target = None
            target_label = f"EMA{d_exit_period}"

        # Pre-compute indicator series over entire dataset
        rsi2_series = self._calculate_rsi_series(closes, 2)
        rsi3_series = self._calculate_rsi_series(closes, 3)
        ema_exit_series = (closes.ewm(span=d_exit_period, adjust=False).mean()
                           if d_exit_period else None)
        ema5_series = closes.ewm(span=5, adjust=False).mean()
        ema8_series = closes.ewm(span=8, adjust=False).mean()
        ema10_series = closes.ewm(span=10, adjust=False).mean()
        ema20_series = closes.ewm(span=20, adjust=False).mean()
        ema50_series = closes.ewm(span=50, adjust=False).mean()
        ema200_series = closes.ewm(span=200, adjust=False).mean()
        vol_avg20_series = volumes.rolling(window=20, min_periods=20).mean()

        # AVWAP anchored from 6-month swing low (for Strategy P)
        avwap_series, avwap_age_series = self._calculate_avwap_series(highs, lows, closes, volumes, lookback=120)

        # Fibonacci levels from 6-month high/low (for Strategy Q)
        fib_high_6m = highs.rolling(window=120, min_periods=120).max()
        fib_low_6m = lows.rolling(window=120, min_periods=120).min()
        fib_range = fib_high_6m - fib_low_6m
        fib_618 = fib_high_6m - 0.618 * fib_range   # 61.8% retracement
        fib_786 = fib_high_6m - 0.70 * fib_range   # 70% retracement (stop)
        fib_ext_121 = fib_high_6m + 0.21 * fib_range  # 121% extension (target)

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
        m_partial_done = False  # Strategy M: partial exit tracking
        m_remaining = 0  # Strategy M: remaining shares after partial
        p_partial_done = False  # Strategy P: partial exit tracking
        p_remaining = 0  # Strategy P: remaining shares after partial
        q_partial_done = False  # Strategy Q: partial exit tracking
        q_remaining = 0  # Strategy Q: remaining shares after partial
        q_fib_stop = 0.0  # Strategy Q: 70% fib level at entry
        q_fib_target = 0.0  # Strategy Q: 121% extension at entry
        trades = []
        was_below_exit_ema = False
        days_below_ema20 = 0  # For Strategy E: consecutive days below EMA(20)
        entry_support = 0.0  # For Strategy I: support level locked at entry
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
            ema_exit = float(ema_exit_series.iloc[i]) if ema_exit_series is not None else None
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
            tw_low = float(two_week_low.iloc[i]) if not pd.isna(two_week_low.iloc[i]) else None
            day = daily.index[i].date()

            if pd.isna(rsi2):
                continue

            # Previous bar values for crossover detection
            prev_close = float(closes.iloc[i - 1]) if i > 0 else None
            prev_ema20 = float(ema20_series.iloc[i - 1]) if i > 0 else None

            if strategy == "I":
                # Strategy I: 6-Month Support Bounce
                # Support = lowest Low of last 120 trading days
                # Entry: Low within 1% of support AND Close > support
                #   + Bounce confirmation: IBS > 0.3 (buyers showed up)
                # Check proximity to either 6M or 1Y support
                near_6m = (sup_6m is not None
                           and low <= sup_6m * 1.01
                           and low >= sup_6m)
                near_1y = (sup_1y is not None
                           and low <= sup_1y * 1.01
                           and low >= sup_1y)

                if not near_6m and not near_1y:
                    # Not near any support — skip entry check but still check exits
                    support = sup_6m if sup_6m is not None else sup_1y
                else:
                    # Use whichever support level triggered
                    support = sup_6m if near_6m else sup_1y

                if support is None:
                    continue

                # rs_ok = rs_filter_series is None or (rs_filter_series.iloc[i] if not pd.isna(rs_filter_series.iloc[i]) else False)
                entry_signal = ((near_6m or near_1y)
                                and price > support
                                and ibs > 0.5
                                and price > open_price)
                                # and rs_ok)  # RS Rating > 50

                # 50% at +5%, remaining 50% at +10%
                i_mode = str(exit_target) if exit_target else "gtt"
                t1_price = entry_price * 1.05 if in_position else 0
                t2_price = entry_price * 1.10 if in_position else 0
                is_red = price < open_price
                below_2w = tw_low is not None and price < tw_low

                def _i_stop(sz):
                    """Check stop for current mode, return True if stopped."""
                    if i_mode == "gtt":
                        if low <= entry_support:
                            trades.append(self._make_trade(
                                entry_date, entry_price, sz, day,
                                entry_support, "GTT_SUPPORT_BREAK"))
                            return True
                    elif i_mode == "2wlow":
                        if price < entry_support:
                            trades.append(self._make_trade(
                                entry_date, entry_price, sz, day, price,
                                "SUPPORT_BREAK"))
                            return True
                        if is_red and below_2w:
                            trades.append(self._make_trade(
                                entry_date, entry_price, sz, day, price,
                                "2WLOW_EXIT"))
                            return True
                    else:  # close
                        if price < entry_support:
                            trades.append(self._make_trade(
                                entry_date, entry_price, sz, day, price,
                                "SUPPORT_BREAK"))
                            return True
                    return False

                if in_position and not partial_exit_done:
                    if _i_stop(shares):
                        in_position = False
                        partial_exit_done = False
                        continue
                    if i_mode == "gtt":
                        if high >= t1_price:
                            half = shares // 2
                            trades.append(self._make_trade(
                                entry_date, entry_price, half, day,
                                t1_price, "GTT_5PCT_PARTIAL"))
                            remaining_shares = shares - half
                            partial_exit_done = True
                            if high >= t2_price:
                                trades.append(self._make_trade(
                                    entry_date, entry_price, remaining_shares,
                                    day, t2_price, "GTT_10PCT"))
                                in_position = False
                                partial_exit_done = False
                            continue
                    else:  # close or 2wlow — targets on Close
                        if price >= t1_price:
                            half = shares // 2
                            trades.append(self._make_trade(
                                entry_date, entry_price, half, day, price,
                                "5PCT_PARTIAL"))
                            remaining_shares = shares - half
                            partial_exit_done = True
                            if price >= t2_price:
                                trades.append(self._make_trade(
                                    entry_date, entry_price, remaining_shares,
                                    day, price, "10PCT"))
                                in_position = False
                                partial_exit_done = False
                            continue

                if in_position and partial_exit_done:
                    if _i_stop(remaining_shares):
                        in_position = False
                        partial_exit_done = False
                        continue
                    if i_mode == "gtt":
                        if high >= t2_price:
                            trades.append(self._make_trade(
                                entry_date, entry_price, remaining_shares,
                                day, t2_price, "GTT_10PCT"))
                            in_position = False
                            partial_exit_done = False
                            continue
                    else:
                        if price >= t2_price:
                            trades.append(self._make_trade(
                                entry_date, entry_price, remaining_shares,
                                day, price, "10PCT"))
                            in_position = False
                            partial_exit_done = False
                            continue

                # Entry (only when fully out)
                if entry_signal and not in_position:
                    shares = int(capital // price)
                    if shares > 0:
                        entry_price = price
                        entry_date = day
                        entry_support = support  # Lock support at entry
                        in_position = True
                        partial_exit_done = False
                continue

            elif strategy == "J":
                # Strategy J: Weekly Close Support Bounce (scale-out)
                # Entry: daily low within 1% of 26-week min weekly CLOSE
                # Stop: close < 26-week min weekly LOW
                # Exit 1: +5% → sell 50%   Exit 2: +10% → sell remaining

                w_support = float(weekly_support_series.iloc[i]) if weekly_support_series is not None and not pd.isna(weekly_support_series.iloc[i]) else None
                w_low_stop = float(weekly_low_stop_series.iloc[i]) if weekly_low_stop_series is not None and not pd.isna(weekly_low_stop_series.iloc[i]) else None

                entry_signal = False
                if w_support is not None and not in_position:
                    close_near = ((price - w_support) / w_support) * 100 if w_support > 0 else 999
                    entry_signal = (close_near >= 0        # close above support
                                    and close_near <= 3.0  # close within +3% of support
                                    and ibs > 0.5          # bounce
                                    and price > open_price) # green candle

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

                j_mode = str(exit_target) if exit_target else "gtt"
                is_red = price < open_price
                below_2w = tw_low is not None and price < tw_low

                def _j_stop(sz):
                    """Check stop for current mode, return True if stopped.
                    Skip support break if Nifty fell same or more since entry."""
                    # Nifty drop shield
                    nifty_shields = False
                    if nifty_at_entry > 0 and nifty_close is not None:
                        nifty_now = float(nifty_close.iloc[i])
                        nifty_pct = (nifty_now - nifty_at_entry) / nifty_at_entry
                        stock_pct = (price - entry_price) / entry_price
                        if nifty_pct <= stock_pct and nifty_pct < 0:
                            nifty_shields = True

                    if nifty_shields:
                        return False  # Ignore stop — market-wide fall

                    if j_mode == "gtt":
                        if low <= entry_stop_j:
                            _j_trade(entry_price, sz, day, entry_stop_j, "GTT_SUPPORT_BREAK")
                            return True
                    elif j_mode == "2wlow":
                        if price < entry_stop_j:
                            _j_trade(entry_price, sz, day, price, "SUPPORT_BREAK")
                            return True
                        if is_red and below_2w:
                            _j_trade(entry_price, sz, day, price, "2WLOW_EXIT")
                            return True
                    else:  # close
                        if price < entry_stop_j:
                            _j_trade(entry_price, sz, day, price, "SUPPORT_BREAK")
                            return True
                    return False

                if in_position and not partial_exit_done:
                    if _j_stop(shares):
                        in_position = False
                        partial_exit_done = False
                        continue
                    if j_mode == "gtt":
                        if high >= t1_price:
                            half = shares // 2
                            _j_trade(entry_price, half, day, t1_price, "GTT_5PCT_PARTIAL")
                            remaining_shares = shares - half
                            partial_exit_done = True
                            if high >= t2_price:
                                _j_trade(entry_price, remaining_shares, day, t2_price, "GTT_10PCT")
                                in_position = False
                                partial_exit_done = False
                            continue
                    else:  # close or 2wlow — targets on Close
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
                    if j_mode == "gtt":
                        if high >= t2_price:
                            _j_trade(entry_price, remaining_shares, day, t2_price, "GTT_10PCT")
                            in_position = False
                            partial_exit_done = False
                            continue
                    else:
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
                        entry_stop_j = w_low_stop if w_low_stop is not None else w_support  # Stop at 26-week weekly low
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

                k_mode = str(exit_target) if exit_target else "gtt"
                is_red = price < open_price
                below_2w = tw_low is not None and price < tw_low

                def _k_stop(sz):
                    """Check stop for current mode, return True if stopped.
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

                    if k_mode == "gtt":
                        if low <= stop_3pct:
                            trades.append(self._make_trade(
                                entry_date, entry_price, sz, day,
                                stop_3pct, "GTT_STOP_3PCT"))
                            return True
                    elif k_mode == "2wlow":
                        if price < stop_3pct:
                            trades.append(self._make_trade(
                                entry_date, entry_price, sz, day,
                                price, "STOP_3PCT"))
                            return True
                        if is_red and below_2w:
                            trades.append(self._make_trade(
                                entry_date, entry_price, sz, day,
                                price, "2WLOW_EXIT"))
                            return True
                    else:  # close
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
                    if k_mode == "gtt":
                        if high >= t1_price:
                            half = shares // 2
                            trades.append(self._make_trade(
                                entry_date, entry_price, half, day,
                                t1_price, "GTT_5PCT_PARTIAL"))
                            remaining_shares = shares - half
                            partial_exit_done = True
                            continue
                    else:  # close or 2wlow — targets on Close
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

            elif strategy == "L":
                # Strategy L: Pure RS System (6-month) with buffer zone
                # Entry: Yesterday RS < +2%, Today RS > +2%
                # Exit: RS drops below -2% OR Close < EMA(50)
                # Min hold: 5 days (no RS exit before day 5)

                RS_ENTRY_THRESH = 0.02   # +2%
                RS_EXIT_THRESH = -0.02   # -2%
                MIN_HOLD_DAYS = 5

                rs_val = float(rs_spread_6m_series.iloc[i]) if (
                    rs_spread_6m_series is not None
                    and not pd.isna(rs_spread_6m_series.iloc[i])) else 0.0
                rs_prev = float(rs_spread_6m_series.iloc[i - 1]) if (
                    i > 0 and rs_spread_6m_series is not None
                    and not pd.isna(rs_spread_6m_series.iloc[i - 1])) else 0.0

                if in_position:
                    hold_days = (day - entry_date).days
                    if price < ema50:
                        trades.append(self._make_trade(
                            entry_date, entry_price, shares, day,
                            price, "EMA50_EXIT"))
                        in_position = False
                        continue
                    if hold_days >= MIN_HOLD_DAYS and rs_val < RS_EXIT_THRESH:
                        trades.append(self._make_trade(
                            entry_date, entry_price, shares, day,
                            price, "RS_NEGATIVE"))
                        in_position = False
                        continue

                if not in_position and rs_prev < RS_ENTRY_THRESH and rs_val > RS_ENTRY_THRESH:
                    shares = int(capital // price)
                    if shares > 0:
                        entry_price = price
                        entry_date = day
                        in_position = True
                continue

            elif strategy == "M":
                # Strategy M: EMA Crossover (10/20)
                # Entry: EMA(10) crosses above EMA(20) AND Close > EMA(50)
                #        AND EMA(20) rising
                # Partial: Close >= Entry + 5% → sell 50%
                # Exit: EMA(10) crosses below EMA(20) → sell remaining

                prev_ema10 = float(ema10_series.iloc[i - 1]) if i > 0 else None
                prev_ema20_val = float(ema20_series.iloc[i - 1]) if i > 0 else None

                if in_position:
                    # Partial: +5% → sell 50%
                    if not m_partial_done and price >= entry_price * 1.05:
                        partial_shares = shares // 2
                        if partial_shares > 0:
                            trades.append(self._make_trade(
                                entry_date, entry_price, partial_shares, day,
                                price, "PARTIAL_5PCT"))
                            m_remaining = shares - partial_shares
                            m_partial_done = True

                    # Exit: EMA(10) crosses below EMA(20)
                    if (prev_ema10 is not None and prev_ema20_val is not None
                            and prev_ema10 >= prev_ema20_val and ema10 < ema20):
                        sz = m_remaining if m_partial_done else shares
                        trades.append(self._make_trade(
                            entry_date, entry_price, sz, day,
                            price, "EMA10_BELOW_EMA20"))
                        in_position = False
                        m_partial_done = False
                        continue

                if not in_position:
                    ema20_rising = (prev_ema20_val is not None and ema20 > prev_ema20_val)
                    if (prev_ema10 is not None and prev_ema20_val is not None
                            and prev_ema10 < prev_ema20_val and ema10 > ema20
                            and price > ema50 and ema20_rising):
                        shares = int(capital // price)
                        if shares > 0:
                            entry_price = price
                            entry_date = day
                            in_position = True
                            m_partial_done = False
                            m_remaining = shares
                continue

            elif strategy == "P":
                # Strategy P: AVWAP from Swing Low Bounce
                # Anchor: AVWAP from 6-month swing low date
                # Entry: Price within 1% of AVWAP from above AND Close > AVWAP
                #        AND IBS > 0.3 AND Close > Open
                # Exit 1: Close >= Entry + 5% → sell 50%
                # Exit 2: Close < AVWAP → sell remaining
                # Stop: Close < Entry - 3%

                avwap = float(avwap_series.iloc[i]) if not pd.isna(avwap_series.iloc[i]) else None
                ibs = float(ibs_series.iloc[i])
                open_price = float(opens.iloc[i])

                if in_position and avwap is not None:
                    # Stop: 3% below entry
                    if price < entry_price * 0.97:
                        sz = p_remaining if p_partial_done else shares
                        trades.append(self._make_trade(
                            entry_date, entry_price, sz, day,
                            price, "STOP_3PCT"))
                        in_position = False
                        p_partial_done = False
                        continue

                    # Partial: +5% → sell 50%
                    if not p_partial_done and price >= entry_price * 1.05:
                        partial_shares = shares // 2
                        if partial_shares > 0:
                            trades.append(self._make_trade(
                                entry_date, entry_price, partial_shares, day,
                                price, "PARTIAL_5PCT"))
                            p_remaining = shares - partial_shares
                            p_partial_done = True

                    # Exit remaining 50%: Close < 3-day low (after partial)
                    # Skip if Nifty itself is weak (Nifty close < Nifty 3-day low close)
                    if p_partial_done:
                        three_day_low = float(lows.iloc[max(0, i-3):i].min()) if i >= 3 else float(lows.iloc[i])
                        nifty_weak = False
                        if nifty_close is not None and i >= 3:
                            nifty_today = float(nifty_close.iloc[i])
                            nifty_3day_low_close = float(nifty_close.iloc[max(0, i-3):i].min())
                            nifty_weak = nifty_today < nifty_3day_low_close
                        if not nifty_weak and price < three_day_low:
                            if p_remaining > 0:
                                trades.append(self._make_trade(
                                    entry_date, entry_price, p_remaining, day,
                                    price, "BELOW_3DAY_LOW"))
                            in_position = False
                            p_partial_done = False
                            continue


                if not in_position and avwap is not None:
                    within_1pct = (price <= avwap * 1.01)
                    if (price > avwap and within_1pct
                            and ibs > 0.5 and price > open_price):
                        shares = int(capital // price)
                        if shares > 0:
                            entry_price = price
                            entry_date = day
                            in_position = True
                            p_partial_done = False
                            p_remaining = shares
                continue

            elif strategy == "Q":
                # Strategy Q: Fibonacci Retracement Bounce (61.8%)
                # Entry: Price within 1% of 61.8% fib from above, Close > 61.8%,
                #        IBS > 0.3, Close > Open
                # Partial: Close >= Entry + 5% → sell 50%
                # Target: 121% extension → sell remaining
                # Stop: Close < 70% retracement

                f618 = float(fib_618.iloc[i]) if not pd.isna(fib_618.iloc[i]) else None
                f786 = float(fib_786.iloc[i]) if not pd.isna(fib_786.iloc[i]) else None
                f121 = float(fib_ext_121.iloc[i]) if not pd.isna(fib_ext_121.iloc[i]) else None
                ibs = float(ibs_series.iloc[i])
                open_price = float(opens.iloc[i])

                if in_position:
                    # Stop: Close < 70% fib level (locked at entry)
                    if price < q_fib_stop:
                        sz = q_remaining if q_partial_done else shares
                        trades.append(self._make_trade(
                            entry_date, entry_price, sz, day,
                            price, "FIB_STOP_70"))
                        in_position = False
                        q_partial_done = False
                        continue

                    # 3-day low exit before partial (skip if Nifty weak)
                    if not q_partial_done and i >= 3:
                        three_day_low = float(lows.iloc[max(0, i-3):i].min())
                        nifty_weak = False
                        if nifty_close is not None:
                            nifty_today = float(nifty_close.iloc[i])
                            nifty_3day_low_close = float(nifty_close.iloc[max(0, i-3):i].min())
                            nifty_weak = nifty_today < nifty_3day_low_close
                        if not nifty_weak and price < three_day_low:
                            trades.append(self._make_trade(
                                entry_date, entry_price, shares, day,
                                price, "BELOW_3DAY_LOW"))
                            in_position = False
                            q_partial_done = False
                            continue

                    # Partial: +5% → sell 50%
                    if not q_partial_done and price >= entry_price * 1.05:
                        partial_shares = shares // 2
                        if partial_shares > 0:
                            trades.append(self._make_trade(
                                entry_date, entry_price, partial_shares, day,
                                price, "PARTIAL_5PCT"))
                            q_remaining = shares - partial_shares
                            q_partial_done = True

                    # Target/Exit remaining: 121% extension OR 3-day low (after partial)
                    # 3-day low skipped if Nifty itself is weak (Nifty close < Nifty 3-day low close)
                    if q_partial_done:
                        if price >= q_fib_target:
                            if q_remaining > 0:
                                trades.append(self._make_trade(
                                    entry_date, entry_price, q_remaining, day,
                                    price, "FIB_TARGET_121"))
                            in_position = False
                            q_partial_done = False
                            continue
                        three_day_low = float(lows.iloc[max(0, i-3):i].min()) if i >= 3 else float(lows.iloc[i])
                        nifty_weak = False
                        if nifty_close is not None and i >= 3:
                            nifty_today = float(nifty_close.iloc[i])
                            nifty_3day_low_close = float(nifty_close.iloc[max(0, i-3):i].min())
                            nifty_weak = nifty_today < nifty_3day_low_close
                        if not nifty_weak and price < three_day_low:
                            if q_remaining > 0:
                                trades.append(self._make_trade(
                                    entry_date, entry_price, q_remaining, day,
                                    price, "BELOW_3DAY_LOW"))
                            in_position = False
                            q_partial_done = False
                            continue

                if not in_position and f618 is not None and f786 is not None and f121 is not None:
                    within_1pct = (price <= f618 * 1.01)
                    vol_spike = vol_avg20 > 0 and vol_today > 1.5 * vol_avg20
                    if (price > f618 and within_1pct
                            and ibs > 0.5 and price > open_price
                            and vol_spike):
                        shares = int(capital // price)
                        if shares > 0:
                            entry_price = price
                            entry_date = day
                            in_position = True
                            q_partial_done = False
                            q_remaining = shares
                            q_fib_stop = f786
                            q_fib_target = f121
                continue

            elif strategy == "E":
                # Strategy E: EMA Pullback (scale-out)
                # Entry: EMA(50) > EMA(200) + Close crossed back above EMA(20)
                entry_signal = (ema50 > ema200
                                and prev_close is not None
                                and prev_ema20 is not None
                                and prev_close < prev_ema20
                                and price > ema20)

                if in_position and not partial_exit_done:
                    # Exit 1: +5% from entry → sell 50%
                    if price >= entry_price * 1.05:
                        half = shares // 2
                        trades.append(self._make_trade(
                            entry_date, entry_price, half, day, price,
                            "5PCT_PARTIAL"))
                        remaining_shares = shares - half
                        partial_exit_done = True
                        continue
                    # Exit 2: Close < EMA(20) → sell all
                    elif price < ema20:
                        trades.append(self._make_trade(
                            entry_date, entry_price, shares, day, price,
                            "EMA20_EXIT"))
                        in_position = False
                        partial_exit_done = False
                        continue
                    # Global: Close < EMA(50) → sell all
                    elif price < ema50:
                        trades.append(self._make_trade(
                            entry_date, entry_price, shares, day, price,
                            "EMA50_EXIT"))
                        in_position = False
                        partial_exit_done = False
                        continue

                if in_position and partial_exit_done:
                    # Second leg: remaining shares exit at EMA(20) or EMA(50)
                    if price < ema20:
                        trades.append(self._make_trade(
                            entry_date, entry_price, remaining_shares,
                            day, price, "EMA20_EXIT"))
                        in_position = False
                        partial_exit_done = False
                        continue
                    elif price < ema50:
                        trades.append(self._make_trade(
                            entry_date, entry_price, remaining_shares,
                            day, price, "EMA50_EXIT"))
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
                continue

            elif strategy == "F":
                # Strategy F: NR7 Squeeze (scale-out)
                # Entry: NR7 AND RSI(2) < 30 AND Close > EMA(50) AND EMA(50) > EMA(200)
                entry_signal = (nr7
                                and rsi2 < 30
                                and price > ema50
                                and ema50 > ema200)

                stop_hit = in_position and price < entry_price * 0.97

                if in_position and not partial_exit_done:
                    # Exit 1: +5% from entry → sell 50%
                    if price >= entry_price * 1.05:
                        half = shares // 2
                        trades.append(self._make_trade(
                            entry_date, entry_price, half, day, price,
                            "5PCT_PARTIAL"))
                        remaining_shares = shares - half
                        partial_exit_done = True
                        continue
                    # Stop: 3% or Close < EMA(50) → sell all
                    elif stop_hit:
                        trades.append(self._make_trade(
                            entry_date, entry_price, shares, day, price,
                            "STOP_3PCT"))
                        in_position = False
                        partial_exit_done = False
                        continue
                    elif price < ema50:
                        trades.append(self._make_trade(
                            entry_date, entry_price, shares, day, price,
                            "EMA50_EXIT"))
                        in_position = False
                        partial_exit_done = False
                        continue

                if in_position and partial_exit_done:
                    # Exit 2: Close > EMA(10) → sell remaining
                    if price > ema10:
                        trades.append(self._make_trade(
                            entry_date, entry_price, remaining_shares,
                            day, price, "EMA10_TARGET"))
                        in_position = False
                        partial_exit_done = False
                        continue
                    # Stop still applies to remaining
                    elif stop_hit:
                        trades.append(self._make_trade(
                            entry_date, entry_price, remaining_shares,
                            day, price, "STOP_3PCT"))
                        in_position = False
                        partial_exit_done = False
                        continue
                    elif price < ema50:
                        trades.append(self._make_trade(
                            entry_date, entry_price, remaining_shares,
                            day, price, "EMA50_EXIT"))
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
                continue

            elif strategy == "G":
                # Strategy G: IBS Extreme (scale-out)
                # Entry: IBS < 0.2 AND Close > EMA(50) AND EMA(50) > EMA(200)
                entry_signal = (ibs < 0.2
                                and price > ema50
                                and ema50 > ema200)

                stop_hit = in_position and price < entry_price * 0.97

                if in_position and not partial_exit_done:
                    # Exit 1: +5% from entry → sell 50%
                    if price >= entry_price * 1.05:
                        half = shares // 2
                        trades.append(self._make_trade(
                            entry_date, entry_price, half, day, price,
                            "5PCT_PARTIAL"))
                        remaining_shares = shares - half
                        partial_exit_done = True
                        continue
                    # Stop: 3% or Close < EMA(50) → sell all
                    elif stop_hit:
                        trades.append(self._make_trade(
                            entry_date, entry_price, shares, day, price,
                            "STOP_3PCT"))
                        in_position = False
                        partial_exit_done = False
                        continue
                    elif price < ema50:
                        trades.append(self._make_trade(
                            entry_date, entry_price, shares, day, price,
                            "EMA50_EXIT"))
                        in_position = False
                        partial_exit_done = False
                        continue

                if in_position and partial_exit_done:
                    # Exit 2: IBS > 0.8 → sell remaining
                    if ibs > 0.8:
                        trades.append(self._make_trade(
                            entry_date, entry_price, remaining_shares,
                            day, price, "IBS_HIGH"))
                        in_position = False
                        partial_exit_done = False
                        continue
                    # Stop still applies to remaining
                    elif stop_hit:
                        trades.append(self._make_trade(
                            entry_date, entry_price, remaining_shares,
                            day, price, "STOP_3PCT"))
                        in_position = False
                        partial_exit_done = False
                        continue
                    elif price < ema50:
                        trades.append(self._make_trade(
                            entry_date, entry_price, remaining_shares,
                            day, price, "EMA50_EXIT"))
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
                continue

            elif strategy == "H":
                # Strategy H: Consecutive Down Days (scale-out)
                # Check 4 consecutive lower closes: C[i] < C[i-1] < C[i-2] < C[i-3]
                has_4_down = (i >= 4
                              and closes.iloc[i] < closes.iloc[i-1]
                              and closes.iloc[i-1] < closes.iloc[i-2]
                              and closes.iloc[i-2] < closes.iloc[i-3]
                              and closes.iloc[i-3] < closes.iloc[i-4])
                entry_signal = (has_4_down
                                and price > ema50
                                and ema50 > ema200)

                stop_hit = in_position and price < entry_price * 0.97

                if in_position and not partial_exit_done:
                    # Exit 1: +5% from entry → sell 50%
                    if price >= entry_price * 1.05:
                        half = shares // 2
                        trades.append(self._make_trade(
                            entry_date, entry_price, half, day, price,
                            "5PCT_PARTIAL"))
                        remaining_shares = shares - half
                        partial_exit_done = True
                        continue
                    # Stop: 3% or Close < EMA(50) → sell all
                    elif stop_hit:
                        trades.append(self._make_trade(
                            entry_date, entry_price, shares, day, price,
                            "STOP_3PCT"))
                        in_position = False
                        partial_exit_done = False
                        continue
                    elif price < ema50:
                        trades.append(self._make_trade(
                            entry_date, entry_price, shares, day, price,
                            "EMA50_EXIT"))
                        in_position = False
                        partial_exit_done = False
                        continue

                if in_position and partial_exit_done:
                    # Exit 2: first up close → sell remaining
                    if prev_close is not None and price > prev_close:
                        trades.append(self._make_trade(
                            entry_date, entry_price, remaining_shares,
                            day, price, "UP_CLOSE"))
                        in_position = False
                        partial_exit_done = False
                        continue
                    # Stop still applies
                    elif stop_hit:
                        trades.append(self._make_trade(
                            entry_date, entry_price, remaining_shares,
                            day, price, "STOP_3PCT"))
                        in_position = False
                        partial_exit_done = False
                        continue
                    elif price < ema50:
                        trades.append(self._make_trade(
                            entry_date, entry_price, remaining_shares,
                            day, price, "EMA50_EXIT"))
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
                continue

            elif strategy == "D":
                # Strategy D: RSI Reversal
                entry_signal = (price > ema50
                                and price > ema200
                                and not pd.isna(rsi2_prev)
                                and rsi2_prev < 10
                                and rsi2 > 10)

                # --- Single exit mode (EMA or %) ---
                if in_position and not use_pct_target and ema_exit is not None:
                    if price <= ema_exit:
                        was_below_exit_ema = True

                stop_hit = in_position and price < entry_price * 0.97
                if use_pct_target:
                    target_hit = in_position and price >= entry_price * (1 + pct_target)
                else:
                    target_hit = was_below_exit_ema and price > ema_exit
                if target_hit:
                    exit_signal = True
                    exit_reason = f"{target_label}_TARGET"
                elif stop_hit:
                    exit_signal = True
                    exit_reason = "STOP_3PCT"
                elif price < ema50:
                    exit_signal = True
                    exit_reason = "EMA50_EXIT"
                elif price < ema200:
                    exit_signal = True
                    exit_reason = "EMA200_EXIT"
                else:
                    exit_signal = False
                    exit_reason = ""
            else:
                # Entry conditions (same for B and C)
                entry_signal = (rsi2 >= rsi2_entry
                                and price > ema50
                                and price > ema200)
                # Exit conditions
                if strategy == "B":
                    exit_signal = (price < ema10
                                   or price < ema50
                                   or price < ema200)
                    exit_reason = "EMA10_EXIT" if price < ema10 else (
                        "EMA50_EXIT" if price < ema50 else "EMA200_EXIT")
                else:  # Strategy C
                    exit_signal = (rsi2 < 30
                                   or price < ema50
                                   or price < ema200)
                    exit_reason = "RSI2_EXIT" if rsi2 < 30 else (
                        "EMA50_EXIT" if price < ema50 else "EMA200_EXIT")

            # Signal logic (for non-scale-out modes)
            if entry_signal and not in_position:
                shares = int(capital // price)
                if shares > 0:
                    entry_price = price
                    entry_date = day
                    in_position = True
                    was_below_exit_ema = False
                    days_below_ema20 = 0

            elif exit_signal and in_position:
                trades.append(self._make_trade(
                    entry_date, entry_price, shares, day, price, exit_reason
                ))
                in_position = False
                was_below_exit_ema = False
                days_below_ema20 = 0

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
            "exit_target": target_label if strategy == "D" else (
                str(exit_target) if strategy in ("I", "J", "K") and exit_target else None),
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
                              progress_callback=None):
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

        end_date = datetime.now()
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
            two_week_low = lows.rolling(window=10, min_periods=10).min().shift(1)

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
            ema10_series = closes.ewm(span=10, adjust=False).mean()
            ema20_series = closes.ewm(span=20, adjust=False).mean()
            ema50_series = closes.ewm(span=50, adjust=False).mean()
            ema200_series = closes.ewm(span=200, adjust=False).mean()

            # RSI(2)
            rsi2_series = self._calculate_rsi_series(closes, 2)

            # RS spread for Strategy K (3M) and L (6M)
            rs_spread_series = None
            rs_spread_6m_series = None
            if not nifty_raw.empty:
                nifty_close = nifty_raw["Close"].reindex(daily.index, method="ffill")
                stock_3m = closes.pct_change(periods=63)
                nifty_3m = nifty_close.pct_change(periods=63)
                rs_spread_series = stock_3m - nifty_3m
                stock_6m = closes.pct_change(periods=126)
                nifty_6m = nifty_close.pct_change(periods=126)
                rs_spread_6m_series = stock_6m - nifty_6m

            # AVWAP anchored from 6-month swing low (for Strategy P)
            avwap_series, avwap_age_series = self._calculate_avwap_series(highs, lows, closes, vol_series, lookback=120)

            # Fibonacci levels from 6-month high/low (for Strategy Q)
            fib_high_6m = highs.rolling(window=120, min_periods=120).max()
            fib_low_6m = lows.rolling(window=120, min_periods=120).min()
            fib_range = fib_high_6m - fib_low_6m
            fib_618_series = fib_high_6m - 0.618 * fib_range
            fib_786_series = fib_high_6m - 0.70 * fib_range
            fib_ext_121_series = fib_high_6m + 0.21 * fib_range

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
                "two_week_low": two_week_low,
                "weekly_support": weekly_support_series,
                "weekly_low_stop": weekly_low_stop_series,
                "rsi2": rsi2_series,
                "volume": vol_series,
                "vol_avg20": vol_avg20,
                "avwap": avwap_series,
                "avwap_age": avwap_age_series,
                "fib_618": fib_618_series,
                "fib_786": fib_786_series,
                "fib_ext_121": fib_ext_121_series,
                "ema10": ema10_series,
                "ema20": ema20_series,
                "ema50": ema50_series,
                "ema200": ema200_series,
                "rs_spread": rs_spread_series,
                "rs_spread_6m": rs_spread_6m_series,
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
        if not nifty_raw.empty:
            for ts, row in nifty_raw.iterrows():
                nifty_close_by_date[ts.date()] = float(row["Close"])

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
                    # J exits (close mode): support break, +5% partial, +10% remaining
                    entry_stop = pos["entry_stop_j"]
                    t1 = pos["entry_price"] * 1.05
                    t2 = pos["entry_price"] * 1.10

                    # Nifty drop shield: skip support break if Nifty fell same or more
                    j_nifty_shields = False
                    j_nifty_entry = pos.get("nifty_at_entry", 0.0)
                    if j_nifty_entry > 0 and nifty_today_close > 0:
                        nifty_pct = (nifty_today_close - j_nifty_entry) / j_nifty_entry
                        stock_pct = (price - pos["entry_price"]) / pos["entry_price"]
                        if nifty_pct <= stock_pct and nifty_pct < 0:
                            j_nifty_shields = True

                    if not pos["partial_exit_done"]:
                        # Stop check
                        if not j_nifty_shields and price < entry_stop:
                            trades.append(self._make_portfolio_trade(
                                pos, pos["shares"], day, price,
                                "SUPPORT_BREAK"))
                            exited = True
                        elif price >= t1:
                            half = pos["shares"] // 2
                            trades.append(self._make_portfolio_trade(
                                pos, half, day, price, "5PCT_PARTIAL"))
                            pos["remaining_shares"] = pos["shares"] - half
                            pos["partial_exit_done"] = True
                            if price >= t2:
                                trades.append(self._make_portfolio_trade(
                                    pos, pos["remaining_shares"], day, price,
                                    "10PCT"))
                                exited = True
                    else:
                        # Remaining shares
                        if not j_nifty_shields and price < entry_stop:
                            trades.append(self._make_portfolio_trade(
                                pos, pos["remaining_shares"], day, price,
                                "SUPPORT_BREAK"))
                            exited = True
                        # Exit remaining: Close < 3-day low (skip if Nifty weak)
                        elif not nifty_weak_today and day_idx >= 3:
                            three_day_low = float(ind["lows"].iloc[max(0, i-3):i].min())
                            if price < three_day_low:
                                trades.append(self._make_portfolio_trade(
                                    pos, pos["remaining_shares"], day, price,
                                    "BELOW_3DAY_LOW"))
                                exited = True
                        if not exited and price >= t2:
                            trades.append(self._make_portfolio_trade(
                                pos, pos["remaining_shares"], day, price,
                                "10PCT"))
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

                elif pos["strategy"] == "L":
                    # L exits: RS < -2% (with 5-day min hold) OR Close < EMA(50)
                    rs_val = 0.0
                    if ind["rs_spread_6m"] is not None and not pd.isna(ind["rs_spread_6m"].iloc[i]):
                        rs_val = float(ind["rs_spread_6m"].iloc[i])
                    ema50 = float(ind["ema50"].iloc[i])
                    hold_days = (day - pos["entry_date"]).days
                    if price < ema50:
                        trades.append(self._make_portfolio_trade(
                            pos, pos["shares"], day, price, "EMA50_EXIT"))
                        exited = True
                    elif hold_days >= 5 and rs_val < -0.02:
                        trades.append(self._make_portfolio_trade(
                            pos, pos["shares"], day, price, "RS_NEGATIVE"))
                        exited = True

                elif pos["strategy"] == "M":
                    ema10_val = float(ind["ema10"].iloc[i])
                    ema20_val = float(ind["ema20"].iloc[i])
                    prev_e10 = float(ind["ema10"].iloc[i - 1]) if i > 0 else ema10_val
                    prev_e20 = float(ind["ema20"].iloc[i - 1]) if i > 0 else ema20_val
                    # M partial: +5% → sell 50%
                    if not pos["partial_exit_done"] and price >= pos["entry_price"] * 1.05:
                        partial_shares = pos["shares"] // 2
                        if partial_shares > 0:
                            trades.append(self._make_portfolio_trade(
                                pos, partial_shares, day, price, "PARTIAL_5PCT"))
                            pos["remaining_shares"] = pos["shares"] - partial_shares
                            pos["partial_exit_done"] = True
                    # M exit: EMA(10) crosses below EMA(20)
                    if prev_e10 >= prev_e20 and ema10_val < ema20_val:
                        sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                        trades.append(self._make_portfolio_trade(
                            pos, sz, day, price, "EMA10_BELOW_EMA20"))
                        exited = True

                elif pos["strategy"] == "P":
                    avwap_val = float(ind["avwap"].iloc[i]) if not pd.isna(ind["avwap"].iloc[i]) else None
                    if avwap_val is not None:
                        # P stop: 3% below entry
                        if price < pos["entry_price"] * 0.97:
                            sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                            trades.append(self._make_portfolio_trade(
                                pos, sz, day, price, "STOP_3PCT"))
                            exited = True
                        else:
                            # P partial: +5% → sell 50%
                            if not pos["partial_exit_done"] and price >= pos["entry_price"] * 1.05:
                                partial_shares = pos["shares"] // 2
                                if partial_shares > 0:
                                    trades.append(self._make_portfolio_trade(
                                        pos, partial_shares, day, price, "PARTIAL_5PCT"))
                                    pos["remaining_shares"] = pos["shares"] - partial_shares
                                    pos["partial_exit_done"] = True
                            # Exit remaining 50%: Close < 3-day low (after partial)
                            # Skip if Nifty itself is weak (Nifty close < Nifty 3-day low close)
                            if pos["partial_exit_done"] and not nifty_weak_today:
                                three_day_low = float(ind["lows"].iloc[max(0, i-3):i].min()) if i >= 3 else low
                                if price < three_day_low:
                                    if pos["remaining_shares"] > 0:
                                        trades.append(self._make_portfolio_trade(
                                            pos, pos["remaining_shares"], day, price, "BELOW_3DAY_LOW"))
                                    exited = True

                elif pos["strategy"] == "Q":
                    # Q stop: Close < 70% fib level (locked at entry)
                    if price < pos.get("q_fib_stop", 0):
                        sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                        trades.append(self._make_portfolio_trade(
                            pos, sz, day, price, "FIB_STOP_70"))
                        exited = True
                    else:
                        # 3-day low exit before partial (skip if Nifty weak)
                        if not pos["partial_exit_done"] and not nifty_weak_today and i >= 3:
                            three_day_low = float(ind["lows"].iloc[max(0, i-3):i].min())
                            if price < three_day_low:
                                trades.append(self._make_portfolio_trade(
                                    pos, pos["shares"], day, price, "BELOW_3DAY_LOW"))
                                exited = True
                        # Q partial: +5% → sell 50%
                        if not exited and not pos["partial_exit_done"] and price >= pos["entry_price"] * 1.05:
                            partial_shares = pos["shares"] // 2
                            if partial_shares > 0:
                                trades.append(self._make_portfolio_trade(
                                    pos, partial_shares, day, price, "PARTIAL_5PCT"))
                                pos["remaining_shares"] = pos["shares"] - partial_shares
                                pos["partial_exit_done"] = True
                        # Q target/exit remaining: 121% extension OR 3-day low (after partial)
                        # 3-day low skipped if Nifty itself is weak
                        if pos["partial_exit_done"]:
                            if price >= pos.get("q_fib_target", float("inf")):
                                if pos["remaining_shares"] > 0:
                                    trades.append(self._make_portfolio_trade(
                                        pos, pos["remaining_shares"], day, price, "FIB_TARGET_121"))
                                exited = True
                            elif not nifty_weak_today:
                                three_day_low = float(ind["lows"].iloc[max(0, i-3):i].min()) if i >= 3 else low
                                if price < three_day_low:
                                    if pos["remaining_shares"] > 0:
                                        trades.append(self._make_portfolio_trade(
                                            pos, pos["remaining_shares"], day, price, "BELOW_3DAY_LOW"))
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
                            if (close_near >= 0 and close_near <= 3.0
                                    and ibs > 0.5 and is_green):
                                signals.append({
                                    "symbol": ticker,
                                    "strategy": "J",
                                    "price": price,
                                    "ibs": ibs,
                                    "entry_support_j": ws,
                                    "entry_stop_j": wls,  # Stop at 26-week weekly low
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

                # Strategy L entry: 6M RS crosses above +2% buffer
                if "L" in strategies:
                    if ind["rs_spread_6m"] is not None and not pd.isna(ind["rs_spread_6m"].iloc[i]):
                        rs_val = float(ind["rs_spread_6m"].iloc[i])
                        rs_prev_l = 0.0
                        if i > 0 and not pd.isna(ind["rs_spread_6m"].iloc[i - 1]):
                            rs_prev_l = float(ind["rs_spread_6m"].iloc[i - 1])
                        if rs_prev_l < 0.02 and rs_val > 0.02:
                            already = any(s["symbol"] == ticker for s in signals)
                            if not already:
                                signals.append({
                                    "symbol": ticker,
                                    "strategy": "L",
                                    "price": price,
                                })

                # Strategy M entry: EMA(10) crosses above EMA(20)
                #   + Close > EMA(50) + EMA(20) rising
                if "M" in strategies:
                    ema10_val = float(ind["ema10"].iloc[i])
                    ema20_val = float(ind["ema20"].iloc[i])
                    ema50_val = float(ind["ema50"].iloc[i])
                    prev_e10 = float(ind["ema10"].iloc[i - 1]) if i > 0 else ema10_val
                    prev_e20 = float(ind["ema20"].iloc[i - 1]) if i > 0 else ema20_val
                    ema20_rising = (ema20_val > prev_e20)
                    if (prev_e10 < prev_e20 and ema10_val > ema20_val
                            and price > ema50_val and ema20_rising):
                        already = any(s["symbol"] == ticker for s in signals)
                        if not already:
                            signals.append({
                                "symbol": ticker,
                                "strategy": "M",
                                "price": price,
                            })

                # Strategy P entry: AVWAP bounce
                #   Close > AVWAP, within 1% of AVWAP, IBS > 0.3, green candle
                if "P" in strategies:
                    avwap_val = float(ind["avwap"].iloc[i]) if not pd.isna(ind["avwap"].iloc[i]) else None
                    if avwap_val is not None:
                        ibs_val = float(ind["ibs"].iloc[i])
                        open_price = float(ind["opens"].iloc[i])
                        within_1pct = (price <= avwap_val * 1.01)
                        if (price > avwap_val and within_1pct
                                and ibs_val > 0.5 and price > open_price):
                            already = any(s["symbol"] == ticker for s in signals)
                            if not already:
                                signals.append({
                                    "symbol": ticker,
                                    "strategy": "P",
                                    "price": price,
                                })

                # Strategy Q entry: Fibonacci 61.8% bounce
                if "Q" in strategies:
                    f618 = float(ind["fib_618"].iloc[i]) if not pd.isna(ind["fib_618"].iloc[i]) else None
                    f786 = float(ind["fib_786"].iloc[i]) if not pd.isna(ind["fib_786"].iloc[i]) else None
                    f121 = float(ind["fib_ext_121"].iloc[i]) if not pd.isna(ind["fib_ext_121"].iloc[i]) else None
                    if f618 is not None and f786 is not None and f121 is not None:
                        ibs_val = float(ind["ibs"].iloc[i])
                        open_price = float(ind["opens"].iloc[i])
                        vol_now = float(ind["volume"].iloc[i])
                        vol_avg = float(ind["vol_avg20"].iloc[i]) if not pd.isna(ind["vol_avg20"].iloc[i]) else 0.0
                        within_1pct = (price <= f618 * 1.01)
                        vol_spike = vol_avg > 0 and vol_now > 1.5 * vol_avg
                        if (price > f618 and within_1pct
                                and ibs_val > 0.5 and price > open_price
                                and vol_spike):
                            already = any(s["symbol"] == ticker for s in signals)
                            if not already:
                                signals.append({
                                    "symbol": ticker,
                                    "strategy": "Q",
                                    "price": price,
                                    "q_fib_stop": f786,
                                    "q_fib_target": f121,
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
                        elif sig["strategy"] == "Q":
                            pos["q_fib_stop"] = sig["q_fib_stop"]
                            pos["q_fib_target"] = sig["q_fib_target"]
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

        strat_labels = {"J": "J(3:15)", "K": "K(3:15)", "L": "L(Pure RS)", "M": "M(EMA 10/20)", "P": "P(AVWAP)", "Q": "Q(Fib 61.8)"}
        strat_label = " + ".join(strat_labels.get(s, s) for s in strategies)

        return {
            "strategy": "Portfolio",
            "strategies": strategies,
            "strategies_label": strat_label,
            "exit_target": "close",
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

    def run_all_stocks(self, period_days, capital=100000, progress_callback=None, universe=50):
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
            end_date = datetime.now()
            daily_start = end_date - timedelta(days=period_days + 500)
            try:
                daily = yf.Ticker(nse_symbol).history(start=daily_start, end=end_date)
            except Exception:
                daily = pd.DataFrame()

            row = {"symbol": ticker, "results": {}}

            for strat, exit_tgt, label in BATCH_VARIANTS:
                result = self.run(ticker, period_days, strategy=strat,
                                  capital=capital, exit_target=exit_tgt,
                                  _daily_data=daily)
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

