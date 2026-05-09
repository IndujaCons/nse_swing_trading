"""Strategy MW — Weekly ADX trend.

Per-ticker compute. Returns the MW signal dict, or None. Suppressed if
the ticker already produced J, T, or R (dedup).
"""

from datetime import datetime
import numpy as np
import pandas as pd


def compute_mw_signal(ticker, daily, closes, highs, lows, i, price,
                      ibs, is_green, jtr_signal_tickers, actual_date,
                      calc_adx):
    """Returns the MW signal dict if today's bar passes, else None.

    `calc_adx` is _calculate_adx_series from data/live_signals_engine.py
    (passed in to avoid circular import). `actual_date` is the latest
    trading day across the scan (used to align weekly bars). Math
    identical to the original inline block."""
    if ticker in jtr_signal_tickers:
        return None
    if not (is_green and ibs > 0.5):
        return None
    try:
        weekly_mw = daily.resample("W-FRI").agg({
            "Open":  "first", "High": "max", "Low": "min",
            "Close": "last",  "Volume": "sum",
        }).dropna()
        if len(weekly_mw) < 28:
            return None

        mw_adx, mw_pdi, mw_mdi = calc_adx(
            weekly_mw["High"], weekly_mw["Low"], weekly_mw["Close"])
        w_dates = weekly_mw.index
        day_ts = pd.Timestamp(actual_date or datetime.now().date())
        if w_dates.tz:
            day_ts = day_ts.tz_localize(w_dates.tz)
        w_before = w_dates[w_dates < day_ts]
        if len(w_before) < 2:
            return None

        w_idx = len(w_before) - 1
        curr_adx = float(mw_adx.iloc[w_idx])
        prev_adx = float(mw_adx.iloc[w_idx - 1])
        plus_di  = float(mw_pdi.iloc[w_idx])
        minus_di = float(mw_mdi.iloc[w_idx])
        if not (not np.isnan(curr_adx) and not np.isnan(prev_adx)
                and curr_adx >= 25 and curr_adx > prev_adx
                and plus_di > minus_di):
            return None

        prev_close_mw = closes.shift(1)
        tr_mw = pd.concat([
            highs - lows,
            (highs - prev_close_mw).abs(),
            (lows  - prev_close_mw).abs(),
        ], axis=1).max(axis=1)
        atr14_mw = float(tr_mw.ewm(alpha=1/14, min_periods=14, adjust=False).mean().iloc[i])
        atr_norm_mw = round(atr14_mw / price * 100, 2) if price > 0 else 99.0

        return {
            "ticker":   ticker,
            "price":    round(price, 2),
            "adx":      round(curr_adx, 1),
            "plus_di":  round(plus_di, 1),
            "minus_di": round(minus_di, 1),
            "stop_pct": 5.0,
            "atr_pct":  atr_norm_mw,
        }
    except Exception:
        return None
