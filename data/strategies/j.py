"""Strategy J — Weekly Close Support Bounce.

Per-ticker compute. Returns the J signal dict, or None.
"""

import numpy as np
import pandas as pd


def compute_j_signal(ticker, daily, closes, opens, highs, lows, i, price,
                     low, ibs, is_green):
    """Returns the J signal dict if today's bar passes the filters, else None.
    Math identical to the original inline block."""
    try:
        weekly = daily.resample("W-FRI").agg({
            "Open":  "first", "High": "max", "Low": "min",
            "Close": "last",  "Volume": "sum",
        }).dropna()
        if len(weekly) < 27:
            return None

        # 26-week support (skip last 2 weeks — use proven, not noise).
        w_support_series = weekly["Close"].rolling(window=26, min_periods=26).min().shift(2)
        w_support_daily = w_support_series.reindex(daily.index, method="ffill")
        ws = float(w_support_daily.iloc[i]) if not pd.isna(w_support_daily.iloc[i]) else None

        # 26-week weekly low for stop (also skip last 2 weeks).
        w_low_stop_series = weekly["Low"].rolling(window=26, min_periods=26).min().shift(2)
        w_low_stop_daily = w_low_stop_series.reindex(daily.index, method="ffill")
        wls = float(w_low_stop_daily.iloc[i]) if not pd.isna(w_low_stop_daily.iloc[i]) else None

        if not (ws and ws > 0):
            return None

        close_near_pct = ((price - ws) / ws) * 100

        # CCI(20) confirmation.
        tp = (highs + lows + closes) / 3
        sma_tp = tp.rolling(window=20, min_periods=20).mean()
        mean_dev = tp.rolling(window=20, min_periods=20).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        cci_series = (tp - sma_tp) / (0.015 * mean_dev)
        cci_val = float(cci_series.iloc[i]) if not pd.isna(cci_series.iloc[i]) else 0.0

        if not (0 <= close_near_pct <= 3.0
                and ibs > 0.5 and is_green and cci_val > -100):
            return None

        raw_stop = wls if wls else ws
        j_stop_pct = round((price - raw_stop) / price * 100, 2) if price > 0 else 99.0

        # ATR14 for ranking.
        prev_close_j = closes.shift(1)
        tr_j = pd.concat([
            highs - lows,
            (highs - prev_close_j).abs(),
            (lows  - prev_close_j).abs(),
        ], axis=1).max(axis=1)
        atr14_j = float(tr_j.ewm(alpha=1/14, min_periods=14, adjust=False).mean().iloc[i])
        atr_norm_j = round(atr14_j / price * 100, 2) if price > 0 else 99.0

        return {
            "ticker":         ticker,
            "price":          round(price, 2),
            "support":        round(ws, 2),
            "stop":           round(raw_stop, 2),
            "stop_pct":       j_stop_pct,
            "close_near_pct": round(close_near_pct, 2),
            "ibs":            round(ibs, 2),
            "low":            round(low, 2),
            "atr_pct":        atr_norm_j,
        }
    except Exception:
        return None
