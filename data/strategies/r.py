"""Strategy R — Bullish RSI Divergence (regular + hidden).

Per-ticker compute. Returns the R signal dict, or None. Suppressed if
the ticker already produced J or T (dedup).
"""

import pandas as pd


def compute_r_signal(ticker, closes, highs, lows, i, price,
                     ibs, is_green, jt_signal_tickers,
                     calc_rsi, find_swing_lows,
                     detect_bullish_divergence, detect_hidden_bullish_divergence):
    """Returns the R signal dict if today's bar passes, else None.

    Helpers (`calc_rsi`, `find_swing_lows`, the two `detect_*` functions)
    are passed in to avoid a circular import — they live in
    data/live_signals_engine.py. Math identical to the original inline block."""
    if ticker in jt_signal_tickers:
        return None
    if not (is_green and ibs > 0.5):
        return None
    try:
        rsi14_series = calc_rsi(closes, 14)
        swing_lows = find_swing_lows(lows)
        rsi14_vals = rsi14_series.values
        lows_vals = lows.values

        divergence, swing_low_val, rsi_at_low = detect_bullish_divergence(
            lows_vals, rsi14_vals, i, swing_lows, rsi_threshold=35)
        r_div_type = "regular"
        if not divergence:
            ema50_val = float(closes.ewm(span=50, adjust=False).mean().iloc[i])
            if price > ema50_val:
                divergence, swing_low_val, rsi_at_low = detect_hidden_bullish_divergence(
                    lows_vals, rsi14_vals, i, swing_lows)
                if divergence:
                    r_div_type = "hidden"

        if not (divergence and swing_low_val is not None):
            return None

        rsi14_at_bar = float(rsi14_series.iloc[i]) if not pd.isna(rsi14_series.iloc[i]) else 0.0
        r_struct_stop = round(swing_low_val * 0.99, 2)
        r_stop_pct = round((price - r_struct_stop) / price * 100, 2) if price > 0 else 99.0
        r_min_stop = 2.0 if r_div_type == "hidden" else 0.0
        if not (r_min_stop < r_stop_pct <= 6.0):
            return None

        # ATR14 for ranking.
        prev_close_r = closes.shift(1)
        tr_r = pd.concat([
            highs - lows,
            (highs - prev_close_r).abs(),
            (lows  - prev_close_r).abs(),
        ], axis=1).max(axis=1)
        atr14_r = float(tr_r.ewm(alpha=1/14, min_periods=14, adjust=False).mean().iloc[i])
        atr_norm_r = round(atr14_r / price * 100, 2) if price > 0 else 99.0

        return {
            "ticker":     ticker,
            "price":      round(price, 2),
            "rsi14":      round(rsi14_at_bar, 1),
            "rsi_at_low": round(float(rsi_at_low), 1),
            "swing_low":  round(swing_low_val, 2),
            "stop":       r_struct_stop,
            "stop_pct":   r_stop_pct,
            "atr_pct":    atr_norm_r,
            "div_type":   r_div_type,
        }
    except Exception:
        return None
