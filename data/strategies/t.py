"""Strategy T — Keltner Channel Pullback.

Per-ticker compute. Returns the T signal dict, or None.
"""

import pandas as pd


def compute_t_signal(ticker, closes, highs, lows, i, price,
                     ibs, is_green, j_signal_tickers):
    """Returns the T signal dict if today's bar passes, else None.
    `j_signal_tickers` is the set of tickers that already produced a J
    signal — T is suppressed for them (matches backtest dedup).
    Math identical to the original inline block."""
    try:
        ema20_val = float(closes.ewm(span=20, adjust=False).mean().iloc[i])
        prev_close_s = closes.shift(1)
        true_range = pd.concat([
            highs - lows,
            (highs - prev_close_s).abs(),
            (lows  - prev_close_s).abs(),
        ], axis=1).max(axis=1)
        atr14 = float(true_range.ewm(alpha=1/14, min_periods=14, adjust=False).mean().iloc[i])
        if atr14 <= 0:
            return None

        upper_keltner = ema20_val + 2 * atr14
        near_ema20 = abs(price - ema20_val) / ema20_val <= 0.01

        ema20_s = closes.ewm(span=20, adjust=False).mean()
        atr14_s = true_range.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        was_at_upper = False
        for lb_j in range(max(0, i - 10), i):
            past_high  = float(highs.iloc[lb_j])
            past_ema20 = float(ema20_s.iloc[lb_j])
            past_atr14 = float(atr14_s.iloc[lb_j]) if not pd.isna(atr14_s.iloc[lb_j]) else 0.0
            if past_high >= past_ema20 + 2 * past_atr14:
                was_at_upper = True
                break

        if not (near_ema20 and was_at_upper and is_green
                and ticker not in j_signal_tickers and ibs > 0.5):
            return None

        atr_norm_t = round(atr14 / price * 100, 2) if price > 0 else 99.0
        return {
            "ticker":        ticker,
            "price":         round(price, 2),
            "ema20":         round(ema20_val, 2),
            "upper_keltner": round(upper_keltner, 2),
            "stop_pct":      5.0,
            "atr_pct":       atr_norm_t,
        }
    except Exception:
        return None
