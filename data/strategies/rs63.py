"""RS63 satellite strategy — Vivek Bajaj adapted.

Per-ticker compute step. Returns a signal dict if today's bar passes the
RS63(5d) > 0 + RSI(14, 3d avg) > 50 + IBS > 0.5 + green-candle filter
PLUS the optional 1h-RS63 confirmation. None otherwise.
"""

import pandas as pd
import yfinance as yf


def compute_rs63_signal(ticker, daily, closes, lows, volumes, i, price, low,
                        ibs, is_green, bench_raw):
    """Returns the RS63 candidate dict for one ticker, or None.
    Math identical to the original inline block."""
    if i < 70 or bench_raw is None or bench_raw.empty:
        return None
    try:
        bench_aligned = bench_raw["Close"].reindex(daily.index, method="ffill")
        rs_ratio = closes / bench_aligned
        rs63_raw = (rs_ratio / rs_ratio.shift(63) - 1) * 100
        rs63_smooth = rs63_raw.rolling(5, min_periods=1).mean()

        # RSI(14) smoothed 3-day avg
        delta_r = closes.diff()
        gain_r = delta_r.clip(lower=0)
        loss_r = (-delta_r).clip(lower=0)
        avg_gain_r = gain_r.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss_r = loss_r.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rsi_raw = 100 - (100 / (1 + avg_gain_r / avg_loss_r.replace(0, 1e-10)))
        rsi_smooth = rsi_raw.rolling(3, min_periods=1).mean()

        rs63_v = float(rs63_smooth.iloc[i]) if not pd.isna(rs63_smooth.iloc[i]) else -1
        rsi_v  = float(rsi_smooth.iloc[i])  if not pd.isna(rsi_smooth.iloc[i])  else 0

        if not (rs63_v > 0 and rsi_v > 50 and ibs > 0.5 and is_green):
            return None

        # 1-hour RS63 filter — must also be positive (or unavailable).
        rs63_1h_val = None
        try:
            h1_stk = yf.Ticker(f"{ticker}.NS").history(period="20d", interval="1h")
            h1_bch = yf.Ticker("^CNX200").history(period="20d", interval="1h")
            if len(h1_stk) >= 64 and not h1_bch.empty:
                h1_cls = h1_stk["Close"].astype(float)
                h1_bch_idx = h1_bch["Close"].astype(float)
                h1_bch_idx.index = h1_bch_idx.index.tz_localize(None) if h1_bch_idx.index.tzinfo else h1_bch_idx.index
                h1_cls.index     = h1_cls.index.tz_localize(None)     if h1_cls.index.tzinfo     else h1_cls.index
                h1_bch_aligned = h1_bch_idx.reindex(h1_cls.index, method="ffill").ffill()
                rs_1h = h1_cls / h1_bch_aligned
                rs63_1h_raw = (rs_1h / rs_1h.shift(63) - 1) * 100
                rs63_1h_sma = rs63_1h_raw.rolling(5, min_periods=1).mean()
                val = float(rs63_1h_sma.iloc[-1]) if not pd.isna(rs63_1h_sma.iloc[-1]) else None
                rs63_1h_val = round(val, 1) if val is not None else None
        except Exception:
            pass

        if rs63_1h_val is not None and rs63_1h_val <= 0:
            return None

        # Stop distance for ranking.
        low_20d = float(lows.rolling(20).min().iloc[i]) if i >= 20 else low
        stop_pct = round((price - low_20d) / price * 100, 1) if price > 0 else 99
        sl_price = round(price * 0.92, 1)

        # Volume ratio (today vs 20d avg).
        vol_today   = float(volumes.iloc[i])
        vol_20d_avg = float(volumes.iloc[max(0, i - 20):i].mean()) if i >= 5 else vol_today
        vol_ratio   = round(vol_today / vol_20d_avg, 1) if vol_20d_avg > 0 else None

        return {
            "ticker":    ticker,
            "price":     round(price, 2),
            "rs63":      round(rs63_v, 1),
            "rs63_1h":   rs63_1h_val,
            "rsi":       round(rsi_v, 1),
            "ibs":       round(ibs, 2),
            "stop_pct":  stop_pct,
            "sl_price":  sl_price,
            "vol_ratio": vol_ratio,
            "rank":      0,  # set after sorting
        }
    except Exception:
        return None
