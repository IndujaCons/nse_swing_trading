"""IBD-style RS rating per ticker.

Per-ticker compute step — produces the candidate dict accumulated in
`rs_ibd_candidates` (ranked by `weighted` score after the loop) AND
returns the 5-day weighted-score history used for the consecutive-day
filter.
"""

import pandas as pd


def compute_rs_ibd_features(ticker, daily, closes, highs, lows, i, price,
                            bench_ret123_val, nifty_regime_on, sector_map):
    """Returns (candidate_dict, day_scores) — both None if insufficient data.
    Math identical to the original inline block.

    `sector_map` is the STOCK_SECTOR_MAP dict (ticker → 'OTHER' fallback).
    """
    if i < 252:
        return None, None
    try:
        # 5-day weighted-score history (used for consecutive-day filter post-loop).
        day_scores = []
        for d_offset in range(4, -1, -1):  # days i-4 through i
            di = i - d_offset
            if di >= 252:
                dq4 = (float(closes.iloc[di])       / float(closes.iloc[di - 63])  - 1) * 100
                dq3 = (float(closes.iloc[di - 63])  / float(closes.iloc[di - 126]) - 1) * 100
                dq2 = (float(closes.iloc[di - 126]) / float(closes.iloc[di - 189]) - 1) * 100
                dq1 = (float(closes.iloc[di - 189]) / float(closes.iloc[di - 252]) - 1) * 100
                day_scores.append(0.4 * dq4 + 0.2 * dq3 + 0.2 * dq2 + 0.2 * dq1)
        weighted = day_scores[-1] if day_scores else 0

        # RS-123d (display only).
        stock_ret123 = (price / float(closes.iloc[i - 123]) - 1) * 100
        rs_val = stock_ret123 - bench_ret123_val

        # Price > 30-week EMA.
        weekly_rs = daily.resample("W-FRI").agg({"Close": "last"}).dropna()
        ema30w_val = None
        if len(weekly_rs) >= 30:
            ema30w = weekly_rs["Close"].ewm(span=30, adjust=False).mean()
            ema30w_daily = ema30w.reindex(daily.index, method="ffill")
            ema30w_val = float(ema30w_daily.iloc[i])

        # dist_high: price ≥ 3% below 20d high.
        high_20d = float(highs.iloc[max(0, i - 20):i].max()) if i >= 20 else price
        dist_high_pct = round((high_20d - price) / high_20d * 100, 1) if high_20d > 0 else 0.0

        # ATR%.
        prev_cl_rs = closes.shift(1)
        tr_rs = pd.concat([highs - lows, (highs - prev_cl_rs).abs(), (lows - prev_cl_rs).abs()], axis=1).max(axis=1)
        atr14_rs = float(tr_rs.ewm(alpha=1/14, min_periods=14, adjust=False).mean().iloc[i])
        atr_pct_rs = round(atr14_rs / price * 100, 2) if price > 0 else 99.0

        candidate = {
            "ticker":         ticker,
            "price":          round(price, 2),
            "weighted":       weighted,
            "rs_pct":         round(rs_val, 1),
            "dist_high_pct":  dist_high_pct,
            "atr_pct":        atr_pct_rs,
            "stop_pct":       8.0,
            "ema30w_val":     ema30w_val,
            "above_ema":      ema30w_val is not None and not pd.isna(ema30w_val) and price > ema30w_val,
            "dist_high_ok":   not (i >= 20 and high_20d > 0 and (high_20d - price) / high_20d < 0.03),
            "sector":         (sector_map or {}).get(ticker, "OTHER"),
            "regime_off":     not nifty_regime_on,
        }
        return candidate, day_scores
    except Exception:
        return None, None
