"""Relative Rotation Graph (RRG) engine — pure compute, no I/O.

Implements a standard, widely-published open-source approximation of the
JdK RS-Ratio / RS-Momentum methodology popularized by StockCharts
(stockcharts.com/.../relative-rotation-graphs-rrg-charts). StockCharts does
not publish the exact proprietary formula, so this is a reasonable rolling
Z-score reconstruction, not the trademarked exact algorithm.

For each security: RS-Ratio is a rolling Z-score of the price ratio
(security/benchmark), centered at 100. RS-Momentum is a rolling Z-score of
the rate-of-change of RS-Ratio, also centered at 100. Both axes fluctuate
around 100 — above 100 = outperforming/accelerating, below = the opposite.
The four quadrants (Leading/Weakening/Lagging/Improving) split at (100,100).
"""

import numpy as np
import pandas as pd


def compute_rrg(closes, bench, *, tail_periods=5, norm_window=12,
                 roc_window=5, resample_rule="W-FRI"):
    """
    closes: dict {name: pd.Series of daily Close, tz-naive DatetimeIndex}
    bench:  pd.Series of daily Close (benchmark), tz-naive DatetimeIndex

    resample_rule: a pandas resample rule (e.g. "W-FRI" for weekly), or None
    to use raw daily bars unresampled (daily mode). norm_window/roc_window
    are in units of the resulting bar (weeks if resampled, trading days if
    not) — callers should pick window sizes appropriate to the interval
    (e.g. weekly: norm_window=12, roc_window=5; daily: norm_window=20,
    roc_window=5 — roughly 1 month / 1 week of trading days).

    Returns:
    {
      "as_of": "YYYY-MM-DD" | None,
      "benchmark": "NIFTY 200",
      "params": {tail_periods, norm_window, roc_window, resample_rule},
      "sectors": [
        {"name": str, "tail": [{"date": "YYYY-MM-DD", "x": float, "y": float}, ...],
         "current": {"x": float, "y": float, "quadrant": str}, "full_tail": bool},
        ...
      ],
      "skipped": [str, ...],
    }
    """
    min_bars = 2 * norm_window + roc_window  # smallest window that yields 1 valid point

    sectors, skipped, as_of = [], [], None

    for name, c in closes.items():
        b_aligned = bench.reindex(c.index, method="ffill")
        if resample_rule:
            cw = c.resample(resample_rule).last()
            bw_aligned = b_aligned.resample(resample_rule).last()
        else:
            cw, bw_aligned = c, b_aligned

        rs = (cw / bw_aligned).dropna()
        if len(rs) < min_bars:
            skipped.append(name)
            continue

        m = rs.rolling(norm_window, min_periods=norm_window).mean()
        sd = rs.rolling(norm_window, min_periods=norm_window).std().replace(0, 1e-10)
        rs_ratio = 100 + (rs - m) / sd

        roc = rs_ratio / rs_ratio.shift(roc_window) - 1
        rm = roc.rolling(norm_window, min_periods=norm_window).mean()
        rsd = roc.rolling(norm_window, min_periods=norm_window).std().replace(0, 1e-10)
        rs_mom = 100 + (roc - rm) / rsd

        df = pd.concat([rs_ratio.rename("x"), rs_mom.rename("y")], axis=1).dropna()
        if df.empty:
            skipped.append(name)
            continue

        tail_df = df.iloc[-tail_periods:]
        tail = [{"date": d.date().isoformat(), "x": float(row.x), "y": float(row.y)}
                for d, row in tail_df.iterrows()]
        last = tail_df.iloc[-1]
        x, y = float(last.x), float(last.y)
        quadrant = ("leading" if x >= 100 and y >= 100 else
                    "weakening" if x >= 100 else
                    "improving" if y >= 100 else
                    "lagging")

        sectors.append({
            "name": name,
            "tail": tail,
            "current": {"x": x, "y": y, "quadrant": quadrant},
            "full_tail": len(tail_df) >= tail_periods,
        })
        last_date = tail_df.index[-1].date().isoformat()
        if as_of is None or last_date > as_of:
            as_of = last_date

    return {
        "as_of": as_of,
        "benchmark": "NIFTY 200",
        "params": {"tail_periods": tail_periods, "norm_window": norm_window,
                    "roc_window": roc_window, "resample_rule": resample_rule},
        "sectors": sectors,
        "skipped": skipped,
    }


# ── Sector → constituent stock resolution (for drill-down) ────────────────────
# 4 sectors aren't in STOCK_SECTOR_MAP (it covers 17 of 19) — those fall back
# to synth_sector_index.py's constituent sources (niftyindices CSV / hardcoded
# lists), the same lookup already used to build their synthetic index series.
_NOT_IN_STOCK_SECTOR_MAP = {
    "NIFTY OIL & GAS": "NIFTY_OIL_GAS",
    "NIFTY INDIA DEFENCE": "NIFTY_INDIA_DEFENCE",
    "NIFTY INDIA MFG": "NIFTY_INDIA_MFG",
    "NIFTY CONSUMER DURABLES": "NIFTY_CONSUMER_DURABLES",
}


def get_sector_constituents(sector_name):
    """Return the list of constituent stock tickers for a sector name (e.g.
    "NIFTY METAL"). Tries sector_mapping.STOCK_SECTOR_MAP first (17 of 19
    sectors, no I/O); falls back to synth_sector_index.fetch_constituents()
    for the 4 sectors not in that map (may hit niftyindices.com for those)."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sector_mapping import STOCK_SECTOR_MAP
    syms = [t for t, s in STOCK_SECTOR_MAP.items() if s == sector_name]
    if syms:
        return syms
    if sector_name in _NOT_IN_STOCK_SECTOR_MAP:
        from data.synth_sector_index import fetch_constituents
        return fetch_constituents(_NOT_IN_STOCK_SECTOR_MAP[sector_name])
    return []
