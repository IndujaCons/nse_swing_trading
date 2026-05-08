"""
score_live_sectors() — today's NSE sectoral-index Z-score ranking.

Mirrors data/etf_core_zscore_backtest.py:score_live() but operates on
the Phase 1 frozen SECTOR_UNIVERSE (yfinance series + 5 synthetic .pkl
fallbacks for HEALTHCARE / OIL & GAS / DEFENCE / INDIA MFG / CONSUMER
DURABLES, built by data/synth_sector_index.py).

Used by the Mom20 basket flow to identify today's top-5 sectors for ETF
top-up (data/mom20_basket.py) and by anything else that needs a quick
sector-Z snapshot without running the full backtest.
"""

import glob
import os
import pickle
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

from data.sector_zscore_backtest import SECTOR_UNIVERSE  # 19-sector frozen list

LONG_PD  = 252   # 12m
SHORT_PD = 63    # 3m
W12, W3  = 0.50, 0.50

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache")


def _fetch_yf(yf_sym: str, start: str, end: str) -> pd.Series:
    try:
        df = yf.download(yf_sym, start=start, end=end,
                         progress=False, auto_adjust=True, timeout=30)
        if df.empty:
            return pd.Series(dtype=float)
        s = df["Close"].squeeze().dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s
    except Exception:
        return pd.Series(dtype=float)


def _load_synth(synth_name: str) -> pd.Series:
    """Load latest synth_<name>_*.pkl. Returns empty Series if missing."""
    pattern = os.path.join(CACHE_DIR, f"synth_{synth_name}_*.pkl")
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.Series(dtype=float)
    with open(files[-1], "rb") as f:
        s = pickle.load(f)
    if hasattr(s.index, "tz") and s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s


def _fetch_live_sector_prices() -> dict:
    """Pull each sector's close series. Synth-prefixed entries load from
    pkl; yfinance entries fetch in ONE bulk call with a hard 30s timeout
    so a single hung Yahoo response can't freeze /api/sector-ranking
    (the freeze used to surface as 'Error: Unexpected token <' in the UI
    because the Flask worker was 504'ing back to the proxy)."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
    start = (pd.Timestamp.today() - pd.Timedelta(days=400)).strftime('%Y-%m-%d')
    end   = (date.today() + timedelta(days=1)).isoformat()

    closes = {}

    # Synth fallbacks first — fast pkl reads, no network.
    yf_pairs = []
    for sec, source in SECTOR_UNIVERSE:
        if source.startswith("synth:"):
            synth_name = source.split(":", 1)[1].lower()
            closes[sec] = _load_synth(synth_name)
        else:
            yf_pairs.append((sec, source))

    # Bulk yfinance for the remaining sectors, hard-capped at 30s.
    if yf_pairs:
        yf_syms = [src for _, src in yf_pairs]
        bulk_df = None
        try:
            def _bulk():
                return yf.download(yf_syms, start=start, end=end, progress=False,
                                   auto_adjust=True, group_by='ticker',
                                   threads=True, timeout=15)
            with ThreadPoolExecutor(max_workers=1) as _e:
                bulk_df = _e.submit(_bulk).result(timeout=30)
        except FutTimeout:
            print("[score_live_sectors] bulk yfinance fetch timed out after 30s")
        except Exception as e:
            print(f"[score_live_sectors] bulk fetch failed: {e}")

        for sec, sym in yf_pairs:
            s = pd.Series(dtype=float)
            try:
                if bulk_df is not None and not bulk_df.empty:
                    if isinstance(bulk_df.columns, pd.MultiIndex) and sym in bulk_df.columns.get_level_values(0):
                        s = bulk_df[sym]["Close"].dropna()
                    elif len(yf_syms) == 1:
                        s = bulk_df["Close"].squeeze().dropna()
                if len(s) > 0 and hasattr(s.index, 'tz') and s.index.tz is not None:
                    s.index = s.index.tz_localize(None)
            except Exception:
                pass
            closes[sec] = s

    return closes


def score_live_sectors() -> list[dict]:
    """Return today's sector Z-score ranking. Each entry:
        {symbol, score, rank, ret_12m, ret_3m, beta, price}

    Same scoring math as Phase 1 backtest (mirrored from
    etf_core_zscore_backtest.py:344–376) — MR_12, MR_3, Z-score
    cross-sectionally, weighted blend, normalised score.
    """
    closes = _fetch_live_sector_prices()

    # Need at least LONG_PD+10 days of data per sector to score
    eligible_syms = [s for s, c in closes.items() if len(c) >= LONG_PD + 10]
    if len(eligible_syms) < 3:
        return []

    today = max(c.index.max() for c in closes.values() if len(c) > 0)

    score_data = {}
    for sym in eligible_syms:
        s = closes[sym]
        idx = s.index.get_indexer([today], method='ffill')[0]
        if idx < LONG_PD:
            continue
        try:
            px  = float(s.iloc[idx])
            r12 = float(s.iloc[idx] / s.iloc[idx - LONG_PD] - 1)
            r3  = float(s.iloc[idx] / s.iloc[idx - SHORT_PD] - 1)
            dr  = s.pct_change()
            v   = float(dr.iloc[max(0, idx - LONG_PD): idx + 1].std() * np.sqrt(LONG_PD))
            if v == 0:
                continue
            score_data[sym] = {
                "px": px, "r12": r12, "r3": r3, "v": v,
                "mr12": r12 / v, "mr3": r3 / v,
            }
        except Exception:
            continue

    if len(score_data) < 3:
        return []

    eligible = list(score_data.keys())
    arr12 = np.array([score_data[s]["mr12"] for s in eligible])
    arr3  = np.array([score_data[s]["mr3"]  for s in eligible])

    def _z(a):
        mu, sd = a.mean(), a.std()
        return (a - mu) / sd if sd > 0 else np.zeros_like(a)

    waz = W12 * _z(arr12) + W3 * _z(arr3)
    sc  = np.where(waz >= 0, 1 + waz, 1.0 / (1 - waz))

    results = []
    for i, sym in enumerate(eligible):
        d = score_data[sym]
        results.append({
            "symbol":  sym,
            "score":   round(float(sc[i]), 3),
            "price":   round(d["px"], 2),
            "ret_12m": round(d["r12"] * 100, 1),
            "ret_3m":  round(d["r3"]  * 100, 1),
            "beta":    None,   # not needed for ranking; Phase 1 doesn't beta-cap sectors
        })
    results.sort(key=lambda x: -x["score"])
    for i, r in enumerate(results):
        r["rank"] = i + 1
    return results


if __name__ == "__main__":
    ranking = score_live_sectors()
    print(f"Sector Z-score ranking ({len(ranking)} sectors):\n")
    for r in ranking:
        print(f"  #{r['rank']:>2}  {r['symbol']:<26} {r['score']:>6.3f}  "
              f"12m {r['ret_12m']:+6.1f}%  3m {r['ret_3m']:+6.1f}%")
