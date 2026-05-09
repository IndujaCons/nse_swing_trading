"""
Live Signals Engine — scans for Strategy J, T, R, and MW entry signals in real-time,
manages positions (with 3-stage exits for T and R), and checks exit conditions.
"""

import csv
import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import yfinance as yf

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import (
    LIVE_SIGNALS_CACHE_FILE, LIVE_POSITIONS_FILE, LIVE_SIGNALS_HISTORY_FILE,
    get_cache_ttl, load_config
)
from sector_mapping import STOCK_SECTOR_MAP

# Per-strategy compute functions (Phase 2 refactor — see data/strategies/).
from data.strategies.mom20    import compute_mom20_features
from data.strategies.alpha20  import compute_alpha20_features
from data.strategies.rs_ibd   import compute_rs_ibd_features
from data.strategies.rs63     import compute_rs63_signal
from data.strategies.j        import compute_j_signal
from data.strategies.t        import compute_t_signal
from data.strategies.r        import compute_r_signal
from data.strategies.mw       import compute_mw_signal


def _load_latest_pit_constituents(pit_filename: str):
    """Return the most recent constituent list from an NSE PIT JSON file.
    Raises if missing/unreadable — the live scanner has no business falling
    back to a stale hardcoded list.
    """
    path = os.path.join(os.path.dirname(__file__), "..", "nse_const", pit_filename)
    with open(path) as f:
        pit = json.load(f)
    return list(pit[max(pit.keys())])


def get_scan_tickers(universe: int):
    """Return the live-scan ticker universe for a given universe size.

    Sourced from the latest entry in the NSE PIT JSONs so reconstitutions
    flow through automatically. Used by both LiveSignalsEngine.scan_entry_signals
    and the /api/live-signals/refresh endpoint (Zerodha LTP fetch).
    """
    if universe <= 50:
        return NIFTY_50_TICKERS
    if universe <= 100:
        return NIFTY_50_TICKERS + NIFTY_NEXT50_TICKERS
    if universe <= 200:
        return _load_latest_pit_constituents("nifty200_pit.json")
    # >200: alphabetical first 150 of current Nifty 500 minus current Nifty 100
    n500 = _load_latest_pit_constituents("nifty500_pit.json")
    nifty100_set = set(NIFTY_50_TICKERS + NIFTY_NEXT50_TICKERS)
    return [t for t in n500 if t not in nifty100_set][:150]

# Sectoral Indices (Yahoo Finance symbols) for RS momentum
SECTORAL_INDICES = {
    "NIFTY PVT BANK": "NIFTY_PVT_BANK.NS",
    "NIFTY PSU BANK": "^CNXPSUBANK",
    "NIFTY IT": "^CNXIT",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY METAL": "^CNXMETAL",
    "NIFTY REALTY": "^CNXREALTY",
    "NIFTY ENERGY": "^CNXENERGY",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY MEDIA": "^CNXMEDIA",
    "NIFTY INFRA": "^CNXINFRA",
    "NIFTY FIN SERVICE": "NIFTY_FIN_SERVICE.NS",
    "NIFTY COMMODITIES": "^CNXCMDT",
    "NIFTY CONSUMPTION": "^CNXCONSUM",
    "NIFTY HEALTHCARE": "^CNXPHARMA",
    "NIFTY MNC": "^CNXMNC",
    "NIFTY PSE": "^CNXPSE",
}

# Actual Nifty 50 constituents
NIFTY_50_TICKERS = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BPCL",
    "BHARTIARTL", "BRITANNIA", "CIPLA", "COALINDIA", "DRREDDY",
    "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH", "HDFCBANK",
    "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK",
    "ITC", "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK",
    "LT", "M&M", "MARUTI", "NTPC", "NESTLEIND",
    "ONGC", "PIDILITIND", "POWERGRID", "RELIANCE", "SBILIFE",
    "SBIN", "SUNPHARMA", "TCS", "TATACONSUM",
    "TATASTEEL", "TECHM", "TITAN", "ULTRACEMCO", "WIPRO",
]

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

# Nifty 200 / 500 universes are sourced from nse_const/nifty{200,500}_pit.json
# via get_scan_tickers() above — no hardcoded list, so NSE reconstitutions
# flow through automatically (see commit removing NIFTY_200_NEXT100_TICKERS).


def _calculate_rsi_series(closes: pd.Series, period: int) -> pd.Series:
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _find_swing_lows(lows: pd.Series, left: int = 5, right: int = 3) -> list:
    """Find swing low positions: low[i] is min of surrounding window."""
    result = []
    vals = lows.values
    n = len(vals)
    for i in range(left, n - right):
        window = vals[i - left: i + right + 1]
        if vals[i] == np.min(window):
            result.append((i, float(vals[i])))
    return result


def _rsi_near_swing(rsi14_vals, idx, window=3):
    """Get min RSI within ±window bars of a swing low index.

    Price low and RSI low often don't land on the same bar —
    use the lowest RSI in the zone to match how traders read charts.
    """
    start = max(0, idx - window)
    end = min(len(rsi14_vals), idx + window + 1)
    chunk = rsi14_vals[start:end]
    valid = [v for v in chunk if not np.isnan(v)]
    return min(valid) if valid else rsi14_vals[idx]


def _detect_bullish_divergence(lows_vals, rsi14_vals, i, swing_lows,
                               max_lookback=50, min_sep=5,
                               rsi_threshold=40, min_rsi_divergence=3,
                               min_price_drop=0.0, max_curr_age=None):
    """Check for bullish RSI divergence at bar i.

    Uses min RSI within ±3 bars of each swing low to match visual chart reading.
    If max_curr_age set, current swing low must be within that many bars of i.
    """
    recent = [(idx, val) for idx, val in swing_lows
              if idx <= i and i - idx <= max_lookback]
    if len(recent) < 2:
        return False, None, None
    for k in range(len(recent) - 1, 0, -1):
        curr_idx, curr_low = recent[k]
        prev_idx, prev_low = recent[k - 1]
        if max_curr_age is not None and i - curr_idx > max_curr_age:
            continue
        if curr_idx - prev_idx < min_sep:
            continue
        if curr_low >= prev_low * (1 - min_price_drop):
            continue
        curr_rsi = _rsi_near_swing(rsi14_vals, curr_idx)
        prev_rsi = _rsi_near_swing(rsi14_vals, prev_idx)
        if np.isnan(curr_rsi) or np.isnan(prev_rsi):
            continue
        if curr_rsi - prev_rsi < min_rsi_divergence:
            continue
        if curr_rsi >= rsi_threshold:
            continue
        return True, curr_low, curr_rsi
    return False, None, None


def _detect_hidden_bullish_divergence(lows_vals, rsi14_vals, i, swing_lows,
                                       max_lookback=50, min_sep=5,
                                       rsi_threshold=60, min_rsi_divergence=5,
                                       max_curr_age=None):
    """Check for hidden bullish RSI divergence at bar i.

    Hidden bullish divergence (uptrend continuation):
    - Price: current swing low > previous swing low (higher low)
    - RSI(14): current < previous - min_rsi_divergence (lower low in RSI)
    - RSI(14) < rsi_threshold at current swing low
    If max_curr_age set, current swing low must be within that many bars of i.

    Returns (True, swing_low_val, rsi_at_low) or (False, None, None).
    """
    recent = [(idx, val) for idx, val in swing_lows
              if idx <= i and i - idx <= max_lookback]
    if len(recent) < 2:
        return False, None, None
    for k in range(len(recent) - 1, 0, -1):
        curr_idx, curr_low = recent[k]
        prev_idx, prev_low = recent[k - 1]
        if max_curr_age is not None and i - curr_idx > max_curr_age:
            continue
        if curr_idx - prev_idx < min_sep:
            continue
        if curr_low <= prev_low:
            continue
        curr_rsi = _rsi_near_swing(rsi14_vals, curr_idx)
        prev_rsi = _rsi_near_swing(rsi14_vals, prev_idx)
        if np.isnan(curr_rsi) or np.isnan(prev_rsi):
            continue
        if prev_rsi - curr_rsi < min_rsi_divergence:
            continue
        if curr_rsi >= rsi_threshold:
            continue
        return True, curr_low, curr_rsi
    return False, None, None


def _calculate_adx_series(highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14):
    """Calculate ADX, +DI, and -DI series using Wilder's smoothing."""
    prev_high = highs.shift(1)
    prev_low = lows.shift(1)
    prev_close = closes.shift(1)

    tr1 = highs - lows
    tr2 = (highs - prev_close).abs()
    tr3 = (lows - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = highs - prev_high
    down_move = prev_low - lows
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=highs.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=highs.index)

    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    smooth_plus_dm = plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    smooth_minus_dm = minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    plus_di = 100 * smooth_plus_dm / atr
    minus_di = 100 * smooth_minus_dm / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(window=period, min_periods=period).mean()

    return adx, plus_di, minus_di


class LiveSignalsEngine:
    """Scans J/T/R/MW entry signals, manages positions with 3-stage exits, checks exit conditions."""

    def __init__(self, user_id=None):
        self.user_id = user_id
        self.cache_file = LIVE_SIGNALS_CACHE_FILE
        if user_id:
            data_dir = os.path.dirname(LIVE_POSITIONS_FILE)
            self.positions_file = os.path.join(data_dir, f"live_positions_{user_id}.json")
        else:
            self.positions_file = LIVE_POSITIONS_FILE
        self._ensure_dirs()

    def _ensure_dirs(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.positions_file), exist_ok=True)

    # ==================== Sector RS Momentum ====================

    def _compute_sector_momentum(self, end_date):
        """Compute sector RS vs Nifty 50 with multi-day momentum.

        Returns dict: sector_name -> {rs_today, rs_5d_ago, rs_10d_ago, rs_20d_ago,
                                       delta_5d, delta_10d, delta_20d, momentum_score}
        momentum_score: weighted sum of deltas — captures sectors gaining RS fast.
        """
        try:
            nifty = yf.Ticker("^NSEI").history(
                start=end_date - timedelta(days=120), end=end_date)
            if nifty.empty or len(nifty) < 30:
                return {}
            nifty_closes = nifty["Close"]
        except Exception:
            return {}

        sector_momentum = {}
        rs_period = 20  # 20-day return for RS calculation

        for sector_name, symbol in SECTORAL_INDICES.items():
            try:
                sec_data = yf.Ticker(symbol).history(
                    start=end_date - timedelta(days=120), end=end_date)
                if sec_data.empty or len(sec_data) < 30:
                    continue
                sec_closes = sec_data["Close"]

                # Align dates
                common = nifty_closes.index.intersection(sec_closes.index)
                if len(common) < rs_period + 20:
                    continue
                n_aligned = nifty_closes.reindex(common)
                s_aligned = sec_closes.reindex(common)

                # Compute rolling RS (sector return - nifty return) at multiple points
                def rs_at(idx):
                    if idx < rs_period:
                        return None
                    n_ret = (float(n_aligned.iloc[idx]) / float(n_aligned.iloc[idx - rs_period]) - 1) * 100
                    s_ret = (float(s_aligned.iloc[idx]) / float(s_aligned.iloc[idx - rs_period]) - 1) * 100
                    return round(s_ret - n_ret, 2)

                last = len(common) - 1
                rs_today = rs_at(last)
                rs_5d = rs_at(max(rs_period, last - 5))
                rs_10d = rs_at(max(rs_period, last - 10))
                rs_20d = rs_at(max(rs_period, last - 20))

                if rs_today is None:
                    continue

                delta_5d = round(rs_today - rs_5d, 2) if rs_5d is not None else 0
                delta_10d = round(rs_today - rs_10d, 2) if rs_10d is not None else 0
                delta_20d = round(rs_today - rs_20d, 2) if rs_20d is not None else 0

                # Momentum score: weighted — recent change matters more
                # Positive = sector RS is rising (from -10 to -3 = +7 = good)
                momentum_score = round(delta_5d * 3 + delta_10d * 2 + delta_20d * 1, 2)

                sector_momentum[sector_name] = {
                    "rs": rs_today,
                    "rs_5d_ago": rs_5d,
                    "rs_10d_ago": rs_10d,
                    "rs_20d_ago": rs_20d,
                    "delta_5d": delta_5d,
                    "delta_10d": delta_10d,
                    "delta_20d": delta_20d,
                    "momentum": momentum_score,
                }
            except Exception:
                continue

        return sector_momentum

    # ==================== Entry Signal Scanning ====================

    # ── Phase 3 helpers — disk-cached baseline + intraday patch ───────────────

    _BASELINE_MAX_AGE_HOURS = 48  # accept yesterday's snapshot if today's hasn't run

    def _try_load_cached_baseline(self, tickers):
        """Look for a recent data_store/cache/n200_baseline_<date>.pkl.
        Returns the loaded payload dict if usable, else None (signals the
        caller to fall back to a full fetch).

        Constraints for the cache to be 'usable':
          • File age <= _BASELINE_MAX_AGE_HOURS (default 48h)
          • The cached `tickers` list must be a superset of the requested set
            (otherwise an N200 reconstitution would silently drop names).
        """
        import glob, pickle, time as _t
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "data_store", "cache")
        files = sorted(glob.glob(os.path.join(cache_dir, "n200_baseline_*.pkl")))
        if not files:
            return None
        latest = files[-1]
        try:
            age_h = (_t.time() - os.path.getmtime(latest)) / 3600.0
            if age_h > self._BASELINE_MAX_AGE_HOURS:
                print(f"[live_signals] cached baseline {os.path.basename(latest)} "
                      f"is {age_h:.1f}h old — falling back to full fetch")
                return None
            with open(latest, "rb") as f:
                payload = pickle.load(f)
            cached_tickers = set(payload.get("tickers", []))
            if not set(tickers).issubset(cached_tickers):
                missing = set(tickers) - cached_tickers
                print(f"[live_signals] cached baseline missing {len(missing)} "
                      f"tickers (e.g. {list(missing)[:3]}) — falling back to full fetch")
                return None
            print(f"[live_signals] using cached baseline "
                  f"{os.path.basename(latest)} (age {age_h:.1f}h)")
            return payload
        except Exception as e:
            print(f"[live_signals] failed to load cached baseline {latest}: {e}")
            return None

    def _apply_intraday_patch(self, cached, tickers, end_date, progress_callback=None):
        """Take a loaded baseline and patch each ticker's latest bar with
        today's intraday close from a single bulk yf.download(period='1d').
        Returns (nifty_raw, bench_raw, bulk_data) — same contract as the
        slow path so the caller is unaware of which path served the data.

        If the intraday fetch fails, we still return the cached baseline
        as-is — slightly stale data is better than no data."""
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
        import time as _t

        nifty_raw = cached.get("nifty_raw", pd.DataFrame())
        bench_raw = cached.get("bench_raw", pd.DataFrame())
        bulk_data = dict(cached.get("bulk_data", {}))  # shallow copy

        if progress_callback:
            progress_callback(0, len(tickers), "patching latest bar…")

        # One bulk fetch for everything (200 N200 + 2 indices). period='1d'
        # returns today's incomplete daily bar for each instrument.
        all_syms = ["^NSEI", "^CNX200"] + [f"{t}.NS" for t in tickers]
        intraday = None
        t0 = _t.time()
        try:
            with ThreadPoolExecutor(max_workers=1) as _ex:
                intraday = _ex.submit(
                    lambda: yf.download(all_syms, period="1d", progress=False,
                                        auto_adjust=True, group_by="ticker",
                                        threads=True, timeout=15)
                ).result(timeout=30)
        except FutTimeout:
            print("[live_signals] intraday patch fetch timed out — serving cached")
            return nifty_raw, bench_raw, bulk_data
        except Exception as e:
            print(f"[live_signals] intraday patch failed: {e} — serving cached")
            return nifty_raw, bench_raw, bulk_data

        if intraday is None or intraday.empty:
            return nifty_raw, bench_raw, bulk_data

        def _patch(df, sym):
            """Append/replace today's bar in df with the intraday's bar for sym."""
            try:
                if isinstance(intraday.columns, pd.MultiIndex):
                    if sym not in intraday.columns.get_level_values(0):
                        return df
                    today_bar = intraday[sym].dropna(how="all")
                else:
                    today_bar = intraday.dropna(how="all")
                if today_bar.empty:
                    return df
                # Strip tz to match cached series
                if hasattr(today_bar.index, "tz") and today_bar.index.tz is not None:
                    today_bar = today_bar.copy()
                    today_bar.index = today_bar.index.tz_localize(None)
                # Use the LAST row's date as today's bar
                today_idx = today_bar.index[-1].normalize()
                row = today_bar.iloc[-1]
                # Build a one-row frame with the same columns as df
                new_row = pd.DataFrame(
                    {col: [float(row.get(col, row.get("Close", float('nan'))))]
                     for col in df.columns},
                    index=[today_idx])
                # Drop any existing today bar then append
                df = df[df.index < today_idx]
                return pd.concat([df, new_row])
            except Exception:
                return df

        if not nifty_raw.empty:
            nifty_raw = _patch(nifty_raw, "^NSEI")
        if not bench_raw.empty:
            bench_raw = _patch(bench_raw, "^CNX200")
        for ticker in tickers:
            yf_sym = f"{ticker}.NS"
            if yf_sym in bulk_data:
                bulk_data[yf_sym] = _patch(bulk_data[yf_sym], yf_sym)

        elapsed = _t.time() - t0
        print(f"[live_signals] intraday patch applied in {elapsed:.1f}s")
        return nifty_raw, bench_raw, bulk_data


    def _fetch_baseline(self, tickers, daily_start, end_date, progress_callback=None):
        """Fetch all baseline market data needed for signal computation.

        This is the *only* place yfinance is hit during a scan. Returns
        (nifty_raw, bench_raw, bulk_data) where:
          • nifty_raw — Nifty 50 daily OHLCV (regime + Alpha20/Mom20 β reference)
          • bench_raw — Nifty 200 daily OHLCV (RS calculations + Mom20 regime)
          • bulk_data — dict {yf_sym: per-ticker DataFrame} for the N200 universe

        Fast path (Phase 3 — intraday delta):
          • If data_store/cache/n200_baseline_<date>.pkl exists and is recent
            (<48h), load it and patch the latest bar of each series with a
            small bulk yf.download(period='1d'). ~5-10s instead of 30-60s.
          • The pkl is built by the daily cron scripts/refresh_n200_baseline.sh
            running at 08:00 + 16:00 IST.

        Slow path (fallback when the pkl is missing/stale):
          • Indices fetched via yf.Ticker().history() (single-symbol).
          • N200 universe fetched in 15-ticker chunks via yf.download() with
            group_by='ticker' + threads=True. Each chunk capped at 25s by a
            ThreadPoolExecutor so a hung chunk doesn't freeze the whole scan.
          • Tickers missing from bulk_data fall through to the caller's
            _slice_daily() per-ticker fallback (10s cap each).
        """
        # Fast path — try the disk-cached baseline + intraday-patch first.
        cached = self._try_load_cached_baseline(tickers)
        if cached is not None:
            return self._apply_intraday_patch(cached, tickers, end_date,
                                              progress_callback)

        # Slow path — full fetch (existing behaviour).
        # Indices — Nifty 50 (regime, β) and Nifty 200 (RS, Mom20 regime).
        try:
            nifty_raw = yf.Ticker("^NSEI").history(start=daily_start, end=end_date)
        except Exception:
            nifty_raw = pd.DataFrame()
        try:
            bench_raw = yf.Ticker("^CNX200").history(start=daily_start, end=end_date)
        except Exception:
            bench_raw = nifty_raw  # fallback to Nifty 50

        # Chunked N200 bulk fetch.
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
        CHUNK_SIZE = 15
        CHUNK_TIMEOUT = 25  # seconds
        total = len(tickers)
        yf_tickers = [f"{t}.NS" for t in tickers]
        bulk_data = {}

        def _fetch_chunk(chunk_syms):
            return yf.download(chunk_syms, start=daily_start, end=end_date,
                               progress=False, auto_adjust=True,
                               group_by='ticker', threads=True, timeout=15)

        for chunk_start in range(0, len(yf_tickers), CHUNK_SIZE):
            chunk = yf_tickers[chunk_start:chunk_start + CHUNK_SIZE]
            if progress_callback:
                progress_callback(chunk_start, total, "fetching prices…")
            chunk_df = None
            try:
                with ThreadPoolExecutor(max_workers=1) as _ex:
                    chunk_df = _ex.submit(_fetch_chunk, chunk).result(timeout=CHUNK_TIMEOUT)
            except FutTimeout:
                print(f"[live_signals] chunk {chunk_start}-{chunk_start+len(chunk)} "
                      f"timed out after {CHUNK_TIMEOUT}s — falling back to per-ticker for {chunk}")
                continue
            except Exception as e:
                print(f"[live_signals] chunk {chunk_start}-{chunk_start+len(chunk)} fetch failed: {e}")
                continue
            if chunk_df is None or chunk_df.empty:
                continue
            try:
                if isinstance(chunk_df.columns, pd.MultiIndex):
                    for sym in chunk:
                        if sym in chunk_df.columns.get_level_values(0):
                            sub = chunk_df[sym].dropna(how='all')
                            if len(sub) > 0:
                                bulk_data[sym] = sub
                elif len(chunk) == 1:
                    sub = chunk_df.dropna(how='all')
                    if len(sub) > 0:
                        bulk_data[chunk[0]] = sub
            except (KeyError, AttributeError):
                continue

        return nifty_raw, bench_raw, bulk_data


    def scan_entry_signals(self, force_refresh=False, progress_callback=None, scan_date=None, ltp_map=None):
        """Scan for J, T, and R entry signals across Nifty 50/100.

        scan_date: optional "YYYY-MM-DD" string to scan a historical date.
        ltp_map: optional dict of {ticker: price} from Zerodha for real-time prices.
                 When provided, overlays today's close with live LTP.
        """
        is_historical = scan_date is not None

        if not force_refresh and not is_historical:
            cached = self._load_cache()
            if cached is not None:
                return cached

        config = load_config()
        universe = config.get("live_signals_universe", 50)
        tickers = get_scan_tickers(universe)

        total = len(tickers)
        if is_historical:
            end_date = datetime.strptime(scan_date, "%Y-%m-%d") + timedelta(days=1)
        else:
            end_date = datetime.now() + timedelta(days=1)
        daily_start = end_date - timedelta(days=500)

        # ── Fetch baseline (delegated to _fetch_baseline for Phase 3) ──────────
        # Returns Nifty 50 + Nifty 200 indices + dict of N200 ticker DataFrames.
        # The per-ticker fallback for tickers missing from the bulk fetch is
        # handled by _slice_daily below.
        nifty_raw, bench_raw, bulk_data = self._fetch_baseline(
            tickers, daily_start, end_date, progress_callback)

        # Overlay Zerodha LTP on benchmark indices
        if ltp_map:
            for df, ltp_key in [(nifty_raw, "^NSEI"), (bench_raw, "^CNX200")]:
                if not df.empty and ltp_key in ltp_map:
                    li = len(df) - 1
                    lp = ltp_map[ltp_key]
                    df.iloc[li, df.columns.get_loc("Close")] = lp
                    if lp > df.iloc[li]["High"]:
                        df.iloc[li, df.columns.get_loc("High")] = lp
                    if lp < df.iloc[li]["Low"]:
                        df.iloc[li, df.columns.get_loc("Low")] = lp

        # Compute benchmark 123-day return for RS
        bench_ret123_val = 0.0
        bench_ret123_28ago = 0.0
        if not bench_raw.empty and len(bench_raw) > 160:
            bc = bench_raw["Close"]
            bi = len(bc) - 1
            if bi >= 123 and float(bc.iloc[bi - 123]) > 0:
                bench_ret123_val = (float(bc.iloc[bi]) / float(bc.iloc[bi - 123]) - 1) * 100
            if bi >= 123 + 28 and float(bc.iloc[bi - 28 - 123]) > 0:
                bench_ret123_28ago = (float(bc.iloc[bi - 28]) / float(bc.iloc[bi - 28 - 123]) - 1) * 100

        # Mom20 regime: Nifty200 >= SMA200 → ON (hold all, no rebalancing when OFF)
        mom20_regime_on = False
        if not bench_raw.empty and len(bench_raw) >= 200:
            n200_close = float(bench_raw["Close"].iloc[-1])
            n200_sma200 = float(bench_raw["Close"].rolling(200).mean().iloc[-1])
            if not pd.isna(n200_sma200) and n200_close >= n200_sma200:
                mom20_regime_on = True

        # Nifty regime: early_10w_2 mode (OFF below 20w, early ON: above 10w + 10w rising 2wk)
        nifty_regime_on = False
        if not nifty_raw.empty:
            nifty_weekly = nifty_raw["Close"].resample("W-FRI").last().dropna()
            if len(nifty_weekly) >= 20:
                nifty_20w_ema = nifty_weekly.ewm(span=20, adjust=False).mean()
                nifty_10w_ema = nifty_weekly.ewm(span=10, adjust=False).mean()
                last_nifty_close = float(nifty_raw["Close"].iloc[-1])
                nifty_20w_daily = nifty_20w_ema.reindex(nifty_raw.index, method="ffill")
                last_nifty_20w = float(nifty_20w_daily.iloc[-1])
                if last_nifty_close >= last_nifty_20w:
                    nifty_regime_on = True
                else:
                    # Early ON: price > 10w EMA and 10w EMA rising for 2 consecutive weeks
                    nifty_10w_daily = nifty_10w_ema.reindex(nifty_raw.index, method="ffill")
                    last_nifty_10w = float(nifty_10w_daily.iloc[-1])
                    if last_nifty_close > last_nifty_10w and len(nifty_10w_ema) >= 3:
                        e10_now = float(nifty_10w_ema.iloc[-1])
                        e10_1w = float(nifty_10w_ema.iloc[-2])
                        e10_2w = float(nifty_10w_ema.iloc[-3])
                        if e10_now > e10_1w > e10_2w:
                            nifty_regime_on = True

        # Precompute Nifty 50 daily returns (date-indexed for alignment)
        n50_ret_series = None
        n50_market_rets = None
        n50_rm = 0.0
        n50_var = 0.0
        if not nifty_raw.empty and len(nifty_raw) >= 253:
            n50_close_series = nifty_raw["Close"].astype(float)
            # Strip tz so this index aligns with bulk-fetched (tz-naive) stock series
            if n50_close_series.index.tz is not None:
                n50_close_series.index = n50_close_series.index.tz_localize(None)
            n50_ret_series = n50_close_series.pct_change().iloc[-252:]  # date-indexed returns
            n50_market_rets = n50_ret_series.values
            n50_rm = float(np.mean(n50_market_rets))
            n50_var = float(np.var(n50_market_rets))

        j_signals = []
        t_signals = []
        r_signals = []
        rw_signals = []
        mw_signals = []
        rs_signals = []
        rs_ibd_candidates = []  # collect IBD weighted scores for ranking
        rs_ibd_history = {}  # ticker -> list of weighted scores for last 5 days
        mom20_raw = []  # collect momentum data for Mom20/Mom15 scoring after loop
        alpha20_raw = []  # collect data for Alpha20 CAPM scoring after loop
        rs63_signals = []  # RS63 satellite signals
        actual_date = None  # Track actual last trading date from data

        # ── Per-ticker slice helper (fallback for misses in bulk_data) ────────
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout

        def _slice_daily(yf_sym):
            """Pull a single ticker's OHLCV from the bulk dict; fall back to a
            per-ticker fetch if the chunk missed it. Always returns tz-naive
            so its index intersects cleanly with the (now tz-naive) Nifty 50
            reference series — Mom20 beta / Alpha20 / overflow all depend on
            this alignment. Per-ticker fallback is capped at 10s via
            ThreadPoolExecutor so a single hung Yahoo response doesn't stall
            the per-ticker compute loop."""
            df = bulk_data.get(yf_sym, pd.DataFrame())
            if df.empty or len(df) < 210:
                try:
                    with ThreadPoolExecutor(max_workers=1) as _e:
                        df = _e.submit(
                            lambda: yf.Ticker(yf_sym).history(start=daily_start, end=end_date)
                        ).result(timeout=10)
                except (FutTimeout, Exception):
                    df = pd.DataFrame()
            if not df.empty and df.index.tz is not None:
                df = df.copy()
                df.index = df.index.tz_localize(None)
            return df

        for idx, ticker in enumerate(tickers):
            if progress_callback:
                progress_callback(idx + 1, total, ticker)

            daily = _slice_daily(f"{ticker}.NS")

            if daily.empty or len(daily) < 210:
                continue

            closes = daily["Close"]
            opens = daily["Open"]
            highs = daily["High"]
            lows = daily["Low"]
            volumes = daily["Volume"].astype(float)

            i = len(daily) - 1  # Latest bar
            bar_date = daily.index[i]
            if hasattr(bar_date, 'tz') and bar_date.tz is not None:
                bar_date = bar_date.tz_localize(None)
            if actual_date is None or bar_date > actual_date:
                actual_date = bar_date

            # Overlay Zerodha LTP on today's close if available
            if ltp_map and ticker in ltp_map:
                live_price = ltp_map[ticker]
                daily.iloc[i, daily.columns.get_loc("Close")] = live_price
                # Update high/low if LTP extends the range
                if live_price > daily.iloc[i]["High"]:
                    daily.iloc[i, daily.columns.get_loc("High")] = live_price
                if live_price < daily.iloc[i]["Low"]:
                    daily.iloc[i, daily.columns.get_loc("Low")] = live_price
                closes = daily["Close"]
                highs = daily["High"]
                lows = daily["Low"]

            price = float(closes.iloc[i])
            open_price = float(opens.iloc[i])
            low = float(lows.iloc[i])
            high = float(highs.iloc[i])
            hl_range = high - low
            ibs = (price - low) / hl_range if hl_range > 0 else 0.5
            is_green = price > open_price

            # Gap-down detection
            no_gap_down = True
            if i > 0:
                prev_close = float(closes.iloc[i - 1])
                if open_price < prev_close:
                    no_gap_down = False

            # ── Per-strategy compute (Phase 2 refactor) ──────────────────────
            # Each strategy is a pure function in data/strategies/*. Logic
            # is bit-identical to the previous inline blocks; this is purely
            # code organization. Cross-sectional ranking (Mom20 Z-score,
            # Alpha20 ranking, IBD percentile filter, RS63 volume rank)
            # stays here since it operates on the collected lists below.

            # Strategy RS (IBD): per-ticker candidate + 5-day score history
            cand, day_scores = compute_rs_ibd_features(
                ticker, daily, closes, highs, lows, i, price,
                bench_ret123_val, nifty_regime_on, STOCK_SECTOR_MAP)
            if cand is not None:
                rs_ibd_candidates.append(cand)
                rs_ibd_history[ticker] = day_scores

            # Mom20: raw momentum features for cross-sectional Z-score after loop
            mom_feat = compute_mom20_features(
                ticker, closes, i, price, n50_ret_series, n50_var)
            if mom_feat is not None:
                mom20_raw.append(mom_feat)

            # Alpha20: CAPM alpha features for cross-sectional ranking after loop
            alpha_feat = compute_alpha20_features(
                ticker, closes, i, price, n50_ret_series, n50_var)
            if alpha_feat is not None:
                alpha20_raw.append(alpha_feat)

            # RS63 Satellite signal
            rs63_sig = compute_rs63_signal(
                ticker, daily, closes, lows, volumes, i, price, low,
                ibs, is_green, bench_raw)
            if rs63_sig is not None:
                rs63_signals.append(rs63_sig)

            # Skip gap-down stocks for J/T/R/MW
            if not no_gap_down:
                continue

            # Strategy J — weekly support bounce
            j_sig = compute_j_signal(ticker, daily, closes, opens, highs, lows,
                                     i, price, low, ibs, is_green)
            if j_sig is not None:
                j_signals.append(j_sig)

            # Strategy T — Keltner pullback (deduped against J)
            t_sig = compute_t_signal(ticker, closes, highs, lows, i, price,
                                     ibs, is_green,
                                     {s["ticker"] for s in j_signals})
            if t_sig is not None:
                t_signals.append(t_sig)

            # Strategy R — bullish RSI divergence (deduped against J + T)
            r_sig = compute_r_signal(
                ticker, closes, highs, lows, i, price, ibs, is_green,
                {s["ticker"] for s in j_signals} | {s["ticker"] for s in t_signals},
                _calculate_rsi_series, _find_swing_lows,
                _detect_bullish_divergence, _detect_hidden_bullish_divergence)
            if r_sig is not None:
                r_signals.append(r_sig)

            # Strategy RW: DISABLED — weekly RSI divergence (code preserved in momentum_backtest.py)

            # Strategy MW — weekly ADX trend (deduped against J + T + R)
            mw_sig = compute_mw_signal(
                ticker, daily, closes, highs, lows, i, price, ibs, is_green,
                {s["ticker"] for s in j_signals} |
                {s["ticker"] for s in t_signals} |
                {s["ticker"] for s in r_signals},
                actual_date, _calculate_adx_series)
            if mw_sig is not None:
                mw_signals.append(mw_sig)

        # Strategy WT: disabled (was scanning full Nifty 500, too slow)
        wt_signals = []

        # IBD RS Rating: rank all candidates by weighted score → percentile
        # Apply 5-day consecutive filter, skip top 2, filter >= 80
        if rs_ibd_candidates:
            # Rank today's candidates
            rs_ibd_candidates.sort(key=lambda c: c["weighted"])
            n = len(rs_ibd_candidates)
            for rank_i, c in enumerate(rs_ibd_candidates):
                c["ibd_rating"] = max(1, min(99, int(round((rank_i + 1) / n * 99))))

            # Compute ratings for past 4 days (for 5-day consecutive check)
            # Build historical rankings: for each of last 4 days, rank by that day's score
            past_ratings = {}  # ticker -> list of ratings for days -4 to -1
            for day_offset in range(4):  # 0=4 days ago, 3=1 day ago
                day_scores = {}
                for c in rs_ibd_candidates:
                    hist = rs_ibd_history.get(c["ticker"], [])
                    if len(hist) >= 5 and day_offset < len(hist) - 1:
                        day_scores[c["ticker"]] = hist[day_offset]
                if len(day_scores) >= 10:
                    sorted_t = sorted(day_scores.keys(), key=lambda t: day_scores[t])
                    dn = len(sorted_t)
                    for ri, t in enumerate(sorted_t):
                        if t not in past_ratings:
                            past_ratings[t] = []
                        past_ratings[t].append(max(1, min(99, int(round((ri + 1) / dn * 99)))))

            # Filter: today >= 80 + above EMA + dist_high + 5-day consecutive >= 80
            qualified = []
            for c in rs_ibd_candidates:
                if c["ibd_rating"] < 80 or not c["above_ema"] or not c["dist_high_ok"]:
                    continue
                # RS-123d must be positive (stock outperforming benchmark)
                if c["rs_pct"] <= 0:
                    continue
                # 5-day consecutive: check past 4 days all >= 80
                hist_ratings = past_ratings.get(c["ticker"], [])
                if len(hist_ratings) >= 4 and all(r >= 80 for r in hist_ratings):
                    qualified.append(c)

            # Sort by IBD Rating descending (matches backtest ranking)
            qualified.sort(key=lambda c: -c["ibd_rating"])

            for rank_i, c in enumerate(qualified):
                rs_signals.append({
                    "ticker": c["ticker"],
                    "price": c["price"],
                    "rs_pct": c["rs_pct"],
                    "ibd_rating": c["ibd_rating"],
                    "dist_high_pct": c["dist_high_pct"],
                    "atr_pct": c["atr_pct"],
                    "stop_pct": c["stop_pct"],
                    "sector": c["sector"],
                    "regime_off": c["regime_off"],
                    "rs_rank": rank_i + 1,  # 1-based rank (pick #3)
                })

        # Mom20: Z-score momentum ratios and pick top 20
        mom20_signals = []
        if len(mom20_raw) >= 5:
            # Filter beta ≤ 1.2 (frozen Mom20 spec)
            mom20_eligible = [d for d in mom20_raw if d.get("beta") is None or abs(d["beta"]) <= 1.2]
            if len(mom20_eligible) < 5:
                mom20_eligible = mom20_raw  # fallback if too few pass
            # Scoring: 50% Z_12 + 50% Z_3 (frozen spec — NOT z_6)
            mr_12_arr = np.array([d["mr_12"] for d in mom20_eligible])
            mr_3_arr  = np.array([d["mr_3"]  if d.get("mr_3") is not None else d["mr_6"] for d in mom20_eligible])
            z_12 = (mr_12_arr - mr_12_arr.mean()) / mr_12_arr.std() if mr_12_arr.std() > 0 else np.zeros_like(mr_12_arr)
            z_3  = (mr_3_arr  - mr_3_arr.mean())  / mr_3_arr.std()  if mr_3_arr.std()  > 0 else np.zeros_like(mr_3_arr)
            weighted_z = 0.5 * z_12 + 0.5 * z_3
            for idx_m, d in enumerate(mom20_eligible):
                z = weighted_z[idx_m]
                norm_score = (1 + z) if z >= 0 else 1 / (1 - z)
                d["norm_score"] = norm_score
            mom20_eligible.sort(key=lambda d: -d["norm_score"])
            for rank_i, d in enumerate(mom20_eligible[:40]):
                mom20_signals.append({
                    "ticker": d["ticker"],
                    "price": d["price"],
                    "ret_12m": round(d["ret_12m"] * 100, 1),
                    "ret_3m": round(d["ret_3m"] * 100, 1) if d.get("ret_3m") is not None else round(d["ret_6m"] * 100, 1),
                    "ret_6m": round(d["ret_6m"] * 100, 1),
                    "volatility": round(d["sigma"] * 100, 1),
                    "momentum_score": round(d["norm_score"], 3),
                    "beta": round(abs(d["beta"]), 2) if d.get("beta") is not None else None,
                    "rank": rank_i + 1,
                    "stop_pct": 0,
                    "atr_pct": 0,
                })

        # Mom20 overflow: top-40 uncapped minus capped — high-beta RS63 candidates
        mom20_overflow = []
        mom20_unfiltered_ranks = {}
        mom20_all_prices = {}
        if len(mom20_raw) >= 5:
            capped_tickers = {s["ticker"] for s in mom20_signals}
            mr_12_uc = np.array([d["mr_12"] for d in mom20_raw])
            mr_3_uc  = np.array([d["mr_3"] if d.get("mr_3") is not None else d["mr_6"] for d in mom20_raw])
            z12_uc = (mr_12_uc - mr_12_uc.mean()) / mr_12_uc.std() if mr_12_uc.std() > 0 else np.zeros_like(mr_12_uc)
            z3_uc  = (mr_3_uc  - mr_3_uc.mean())  / mr_3_uc.std()  if mr_3_uc.std()  > 0 else np.zeros_like(mr_3_uc)
            wz_uc  = 0.5 * z12_uc + 0.5 * z3_uc
            for idx_u, d in enumerate(mom20_raw):
                z = wz_uc[idx_u]
                d["norm_score_uc"] = (1 + z) if z >= 0 else 1 / (1 - z)
            ov_rank = 0
            for d in sorted(mom20_raw, key=lambda d: -d["norm_score_uc"])[:20]:
                if d["ticker"] not in capped_tickers and d.get("beta") is not None and abs(d["beta"]) > 1.2:
                    ov_rank += 1
                    mom20_overflow.append({
                        "ticker": d["ticker"],
                        "price": d["price"],
                        "ret_12m": round(d["ret_12m"] * 100, 1),
                        "ret_3m": round(d["ret_3m"] * 100, 1) if d.get("mr_3") is not None else round(d["ret_6m"] * 100, 1),
                        "momentum_score": round(d["norm_score_uc"], 3),
                        "beta": round(abs(d["beta"]), 2),
                        "rank": ov_rank,
                    })
            # Unfiltered rank map (no beta cap) — used for exit decisions on held stocks
            mom20_unfiltered_ranks = {
                d["ticker"]: rank + 1
                for rank, d in enumerate(sorted(mom20_raw, key=lambda d: -d.get("norm_score_uc", 0)))
            }
            # All N200 prices (no beta filter) — used for SIP min capital on held stocks
            mom20_all_prices = {d["ticker"]: d["price"] for d in mom20_raw}

        # Mom15: same scoring as Mom20 but beta cap 1.0, top 15
        # Load EPS data for TTM growth column + filtered view
        eps_db = {}
        eps_path = os.path.join(os.path.dirname(__file__), "quarterly_eps.json")
        if os.path.exists(eps_path):
            try:
                with open(eps_path) as _ef:
                    eps_db = json.load(_ef)
            except Exception:
                pass

        def _get_ttm_eps_growth(ticker):
            """Get TTM EPS YoY growth % for a ticker. Returns float or None."""
            if ticker not in eps_db:
                return None
            stock_eps = eps_db[ticker]
            # Use annual EPS (more depth) — format: {"annual": {"Mar 2014": 16.31, ...}}
            annual = stock_eps.get("annual", {}) if isinstance(stock_eps, dict) else {}
            if not annual:
                # Old format: flat dict of quarterly
                annual = stock_eps if isinstance(stock_eps, dict) and "quarterly" not in stock_eps else {}
            if len(annual) < 2:
                return None
            # Sort by year, get latest two
            sorted_periods = sorted(annual.keys(),
                                     key=lambda p: (p.split()[-1], p.split()[0]))
            latest = annual[sorted_periods[-1]]
            prev = annual[sorted_periods[-2]]
            if abs(prev) < 0.01:
                return None
            return (latest / prev - 1) * 100

        mom15_signals = []
        if len(mom20_raw) >= 5:
            # Filter beta <= 1.0
            mom15_pool = [d for d in mom20_raw if d.get("beta") is not None and d["beta"] <= 1.0]
            if len(mom15_pool) >= 5:
                mr_12_arr = np.array([d["mr_12"] for d in mom15_pool])
                mr_6_arr = np.array([d["mr_6"] for d in mom15_pool])
                mr_3_arr = np.array([d.get("mr_3", 0) or 0 for d in mom15_pool])
                z_12 = (mr_12_arr - mr_12_arr.mean()) / mr_12_arr.std() if mr_12_arr.std() > 0 else np.zeros_like(mr_12_arr)
                z_6 = (mr_6_arr - mr_6_arr.mean()) / mr_6_arr.std() if mr_6_arr.std() > 0 else np.zeros_like(mr_6_arr)
                z_3 = (mr_3_arr - mr_3_arr.mean()) / mr_3_arr.std() if mr_3_arr.std() > 0 else np.zeros_like(mr_3_arr)
                # 50/0/50 weighting (frozen: 12m + 3m, no 6m)
                weighted_z = 0.50 * z_12 + 0.00 * z_6 + 0.50 * z_3
                for idx_m, d in enumerate(mom15_pool):
                    z = weighted_z[idx_m]
                    d["norm_score_15"] = (1 + z) if z >= 0 else 1 / (1 - z)
                mom15_pool.sort(key=lambda d: -d["norm_score_15"])
                # Build full ranked list (top 30 for display, selection on top 15)
                for rank_i, d in enumerate(mom15_pool[:30]):
                    ttm_growth = _get_ttm_eps_growth(d["ticker"])
                    mom15_signals.append({
                        "ticker": d["ticker"],
                        "price": d["price"],
                        "ret_12m": round(d["ret_12m"] * 100, 1),
                        "ret_3m": round(d["ret_3m"] * 100, 1) if d.get("ret_3m") is not None else None,
                        "volatility": round(d["sigma"] * 100, 1),
                        "beta": d["beta"],
                        "momentum_score": round(d["norm_score_15"], 3),
                        "ttm_eps_growth": round(ttm_growth, 1) if ttm_growth is not None else None,
                        "eps_pass": ttm_growth is None or ttm_growth > 0,
                        "rank": rank_i + 1,
                        "stop_pct": 0,
                        "atr_pct": 0,
                    })

        # RS63 Satellite: rank by volume ratio high to low
        rs63_signals.sort(key=lambda s: s.get("vol_ratio") or 0, reverse=True)
        for rank_i, s in enumerate(rs63_signals):
            s["rank"] = rank_i + 1

        # Alpha20: rank by highest alpha and pick top 20
        alpha20_signals = []
        if alpha20_raw:
            # Beta cap filter (frozen: beta <= 1.2)
            alpha20_raw = [d for d in alpha20_raw if d["beta"] <= 1.2]
            # Sort by highest alpha
            alpha20_raw.sort(key=lambda d: -d["alpha"])
            for rank_i, d in enumerate(alpha20_raw[:20]):
                if d["alpha"] > 0:  # Only positive alpha
                    alpha20_signals.append({
                        "ticker": d["ticker"],
                        "price": d["price"],
                        "alpha": d["alpha"],
                        "beta": d["beta"],
                        "rank": rank_i + 1,
                        "stop_pct": 0,
                        "atr_pct": 0,
                    })

        # Compute sector momentum
        sector_momentum = self._compute_sector_momentum(end_date)

        # Attach sector info to each signal
        for sig_list in [j_signals, t_signals, r_signals, mw_signals, wt_signals, rs_signals]:
            for s in sig_list:
                sector = STOCK_SECTOR_MAP.get(s["ticker"], "OTHER")
                s["sector"] = sector
                sec_data = sector_momentum.get(sector, {})
                s["sector_rs"] = sec_data.get("rs", 0)
                s["sector_momentum"] = sec_data.get("momentum", 0)
                s["sector_delta_5d"] = sec_data.get("delta_5d", 0)

        # Sort by volatility (lowest ATR% first = calmest stocks)
        # R gets priority in combined ranking (best WR + P&L historically)
        j_signals.sort(key=lambda s: s.get("atr_pct", 99.0))
        t_signals.sort(key=lambda s: s.get("atr_pct", 99.0))
        r_signals.sort(key=lambda s: s.get("atr_pct", 99.0))
        mw_signals.sort(key=lambda s: s.get("atr_pct", 99.0))
        wt_signals.sort(key=lambda s: s.get("atr_pct", 99.0))
        rs_signals.sort(key=lambda s: -s.get("ibd_rating", 0))
        result = {
            "j_signals": j_signals,
            "t_signals": t_signals,
            "r_signals": r_signals,
            "rw_signals": [],  # RW disabled
            "mw_signals": mw_signals,
            "wt_signals": wt_signals,
            "rs_signals": rs_signals,
            "mom15_signals": mom15_signals,
            "mom20_signals": mom20_signals,
            "mom20_overflow": mom20_overflow,
            "mom20_unfiltered_ranks": mom20_unfiltered_ranks,
            "mom20_all_prices": mom20_all_prices,
            "alpha20_signals": alpha20_signals,
            "rs63_signals": rs63_signals[:25],
            "nifty_regime": "ON" if nifty_regime_on else "OFF",
            "mom20_regime": "ON" if mom20_regime_on else "OFF",
            "sector_momentum": sector_momentum,
            "last_updated": datetime.now().isoformat(),
            "universe": universe,
        }

        if actual_date is not None:
            result["actual_date"] = actual_date.strftime("%Y-%m-%d")
        if is_historical:
            result["scan_date"] = scan_date
        else:
            self._save_cache(result)
            self._append_signals_history(j_signals, t_signals, r_signals, rw_signals, mw_signals, wt_signals)

        return result

    def get_cache_age_minutes(self):
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
            last = datetime.fromisoformat(data["last_updated"])
            return (datetime.now() - last).total_seconds() / 60
        except Exception:
            return None

    def _load_cache(self):
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
            last = datetime.fromisoformat(data["last_updated"])
            ttl = get_cache_ttl()
            if (datetime.now() - last).total_seconds() < ttl * 60:
                return data
        except Exception:
            pass
        return None

    def _save_cache(self, data):
        with open(self.cache_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _append_signals_history(self, j_signals, t_signals=None, r_signals=None, rw_signals=None, mw_signals=None, wt_signals=None):
        """Append today's scan results to the history CSV."""
        history_file = LIVE_SIGNALS_HISTORY_FILE
        scan_date = datetime.now().strftime("%Y-%m-%d")
        scan_time = datetime.now().strftime("%H:%M")

        write_header = not os.path.exists(history_file)

        with open(history_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "date", "time", "strategy", "ticker", "price",
                    "support", "close_near_pct", "ibs",
                ])

            for s in j_signals:
                writer.writerow([
                    scan_date, scan_time, "J", s["ticker"], s["price"],
                    s["support"], s["close_near_pct"], s["ibs"],
                ])

            for s in (t_signals or []):
                writer.writerow([
                    scan_date, scan_time, "T", s["ticker"], s["price"],
                    "", "", "",
                ])

            for s in (r_signals or []):
                writer.writerow([
                    scan_date, scan_time, "R", s["ticker"], s["price"],
                    s.get("swing_low", ""), "", "",
                ])

            for s in (rw_signals or []):
                writer.writerow([
                    scan_date, scan_time, "RW", s["ticker"], s["price"],
                    s.get("swing_low", ""), "", "",
                ])

            for s in (mw_signals or []):
                writer.writerow([
                    scan_date, scan_time, "MW", s["ticker"], s["price"],
                    "", "", "",
                ])

            for s in (wt_signals or []):
                writer.writerow([
                    scan_date, scan_time, "WT", s["ticker"], s["price"],
                    "", "", "",
                ])

    # ==================== Position Management ====================

    def _load_positions_data(self):
        try:
            with open(self.positions_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"active": [], "closed": []}

    def _save_positions_data(self, data):
        with open(self.positions_file, 'w') as f:
            json.dump(data, f, indent=2)

    def get_positions(self):
        return self._load_positions_data()["active"]

    def add_position(self, ticker, strategy, entry_price, amount,
                     support_level=0, ibs=0, metadata=None):
        """Create a new position from a Buy signal."""
        shares = int(amount // entry_price)
        if shares <= 0:
            return {"error": "Amount too small for even 1 share"}

        pos = {
            "id": str(uuid.uuid4())[:8],
            "ticker": ticker,
            "strategy": strategy,
            "entry_date": datetime.now().strftime("%Y-%m-%d"),
            "entry_price": round(entry_price, 2),
            "shares": shares,
            "amount_invested": round(shares * entry_price, 2),
            "support_level": round(support_level, 2),
            "partial_exit_done": False,
            "partial_stage": 0,
            "remaining_shares": shares,
            "metadata": metadata or {},
        }
        if self.user_id:
            pos["user_id"] = self.user_id

        data = self._load_positions_data()
        data["active"].append(pos)
        self._save_positions_data(data)
        return pos

    def delete_position(self, position_id):
        """Remove a position without placing any order (e.g. was never filled)."""
        data = self._load_positions_data()
        before = len(data["active"])
        data["active"] = [p for p in data["active"] if p["id"] != position_id]
        if len(data["active"]) == before:
            return {"error": "Position not found"}
        self._save_positions_data(data)
        return {"deleted": position_id}

    def close_position(self, position_id, exit_price, reason, shares_to_exit):
        """Close (or partially close) a position."""
        data = self._load_positions_data()
        active = data["active"]
        pos = None
        pos_idx = None

        for idx, p in enumerate(active):
            if p["id"] == position_id:
                pos = p
                pos_idx = idx
                break

        if pos is None:
            return {"error": "Position not found"}

        current_shares = pos["remaining_shares"]
        shares_to_exit = min(shares_to_exit, current_shares)

        pnl_pct = round(((exit_price - pos["entry_price"]) / pos["entry_price"]) * 100, 2)
        pnl_amount = round((exit_price - pos["entry_price"]) * shares_to_exit, 2)

        exit_date_str = datetime.now().strftime("%Y-%m-%d")
        try:
            holding_days = (datetime.strptime(exit_date_str, "%Y-%m-%d")
                            - datetime.strptime(pos["entry_date"], "%Y-%m-%d")).days
        except Exception:
            holding_days = 0

        closed_record = {
            "id": pos["id"],
            "ticker": pos["ticker"],
            "strategy": pos["strategy"],
            "entry_date": pos["entry_date"],
            "entry_price": pos["entry_price"],
            "exit_date": exit_date_str,
            "exit_price": round(exit_price, 2),
            "shares_exited": shares_to_exit,
            "reason": reason,
            "pnl_pct": pnl_pct,
            "pnl_amount": pnl_amount,
            "holding_days": holding_days,
        }

        remaining = current_shares - shares_to_exit
        if remaining <= 0:
            # Fully closed
            active.pop(pos_idx)
        else:
            # Partial close — advance stage so exit signals move to next level
            pos["remaining_shares"] = remaining
            current_stage = pos.get("partial_stage", 0)
            pos["partial_stage"] = current_stage + 1
            # Mark partial_exit_done after stage 1 (2nd partial) for strategies with 3-stage exits
            if pos["partial_stage"] >= 2:
                pos["partial_exit_done"] = True

        data["closed"].append(closed_record)
        self._save_positions_data(data)
        return closed_record

    # ==================== Exit Signal Checking ====================

    def check_exit_signals(self, ltps=None):
        """Check exit conditions for all active positions.
        ltps: optional dict {ticker: live_price} from Zerodha. Falls back to yfinance close."""
        positions = self.get_positions()
        if not positions:
            return []

        # Fetch Nifty data for Nifty shield
        end_date = datetime.now()
        try:
            nifty_data = yf.Ticker("^NSEI").history(
                start=end_date - timedelta(days=30), end=end_date)
            nifty_closes = nifty_data["Close"]
        except Exception:
            nifty_closes = pd.Series(dtype=float)

        nifty_close_today = float(nifty_closes.iloc[-1]) if len(nifty_closes) > 0 else 0.0

        exit_signals = []

        for pos in positions:
            ticker = pos["ticker"]
            try:
                daily = yf.Ticker(f"{ticker}.NS").history(
                    start=end_date - timedelta(days=30), end=end_date)
            except Exception:
                continue

            if daily.empty or len(daily) < 2:
                continue

            current_price = ltps.get(ticker) if ltps and ticker in ltps else float(daily["Close"].iloc[-1])
            highs = daily["High"]

            entry_price = pos["entry_price"]
            pnl_pct = round(((current_price - entry_price) / entry_price) * 100, 2)

            if pos["strategy"] == "J":
                support = pos["support_level"]
                t1_price = entry_price * 1.05
                t2_price = entry_price * 1.10

                # Nifty drop shield: skip support break if Nifty fell same or more
                j_nifty_entry = pos.get("metadata", {}).get("nifty_at_entry", 0)
                j_nifty_shields = False
                if j_nifty_entry > 0 and nifty_close_today > 0:
                    nifty_pct = (nifty_close_today - j_nifty_entry) / j_nifty_entry
                    stock_pct = (current_price - entry_price) / entry_price
                    if nifty_pct <= stock_pct and nifty_pct < 0:
                        j_nifty_shields = True

                if not pos["partial_exit_done"]:
                    # Support break — full exit (skip if Nifty shielded)
                    if not j_nifty_shields and current_price < support:
                        exit_signals.append(self._make_exit_signal(
                            pos, current_price, pnl_pct, "SUPPORT_BREAK",
                            pos["remaining_shares"]))
                    # +5% partial
                    elif current_price >= t1_price:
                        half = pos["shares"] // 2
                        exit_signals.append(self._make_exit_signal(
                            pos, current_price, pnl_pct, "5PCT_PARTIAL", half))
                else:
                    # Remaining shares after partial
                    if not j_nifty_shields and current_price < support:
                        exit_signals.append(self._make_exit_signal(
                            pos, current_price, pnl_pct, "SUPPORT_BREAK",
                            pos["remaining_shares"]))
                    elif current_price >= t2_price:
                        exit_signals.append(self._make_exit_signal(
                            pos, current_price, pnl_pct, "10PCT_TARGET",
                            pos["remaining_shares"]))
                    else:
                        # Chandelier exit: highest high since entry - 3x ATR(14)
                        entry_date_str = pos.get("entry_date", "")
                        entry_dt = pd.Timestamp(entry_date_str) if entry_date_str else daily.index[0]
                        if entry_dt.tzinfo is None and daily.index.tz is not None:
                            entry_dt = entry_dt.tz_localize(daily.index.tz)
                        since_entry = daily[daily.index >= entry_dt]
                        if len(since_entry) >= 2:
                            highest_high = float(since_entry["High"].max())
                            # ATR(14) from daily data
                            tr = pd.concat([
                                daily["High"] - daily["Low"],
                                (daily["High"] - daily["Close"].shift()).abs(),
                                (daily["Low"] - daily["Close"].shift()).abs()
                            ], axis=1).max(axis=1)
                            atr14 = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else 0.0
                            chandelier_stop = highest_high - 3.0 * atr14
                            if atr14 > 0 and current_price < chandelier_stop:
                                exit_signals.append(self._make_exit_signal(
                                    pos, current_price, pnl_pct, "CHANDELIER_EXIT",
                                    pos["remaining_shares"]))

            elif pos["strategy"] == "T":
                # T: 3-stage exit logic
                # Stage 0: 5% SL → exit all. +6% → sell 1/3, set stage=1.
                # Stage 1: 3% SL → exit remaining. +10% → sell 1/3, set stage=2 (partial_exit_done=True).
                # Stage 2: 3% SL → exit remaining. Upper Keltner → exit remaining.
                stage = pos.get("partial_stage", 0)
                third = pos["shares"] // 3

                # Tighter SL after first partial exit
                sl_pct = 0.03 if stage >= 1 else 0.05
                if current_price <= entry_price * (1 - sl_pct):
                    sz = pos["remaining_shares"]
                    sl_label = f"HARD_SL_{int(sl_pct*100)}PCT"
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, sl_label, sz))
                    continue

                closes_t = daily["Close"]
                highs_t = daily["High"]
                lows_t = daily["Low"]
                ema20_t = float(closes_t.ewm(span=20, adjust=False).mean().iloc[-1])
                prev_close_t = closes_t.shift(1)
                tr1 = highs_t - lows_t
                tr2 = (highs_t - prev_close_t).abs()
                tr3 = (lows_t - prev_close_t).abs()
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr14_t = float(true_range.ewm(alpha=1/14, min_periods=14, adjust=False).mean().iloc[-1])
                upper_keltner = ema20_t + 2 * atr14_t

                if stage == 0 and current_price >= entry_price * 1.06 and third > 0:
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "PARTIAL_6PCT_1of3", third))
                elif stage == 1 and current_price >= entry_price * 1.10 and third > 0:
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "PARTIAL_10PCT_2of3", third))
                elif (stage == 2 or pos["partial_exit_done"]) and current_price >= upper_keltner:
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "KELTNER_UPPER_EXIT",
                        pos["remaining_shares"]))

            elif pos["strategy"] == "R":
                # R: 3-stage exit logic (mirrors T with structural SL)
                stage = pos.get("partial_stage", 0)
                third = pos["shares"] // 3
                structural_stop = pos.get("metadata", {}).get("r_swing_low_stop", 0)

                # Nifty crash shield for R
                r_nifty_entry = pos.get("metadata", {}).get("nifty_at_entry", 0)
                r_nifty_shields = False
                if r_nifty_entry > 0 and nifty_close_today > 0:
                    nifty_pct = (nifty_close_today - r_nifty_entry) / r_nifty_entry
                    stock_pct = (current_price - entry_price) / entry_price
                    if nifty_pct <= stock_pct and nifty_pct < 0:
                        r_nifty_shields = True

                # Structural SL: 1% below divergence swing low — exit all
                if not r_nifty_shields and structural_stop > 0 and current_price <= structural_stop:
                    sz = pos["remaining_shares"]
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "STRUCTURAL_SL", sz))
                    continue

                # Tight SL after first partial (3%) — also shielded
                if not r_nifty_shields and stage >= 1 and current_price <= entry_price * 0.97:
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "HARD_SL_3PCT",
                        pos["remaining_shares"]))
                    continue

                # Compute upper Keltner for final exit
                closes_r = daily["Close"]
                highs_r = daily["High"]
                lows_r = daily["Low"]
                ema20_r = float(closes_r.ewm(span=20, adjust=False).mean().iloc[-1])
                prev_close_r = closes_r.shift(1)
                tr1 = highs_r - lows_r
                tr2 = (highs_r - prev_close_r).abs()
                tr3 = (lows_r - prev_close_r).abs()
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr14_r = float(true_range.ewm(alpha=1/14, min_periods=14, adjust=False).mean().iloc[-1])
                upper_keltner = ema20_r + 2 * atr14_r

                if stage == 0 and current_price >= entry_price * 1.06 and third > 0:
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "PARTIAL_6PCT_1of3", third))
                elif stage == 1 and current_price >= entry_price * 1.10 and third > 0:
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "PARTIAL_10PCT_2of3", third))
                elif (stage == 2 or pos["partial_exit_done"]) and current_price >= upper_keltner:
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "KELTNER_UPPER_EXIT",
                        pos["remaining_shares"]))

            elif pos["strategy"] == "MW":
                # MW exits: 6% SL initial, 3% after P1, breakeven after P2, partials, weekly Keltner
                stage = pos.get("partial_stage", 0)
                third = pos["shares"] // 3

                # Nifty crash shield for MW
                mw_nifty_entry = pos.get("metadata", {}).get("nifty_at_entry", 0)
                mw_nifty_shields = False
                if mw_nifty_entry > 0 and nifty_close_today > 0:
                    nifty_pct = (nifty_close_today - mw_nifty_entry) / mw_nifty_entry
                    stock_pct = (current_price - entry_price) / entry_price
                    if nifty_pct <= stock_pct and nifty_pct < 0:
                        mw_nifty_shields = True

                # Hard SL: 6% initial, 3% after P1, breakeven after P2
                if stage >= 2:
                    mw_sl_price = entry_price
                    sl_label = "BREAKEVEN_SL"
                elif stage >= 1:
                    mw_sl_price = entry_price * 0.97
                    sl_label = "HARD_SL_3PCT"
                else:
                    mw_sl_price = entry_price * 0.94
                    sl_label = "HARD_SL_6PCT"

                if not mw_nifty_shields and current_price <= mw_sl_price:
                    sz = pos["remaining_shares"]
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, sl_label, sz))
                    continue

                # 2-stage partial exits (+6% sell 1/3, +10% sell 1/3)
                if stage == 0 and current_price >= entry_price * 1.06 and third > 0:
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "PARTIAL_6PCT_1of3", third))
                elif stage == 1 and current_price >= entry_price * 1.10 and third > 0:
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "PARTIAL_10PCT_2of3", third))

                # Weekly upper Keltner exit on remaining (after first partial)
                if stage >= 1:
                    try:
                        weekly = yf.Ticker(f"{ticker}.NS").history(
                            start=end_date - timedelta(days=180), end=end_date, interval="1wk")
                        if not weekly.empty and len(weekly) >= 14:
                            w_closes = weekly["Close"]
                            w_ema20 = float(w_closes.ewm(span=20, adjust=False).mean().iloc[-1])
                            w_highs = weekly["High"]
                            w_lows = weekly["Low"]
                            w_prev = w_closes.shift(1)
                            w_tr = pd.concat([w_highs - w_lows, (w_highs - w_prev).abs(), (w_lows - w_prev).abs()], axis=1).max(axis=1)
                            w_atr14 = float(w_tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean().iloc[-1])
                            mw_upper_keltner = w_ema20 + 2 * w_atr14
                            if current_price >= mw_upper_keltner:
                                exit_signals.append(self._make_exit_signal(
                                    pos, current_price, pnl_pct, "KELTNER_UPPER_EXIT",
                                    pos["remaining_shares"]))
                    except Exception:
                        pass

            elif pos["strategy"] == "RS":
                # RS exits: 8% hard SL, 30w EMA trend break, 10d underperformance
                rs_shares = pos["shares"]

                # Nifty crash shield for RS
                rs_nifty_entry = pos.get("metadata", {}).get("nifty_at_entry", 0)
                rs_nifty_shields = False
                if rs_nifty_entry > 0 and nifty_close_today > 0:
                    nifty_pct = (nifty_close_today - rs_nifty_entry) / rs_nifty_entry
                    stock_pct = (current_price - entry_price) / entry_price
                    if nifty_pct <= stock_pct and nifty_pct < 0:
                        rs_nifty_shields = True

                # 1. Hard SL: 8% from entry (with Nifty shield)
                if not rs_nifty_shields and current_price <= entry_price * 0.92:
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "RS_HARD_SL_8PCT", rs_shares))
                    continue

                # 2. Price < 30-week EMA (trend break)
                try:
                    rs_daily = yf.Ticker(f"{ticker}.NS").history(
                        start=end_date - timedelta(days=300), end=end_date)
                    if not rs_daily.empty and len(rs_daily) >= 150:
                        rs_weekly = rs_daily["Close"].resample("W-FRI").last().dropna()
                        if len(rs_weekly) >= 30:
                            rs_ema30w = rs_weekly.ewm(span=30, adjust=False).mean()
                            rs_ema30w_daily = rs_ema30w.reindex(rs_daily.index, method="ffill")
                            rs_ema30w_val = float(rs_ema30w_daily.iloc[-1])
                            if current_price < rs_ema30w_val:
                                exit_signals.append(self._make_exit_signal(
                                    pos, current_price, pnl_pct, "RS_TREND_BREAK", rs_shares))
                                continue
                except Exception:
                    pass

                # 3. RS-21d < 0 for 10 consecutive days (underperformance vs Nifty)
                try:
                    rs_daily2 = yf.Ticker(f"{ticker}.NS").history(
                        start=end_date - timedelta(days=60), end=end_date)
                    if not rs_daily2.empty and len(rs_daily2) >= 22 and len(nifty_closes) >= 22:
                        # Compute 21d return for stock and Nifty
                        stock_closes = rs_daily2["Close"]
                        stock_ret21 = (stock_closes / stock_closes.shift(21) - 1) * 100
                        # Get matching Nifty data
                        nifty_full = yf.Ticker("^NSEI").history(
                            start=end_date - timedelta(days=60), end=end_date)
                        if not nifty_full.empty:
                            nifty_ret21 = (nifty_full["Close"] / nifty_full["Close"].shift(21) - 1) * 100
                            nifty_ret21_daily = nifty_ret21.reindex(stock_closes.index, method="ffill")
                            rs_21d = stock_ret21 - nifty_ret21_daily
                            rs_21d = rs_21d.dropna()
                            if len(rs_21d) >= 10:
                                last_10 = rs_21d.iloc[-10:]
                                if all(v < 0 for v in last_10):
                                    exit_signals.append(self._make_exit_signal(
                                        pos, current_price, pnl_pct, "RS_UNDERPERFORM", rs_shares))
                                    continue
                except Exception:
                    pass

            elif pos["strategy"] == "RS63":
                # RS63 exits: 8% SL, RSI(14,3d avg) < 40 for 3d, RS63(5d avg) < 0, 8wk time stop
                rs63_shares = pos["remaining_shares"]

                # 1. Hard SL: 8% from entry
                if current_price <= entry_price * 0.92:
                    exit_signals.append(self._make_exit_signal(
                        pos, current_price, pnl_pct, "RS63_HARD_SL_8PCT", rs63_shares))
                    continue

                try:
                    # Fetch enough data for RS63 and RSI
                    rs63_daily = yf.Ticker(f"{ticker}.NS").history(
                        start=end_date - timedelta(days=200), end=end_date)
                    if rs63_daily.empty or len(rs63_daily) < 70:
                        continue

                    rs63_closes = rs63_daily["Close"].astype(float)

                    # Fetch benchmark for RS63
                    bench_rs63 = yf.Ticker("^CNX200").history(
                        start=end_date - timedelta(days=200), end=end_date)
                    if not bench_rs63.empty:
                        bench_aligned = bench_rs63["Close"].astype(float).reindex(
                            rs63_daily.index, method="ffill")
                        rs_ratio = rs63_closes / bench_aligned
                        rs63_raw = (rs_ratio / rs_ratio.shift(63) - 1) * 100
                        rs63_smooth = rs63_raw.rolling(5, min_periods=1).mean()

                        # 2. RS63(5d avg) < 0
                        rs63_val = float(rs63_smooth.iloc[-1]) if not pd.isna(rs63_smooth.iloc[-1]) else 0
                        if rs63_val < 0:
                            exit_signals.append(self._make_exit_signal(
                                pos, current_price, pnl_pct, "RS63_NEG", rs63_shares))
                            continue

                    # 3. RSI(14, 3d avg) < 40 for 3 consecutive days
                    delta_r = rs63_closes.diff()
                    gain_r = delta_r.clip(lower=0)
                    loss_r = (-delta_r).clip(lower=0)
                    avg_gain = gain_r.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
                    avg_loss = loss_r.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
                    rsi_raw = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, 1e-10)))
                    rsi_smooth = rsi_raw.rolling(3, min_periods=1).mean()

                    if len(rsi_smooth) >= 3:
                        last3 = rsi_smooth.iloc[-3:]
                        if all(float(v) < 40 for v in last3 if not pd.isna(v)):
                            exit_signals.append(self._make_exit_signal(
                                pos, current_price, pnl_pct, "RS63_RSI_BELOW40_3D", rs63_shares))
                            continue

                    # 4. Time stop: 8 weeks (56 days) + < 3% gain
                    entry_date_str = pos.get("entry_date", "")
                    if entry_date_str:
                        entry_dt = pd.Timestamp(entry_date_str)
                        days_held = (pd.Timestamp.now() - entry_dt).days
                        gain_pct = (current_price - entry_price) / entry_price * 100
                        if days_held >= 56 and gain_pct < 3.0:
                            exit_signals.append(self._make_exit_signal(
                                pos, current_price, pnl_pct, "RS63_TIME_STOP", rs63_shares))
                            continue

                except Exception:
                    pass

        # Underwater exit: if held >= 10 trading days and still below entry, cut it
        # RS strategy uses rs_uw_days=0 (disabled), so skip RS positions
        for pos in positions:
            # Skip RS/RS63 — underwater exit disabled
            if pos["strategy"] in ("RS", "RS63"):
                continue

            # Skip if already flagged for exit above
            already_flagged = any(e["position_id"] == pos["id"] for e in exit_signals)
            if already_flagged:
                continue

            entry_date_str = pos.get("entry_date", "")
            if not entry_date_str:
                continue

            try:
                daily = yf.Ticker(f"{pos['ticker']}.NS").history(
                    start=end_date - timedelta(days=30), end=end_date)
            except Exception:
                continue

            if daily.empty or len(daily) < 2:
                continue

            current_price = ltps.get(pos["ticker"]) if ltps and pos["ticker"] in ltps else float(daily["Close"].iloc[-1])
            entry_price = pos["entry_price"]

            if current_price >= entry_price:
                continue  # Not underwater

            entry_dt = pd.Timestamp(entry_date_str)
            if entry_dt.tzinfo is None and daily.index.tz is not None:
                entry_dt = entry_dt.tz_localize(daily.index.tz)
            trading_days_held = len(daily[daily.index >= entry_dt])
            uw_days = 25 if pos["strategy"] == "MW" else 10
            if trading_days_held >= uw_days:
                pnl_pct = round(((current_price - entry_price) / entry_price) * 100, 2)
                sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                exit_signals.append(self._make_exit_signal(
                    pos, current_price, pnl_pct, "UNDERWATER_EXIT", sz))

        return exit_signals

    def _make_exit_signal(self, pos, current_price, pnl_pct, reason, shares):
        return {
            "position_id": pos["id"],
            "ticker": pos["ticker"],
            "strategy": pos["strategy"],
            "entry_price": pos["entry_price"],
            "current_price": round(current_price, 2),
            "pnl_pct": pnl_pct,
            "reason": reason,
            "shares_to_exit": shares,
            "entry_date": pos["entry_date"],
        }
