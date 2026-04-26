"""
ETF Core — Momentum Rotation Engine
====================================
Holds top 5 ETFs from a 35-ETF universe, ranked by RS63 vs Nifty 200.
Idle capital parked in LIQUIDBEES.

Strategy rules (frozen):
  Entry: checked Friday, executed Monday open
    - RS63 > 2% on 2 consecutive Fridays AND price > SMA63 AND rank <= 5
  Exit (X1/X2/X3): checked Friday, executed Monday open
    - X1: RS63 < 0
    - X2: price < SMA63
    - X3: price < 20d high × 85%
  Exit (X4): checked DAILY, executed next open
    - X4: price > SMA20×1.20 OR price > SMA50×1.30 (overextended)
    - 6-week cooloff on re-entry after X4 exit
  Geography cap: MAX_INTL_SLOTS = 3 (international ETFs)
"""

import json
import os
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import numpy as np

# ── Universe ──────────────────────────────────────────────────────────────────

ETF_UNIVERSE = [
    # Indian Broad Market
    {"symbol": "JUNIORBEES",  "name": "Nippon Nifty Next 50",           "category": "Large-Mid"},
    {"symbol": "MID150BEES",  "name": "Nippon Nifty Midcap 150",        "category": "Midcap"},
    {"symbol": "HDFCSMALL",   "name": "HDFC Nifty Smallcap 250",        "category": "Smallcap"},
    # Indian Sectoral
    {"symbol": "BANKBEES",    "name": "Nippon Nifty Bank BeES",          "category": "Banking"},
    {"symbol": "PSUBNKBEES",  "name": "Nippon Nifty PSU Bank BeES",      "category": "PSU Bank"},
    {"symbol": "ITBEES",      "name": "Nippon Nifty IT BeES",            "category": "IT/Tech"},
    {"symbol": "HEALTHIETF",  "name": "ICICI Nifty Healthcare ETF",      "category": "Healthcare"},
    {"symbol": "AUTOBEES",    "name": "Nippon Nifty Auto BeES",          "category": "Auto"},
    {"symbol": "METALIETF",   "name": "ICICI Nifty Metal ETF",           "category": "Metals"},
    {"symbol": "CONSUMBEES",  "name": "Nippon Nifty Consumption BeES",   "category": "Consumption"},
    {"symbol": "INFRABEES",   "name": "ICICI Nifty Infra ETF",           "category": "Infrastructure"},
    {"symbol": "OILIETF",     "name": "ICICI Nifty Oil & Gas ETF",       "category": "Energy"},
    {"symbol": "MOREALTY",    "name": "Motilal Nifty Realty ETF",        "category": "Realty"},
    # Indian Factor / Thematic
    {"symbol": "CPSEETF",     "name": "CPSE ETF",                        "category": "PSU/Govt"},
    {"symbol": "SETFMOMET",   "name": "Nippon Nifty200 Momentum30 ETF",  "category": "Momentum"},
    {"symbol": "QUAL30IETF",  "name": "HDFC Nifty100 Quality30 ETF",     "category": "Quality"},
    {"symbol": "KOTAKLOWV",   "name": "Kotak Nifty100 Low Vol30 ETF",    "category": "Low Vol"},
    {"symbol": "ALPL30IETF",  "name": "ICICI Alpha Low Vol30 ETF",       "category": "Alpha/LowVol"},
    {"symbol": "MODEFENCE",   "name": "Motilal Nifty India Defence ETF", "category": "Defence"},
    {"symbol": "MOMOMENTUM",  "name": "Nippon Nifty500 Momentum50 ETF",  "category": "Broad Momentum"},
    {"symbol": "MONIFTY500",  "name": "Motilal Nifty 500 ETF",           "category": "Broad 500"},
    # Indian Commodities
    {"symbol": "GOLDBEES",    "name": "Nippon Gold BeES",                "category": "Gold"},
    {"symbol": "SILVERBEES",  "name": "Nippon Silver BeES",              "category": "Silver"},
    {"symbol": "LIQUIDBEES",  "name": "Nippon Liquid BeES (Cash proxy)", "category": "Cash"},
    # Global — US/Intl (no .NS suffix)
    {"symbol": "MON100",      "name": "Motilal Nasdaq 100 ETF",          "category": "US Tech"},
    {"symbol": "SETFNN50",    "name": "SBI Nifty Next 50 ETF",           "category": "Large-Mid"},
    # MASPTOP50 removed — structural loser (-₹21K, 36% WR)
    # HNGSNGBEES removed — China/HK structural bear (-₹102K, 38% WR)
    {"symbol": "FRDM",        "name": "Freedom 100 EM ETF",              "category": "EM ex-Auth"},
    {"symbol": "SOXX",        "name": "iShares Semiconductor ETF",       "category": "Semiconductors"},
    {"symbol": "BOTZ",        "name": "Global X Robotics & AI ETF",      "category": "AI/Robotics"},
    {"symbol": "EWY",         "name": "iShares MSCI South Korea ETF",    "category": "South Korea"},
    {"symbol": "EMXC",        "name": "EM ex-China ETF",                 "category": "EM ex-China"},
    {"symbol": "AVDV",        "name": "Intl SmallCap Value ETF",         "category": "Intl SmallCap"},
    {"symbol": "ILF",         "name": "Latin America 40 ETF",            "category": "Latin America"},
    {"symbol": "XLE",         "name": "US Energy ETF",                   "category": "US Energy"},
    {"symbol": "GDX",         "name": "Gold Miners ETF",                 "category": "Gold Miners"},
    {"symbol": "XME",         "name": "Metals & Mining ETF",             "category": "Metals Mining"},
    {"symbol": "VGK",         "name": "Europe Broad ETF",                "category": "Europe"},
    {"symbol": "XLK",         "name": "US Technology ETF",               "category": "US Tech"},
    {"symbol": "XLP",         "name": "US Consumer Staples ETF",         "category": "US Cons Staples"},
    {"symbol": "TLT",         "name": "US LT Treasuries ETF",            "category": "US Bonds"},
    {"symbol": "XLV",         "name": "US Healthcare ETF",               "category": "US Healthcare"},
    {"symbol": "ITA",         "name": "US Aerospace & Defence ETF",      "category": "US Defence"},
    # EWU removed — UK structural weakness (-₹56K, 33% WR)
    # UCITS-only standalones (no US-listed equivalent in universe)
    {"symbol": "ISF",         "name": "iShares FTSE 100 (UK)",          "category": "UK Large-Cap"},
    {"symbol": "ISAC",        "name": "iShares MSCI ACWI (Global)",     "category": "Global All-Cap"},
]

ETF_MAP = {e["symbol"]: e for e in ETF_UNIVERSE}

# International ETFs — fetched without .NS suffix (US-listed)
# International ETFs for geography cap (MAX_INTL_SLOTS=3)
# MON100/MASPTOP50 are NSE-listed but count as international; the rest are US-listed (no .NS)
INTL_ETFS = {"MON100", "FRDM", "EMXC", "AVDV", "ILF", "XLE", "GDX", "XME", "VGK",
             "SOXX", "BOTZ", "EWY", "XLK", "XLP", "TLT", "XLV", "ITA",
             "ISF", "ISAC"}
US_LISTED_ETFS = {"FRDM", "EMXC", "AVDV", "ILF", "XLE", "GDX", "XME", "VGK",
                  "SOXX", "BOTZ", "EWY", "XLK", "XLP", "TLT", "XLV", "ITA"}  # no .NS suffix
LSE_ETFS = {"ISF", "ISAC"}  # .L suffix (LSE-listed)
MAX_INTL_SLOTS = 5  # geography cap

CORR_GROUPS = [
    {"GOLDBEES", "GDX"},  # physical gold vs gold miners — pick the higher ranked
]

def corr_group_blocked(sym, active_syms, pending_syms, rank_map):
    """Return True if a higher-ranked corr-group peer is already active or pending entry."""
    for group in CORR_GROUPS:
        if sym not in group:
            continue
        my_rank = rank_map.get(sym, 9999)
        for peer in group:
            if peer == sym:
                continue
            if peer in active_syms or peer in pending_syms:
                peer_rank = rank_map.get(peer, 9999)
                if peer_rank <= my_rank:
                    return True
    return False

# NSE display symbol → yfinance ticker (no .NS suffix here; .NS appended at fetch time for domestic)
ETF_YF_OVERRIDES = {
    # MID150BEES uses MID150BEES.NS directly — no override needed
    "HDFCSMALL":  "HDFCSML250",  # HDFC Nifty Smallcap 250
    "MOREALTY":   "MOREALTY",    # Motilal Nifty Realty ETF (same on yfinance)
    "KOTAKLOWV":  "LOWVOLIETF",  # Kotak Nifty100 Low Vol30
    "SETFMOMET":  "MOM30IETF",   # Nippon Nifty200 Momentum30
}

BENCHMARK = "^CNX200"
LIQUIDBEES_SYMBOL = "LIQUIDBEES"

# iNAV symbol overrides (NSE trading symbol for iNAV feed)
# Default pattern: symbol + "INAV" — override only when NSE uses a different name
INAV_OVERRIDES = {
    "MOREALTY":  "MOREALTYINAV",
}
# US-listed and LSE-listed ETFs have no iNAV
_NO_INAV = US_LISTED_ETFS | LSE_ETFS

DATA_STORE = Path(__file__).parent.parent / "data_store"
DATA_STORE.mkdir(exist_ok=True)

DEFAULT_STATE = {
    "slots": [
        {"slot_id": i, "state": "LIQUIDBEES", "symbol": None,
         "entry_price": None, "qty": 0, "entry_date": None, "capital": 200000}
        for i in range(1, 6)
    ],
    "watchlist": [],
    "trade_log": [],
    "last_scan": None,
    "last_scan_result": None,
}


class ETFEngine:
    """Live ETF Core signal engine with position management."""

    def __init__(self, state_file=None):
        if state_file is None:
            self.state_file = DATA_STORE / "etf_state.json"
        else:
            self.state_file = Path(state_file)
        self._ensure_file()

    # ── State Persistence ─────────────────────────────────────────────────────

    def _ensure_file(self):
        if not self.state_file.exists():
            self.state_file.write_text(json.dumps(DEFAULT_STATE, indent=2))

    def _load_state(self):
        with open(self.state_file) as f:
            return json.load(f)

    def _save_state(self, data):
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

    # ── iNAV Fetch ────────────────────────────────────────────────────────────

    def _fetch_inav(self, kite) -> dict:
        """Fetch live iNAV for all Indian ETFs via Kite quote API.
        Returns {symbol: inav_price} — None for symbols with no iNAV or on error."""
        inav_map = {}
        if kite is None:
            return inav_map

        # Build NSE:SYMBOLINAV list for Indian ETFs only
        symbols = [e["symbol"] for e in ETF_UNIVERSE if e["symbol"] not in _NO_INAV]
        inav_keys = {}  # kite_key → etf_symbol
        for sym in symbols:
            inav_sym = INAV_OVERRIDES.get(sym, sym + "INAV")
            inav_keys[f"NSE:{inav_sym}"] = sym

        if not inav_keys:
            return inav_map

        try:
            quotes = kite.quote(list(inav_keys.keys()))
            for kite_key, etf_sym in inav_keys.items():
                q = quotes.get(kite_key, {})
                ltp = q.get("last_price") or (q.get("ohlc") or {}).get("close")
                inav_map[etf_sym] = float(ltp) if ltp else None
        except Exception:
            pass  # iNAV is best-effort; don't fail the scan

        return inav_map

    def _fetch_vix(self) -> float | None:
        """Fetch latest India VIX value via yfinance."""
        try:
            df = yf.download("^INDIAVIX", period="5d", progress=False, auto_adjust=True)
            if not df.empty:
                return float(df["Close"].squeeze().dropna().iloc[-1])
        except Exception:
            pass
        return None

    # ── Indicator Computation ─────────────────────────────────────────────────

    def _compute_indicators(self, closes: pd.Series, bench_closes: pd.Series) -> dict:
        """Compute RS63 (5d SMA of raw_rs), SMA63, SMA20, SMA50, and 20-day rolling high."""
        if len(closes) < 68:  # need 63 + 5 for smoothing
            return {"rs63": None, "sma63": None, "sma20": None, "sma50": None,
                    "price": None, "raw_rs": None, "high20": None}

        # Align benchmark to ETF dates
        bench_aligned = bench_closes.reindex(closes.index, method="ffill")

        # raw_rs = (etf[i]/etf[i-63]) / (bench[i]/bench[i-63]) - 1
        raw_rs = (closes / closes.shift(63)) / (bench_aligned / bench_aligned.shift(63)) - 1

        rs63_series = raw_rs.rolling(5, min_periods=3).mean()
        sma63_series = closes.rolling(63, min_periods=63).mean()
        sma20_series = closes.rolling(20, min_periods=20).mean()
        sma50_series = closes.rolling(50, min_periods=50).mean()
        high20_series = closes.rolling(20, min_periods=10).max()

        def _val(s):
            v = s.iloc[-1]
            return float(v) if not pd.isna(v) else None

        return {
            "rs63": _val(rs63_series),
            "sma63": _val(sma63_series),
            "sma20": _val(sma20_series),
            "sma50": _val(sma50_series),
            "price": _val(closes),
            "raw_rs": _val(raw_rs),
            "high20": _val(high20_series),
        }

    def _lock_swing_high_b(self, symbol: str, exit_date_str: str, closes: pd.Series) -> float:
        """max close in the 15 trading days before exit_date (locked at exit time)."""
        try:
            exit_date = pd.Timestamp(exit_date_str)
            mask = closes.index < exit_date
            prior = closes[mask]
            if len(prior) == 0:
                return None
            last_15 = prior.iloc[-15:]
            return float(last_15.max())
        except Exception:
            return None

    # ── Data Fetching ─────────────────────────────────────────────────────────

    def _kite_daily_closes(self, kite, symbol, from_date, to_date):
        """Fetch daily close Series from Kite for a NSE symbol."""
        try:
            instruments = kite.instruments("NSE")
            token_map = {i["tradingsymbol"]: i["instrument_token"] for i in instruments}
            token = token_map.get(symbol)
            if not token:
                return pd.Series(dtype=float)
            candles = kite.historical_data(token, from_date, to_date, "day")
            if not candles:
                return pd.Series(dtype=float)
            df = pd.DataFrame(candles)
            df["date"] = pd.to_datetime(df["date"])
            if df["date"].dt.tz is not None:
                df["date"] = df["date"].dt.tz_localize(None)
            return df.set_index("date")["close"].rename(symbol)
        except Exception:
            return pd.Series(dtype=float)

    def _fetch_data(self, kite=None) -> tuple:
        """Fetch 150 days of daily closes for all 27 ETFs + benchmark.
        Uses Kite if provided, falls back to yfinance.
        Returns (etf_data dict, bench_series)."""
        import yfinance as yf
        from datetime import date, timedelta

        fetch_days = 160  # 150 trading days + buffer
        from_date = date.today() - timedelta(days=fetch_days)
        to_date = date.today()

        # ── Benchmark ─────────────────────────────────────────────────────────
        bench_closes = pd.Series(dtype=float)
        if kite is not None:
            bench_closes = self._kite_daily_closes(kite, "NIFTY 200", from_date, to_date)

        if len(bench_closes) < 5:
            bench_df = yf.download(BENCHMARK, period="150d", progress=False, auto_adjust=True)
            if bench_df.empty:
                raise RuntimeError(f"Could not fetch benchmark {BENCHMARK}")
            bench_closes = bench_df["Close"].squeeze().dropna()
            if bench_closes.index.tz is not None:
                bench_closes.index = bench_closes.index.tz_localize(None)

        # ── ETFs ──────────────────────────────────────────────────────────────
        etf_data = {}

        if kite is not None:
            # Fetch instruments list once for token lookup
            try:
                instruments = kite.instruments("NSE")
                token_map = {i["tradingsymbol"]: i["instrument_token"] for i in instruments}
            except Exception:
                token_map = {}

            for etf in ETF_UNIVERSE:
                sym = etf["symbol"]
                token = token_map.get(sym)
                closes = pd.Series(dtype=float)
                if token:
                    try:
                        candles = kite.historical_data(token, from_date, to_date, "day")
                        if candles:
                            df = pd.DataFrame(candles)
                            df["date"] = pd.to_datetime(df["date"])
                            if df["date"].dt.tz is not None:
                                df["date"] = df["date"].dt.tz_localize(None)
                            closes = df.set_index("date")["close"].rename(sym)
                    except Exception:
                        pass
                etf_data[sym] = closes
        else:
            etf_data = {}

        # yfinance fallback for any ETF with no Kite data
        missing_yf = [e["symbol"] for e in ETF_UNIVERSE if len(etf_data.get(e["symbol"], [])) < 5]
        if missing_yf:
            # Build yfinance symbols: international ETFs have no .NS suffix
            def _yf_sym(s):
                if s in US_LISTED_ETFS:
                    return s          # US-listed, no suffix
                if s in LSE_ETFS:
                    return s + ".L"   # LSE-listed UCITS
                return ETF_YF_OVERRIDES.get(s, s) + ".NS"

            yf_symbols = [_yf_sym(s) for s in missing_yf]
            try:
                batch_df = yf.download(yf_symbols, period="150d", progress=False,
                                       auto_adjust=True, group_by="ticker")
            except Exception:
                batch_df = None

            for sym in missing_yf:
                sym_yf = _yf_sym(sym)
                try:
                    closes = pd.Series(dtype=float)
                    if batch_df is not None and not batch_df.empty:
                        try:
                            if sym_yf in batch_df.columns.get_level_values(0):
                                closes = batch_df[sym_yf]["Close"].dropna()
                        except (KeyError, TypeError):
                            pass
                        if len(closes) == 0 and "Close" in batch_df.columns:
                            closes = batch_df["Close"].squeeze().dropna()
                    if len(closes) == 0:
                        single = yf.download(sym_yf, period="150d", progress=False, auto_adjust=True)
                        if not single.empty:
                            closes = single["Close"].squeeze().dropna()
                    if len(closes) > 0 and closes.index.tz is not None:
                        closes.index = closes.index.tz_localize(None)
                    etf_data[sym] = closes
                except Exception:
                    etf_data[sym] = pd.Series(dtype=float)

        return etf_data, bench_closes

    # ── Main Scan ─────────────────────────────────────────────────────────────

    def scan(self, kite=None) -> dict:
        """Full scan: fetch data, compute indicators, check signals, return result.
        Pass kite (KiteConnect instance) to use Zerodha data; falls back to yfinance."""
        state = self._load_state()
        scan_time = datetime.now().strftime("%Y-%m-%d %H:%M")

        try:
            etf_data, bench_closes = self._fetch_data(kite=kite)
        except Exception as e:
            return {"error": str(e), "scan_time": scan_time}

        # Fetch iNAV (best-effort, Kite only)
        inav_map = self._fetch_inav(kite)

        # Fetch India VIX (display only)
        vix_val = self._fetch_vix()

        # Compute indicators for all ETFs
        ranked_items = []
        for etf in ETF_UNIVERSE:
            sym = etf["symbol"]
            closes = etf_data.get(sym, pd.Series(dtype=float))

            inav = inav_map.get(sym)

            if len(closes) < 10:
                ranked_items.append({
                    "symbol": sym,
                    "name": etf["name"],
                    "category": etf["category"],
                    "price": None,
                    "sma63": None,
                    "rs63": None,
                    "inav": round(inav, 4) if inav else None,
                    "trend_ok": False,
                    "rs_ok": False,
                    "no_data": True,
                })
                continue

            ind = self._compute_indicators(closes, bench_closes)
            price = ind["price"]
            rs63 = ind["rs63"]
            sma63 = ind["sma63"]
            sma20 = ind["sma20"]
            sma50 = ind["sma50"]
            high20 = ind["high20"]

            trend_ok = (price is not None and sma63 is not None and price > sma63)
            rs_ok = (rs63 is not None and rs63 > 0)

            # X4: overextended check
            x4 = None
            if price is not None:
                if sma20 is not None and price > sma20 * 1.20:
                    x4 = f">20% above SMA20 ({price:.2f} vs {sma20:.2f})"
                elif sma50 is not None and price > sma50 * 1.30:
                    x4 = f">30% above SMA50 ({price:.2f} vs {sma50:.2f})"

            # iNAV premium/discount vs LTP (positive = LTP trading at premium to iNAV)
            inav_prem = None
            if inav and price:
                inav_prem = round((price / inav - 1) * 100, 3)

            ranked_items.append({
                "symbol": sym,
                "name": etf["name"],
                "category": etf["category"],
                "price": round(price, 2) if price else None,
                "sma63": round(sma63, 2) if sma63 else None,
                "sma20": round(sma20, 2) if sma20 else None,
                "sma50": round(sma50, 2) if sma50 else None,
                "rs63": round(rs63 * 100, 3) if rs63 is not None else None,  # as %
                "high20": round(high20, 2) if high20 else None,
                "inav": round(inav, 4) if inav else None,
                "inav_prem": inav_prem,
                "trend_ok": trend_ok,
                "rs_ok": rs_ok,
                "x4": x4,  # None or description string if overextended
                "no_data": False,
                "_closes": closes,  # temp for swing high calc
            })

        # Rank by RS63 descending (only those with valid RS63)
        # Non-UCITS always rank above UCITS regardless of RS63 value
        valid = [x for x in ranked_items if x.get("rs63") is not None]
        invalid = [x for x in ranked_items if x.get("rs63") is None]
        valid.sort(key=lambda x: (1 if x["symbol"] in LSE_ETFS else 0, -x["rs63"]))
        for i, item in enumerate(valid):
            item["rank"] = i + 1
        for item in invalid:
            item["rank"] = None

        all_items = valid + invalid

        # Build rank lookup
        rank_map = {item["symbol"]: item["rank"] for item in all_items}

        # Determine position symbols
        active_symbols = set(
            s["symbol"] for s in state["slots"]
            if s.get("state") == "ACTIVE" and s.get("symbol")
        )

        # Determine watchlist symbols (recently exited, waiting for re-entry)
        wl_symbols = {w["symbol"]: w for w in state.get("watchlist", [])}

        # Check exit signals for active positions
        exit_signals = []
        for slot in state["slots"]:
            if slot.get("state") != "ACTIVE" or not slot.get("symbol"):
                continue
            sym = slot["symbol"]
            item = next((x for x in all_items if x["symbol"] == sym), None)
            if not item:
                continue
            reasons = []
            if item["rs63"] is not None and item["rs63"] < 0:
                reasons.append("X1: RS63 < 0")
            if item.get("price") and item.get("sma63") and item["price"] < item["sma63"]:
                reasons.append("X2: price < SMA63")
            if item.get("price") and item.get("high20") and item["price"] < item["high20"] * 0.85:
                reasons.append("X3: -15% from 20d high")
            if item.get("x4"):
                reasons.append(f"X4: {item['x4']}")
            if reasons:
                exit_signals.append({
                    "symbol": sym,
                    "slot_id": slot["slot_id"],
                    "reasons": reasons,
                    "price": item.get("price"),
                })

        # Check new entry signals (E1) — top 5 by rank, rs_ok, trend_ok, not already active
        free_slots = sum(1 for s in state["slots"]
                         if s.get("state") in ("LIQUIDBEES", "FREE", None))
        intl_active = sum(1 for s in state["slots"]
                          if s.get("state") == "ACTIVE" and s.get("symbol") in INTL_ETFS)
        entry_signals = []
        pending_entry_syms: set[str] = set()  # built as we iterate
        for item in valid:
            if item["rank"] > 5:
                break
            if item["symbol"] in active_symbols:
                continue
            if item["symbol"] in wl_symbols:
                continue  # check re-entry separately
            if corr_group_blocked(item["symbol"], active_symbols, pending_entry_syms, rank_map):
                continue  # corr group: higher-ranked peer already active/pending
            if item["rs_ok"] and item["trend_ok"]:
                intl_blocked = item["symbol"] in INTL_ETFS and intl_active >= MAX_INTL_SLOTS
                pending_entry_syms.add(item["symbol"])
                entry_signals.append({
                    "symbol": item["symbol"],
                    "name": item["name"],
                    "rank": item["rank"],
                    "price": item["price"],
                    "rs63": item["rs63"],
                    "sma63": item["sma63"],
                    "intl_blocked": intl_blocked,
                })

        # Check re-entry signals for watchlist items
        reentry_signals = []
        for sym, wl in wl_symbols.items():
            item = next((x for x in all_items if x["symbol"] == sym), None)
            if not item or item["rank"] is None:
                continue
            if item["rank"] > 5:
                continue
            if not item["rs_ok"] or not item["trend_ok"]:
                continue
            if corr_group_blocked(sym, active_symbols, pending_entry_syms, rank_map):
                continue  # corr group: higher-ranked peer already active/pending
            swing_high = wl.get("swing_high_b")
            price = item.get("price")
            if price is None:
                continue
            # Price > SwingHighB OR within 0.5% above it
            breakout_ok = (swing_high is None or
                           price >= swing_high or
                           price >= swing_high * 0.995)
            if breakout_ok:
                pending_entry_syms.add(sym)
                reentry_signals.append({
                    "symbol": sym,
                    "name": item["name"],
                    "rank": item["rank"],
                    "price": price,
                    "rs63": item["rs63"],
                    "sma63": item["sma63"],
                    "swing_high_b": swing_high,
                    "exit_reason": wl.get("exit_reason"),
                })

        # Annotate items with position/watchlist status
        for item in all_items:
            sym = item["symbol"]
            item["in_position"] = sym in active_symbols
            item["in_watchlist"] = sym in wl_symbols
            item["swing_high_b"] = wl_symbols[sym]["swing_high_b"] if sym in wl_symbols else None

            # Determine display signal
            if item["in_position"]:
                exit_sym = [e for e in exit_signals if e["symbol"] == sym]
                if exit_sym:
                    first_reason = exit_sym[0]["reasons"][0]
                    if "X1" in first_reason:
                        item["signal"] = "EXIT_X1"
                    elif "X2" in first_reason:
                        item["signal"] = "EXIT_X2"
                    elif "X4" in first_reason:
                        item["signal"] = "EXIT_X4"
                    else:
                        item["signal"] = "EXIT_X3"
                else:
                    rank = item.get("rank")
                    item["signal"] = "HOLD" if (rank and rank <= 5) else ("BUFFER" if (rank and rank <= 9) else "HOLD")
            elif item["in_watchlist"]:
                re_sym = [r for r in reentry_signals if r["symbol"] == sym]
                item["signal"] = "RE_ENTRY" if re_sym else "WATCHLIST"
            elif item["rs_ok"] and item["trend_ok"] and item.get("rank") and item["rank"] <= 5:
                item["signal"] = "ENTRY"
            elif item.get("rank") and item["rank"] <= 9:
                item["signal"] = "RANKED"
            else:
                item["signal"] = "RANKED"

            # Remove temporary _closes field
            item.pop("_closes", None)

        # Build slots result
        slots_result = []
        for slot in state["slots"]:
            sym = slot.get("symbol")
            slot_out = dict(slot)
            if sym and slot.get("state") == "ACTIVE":
                item = next((x for x in all_items if x["symbol"] == sym), None)
                if item:
                    slot_out["current_price"] = item.get("price")
                    slot_out["rs63"] = item.get("rs63")
                    slot_out["sma63"] = item.get("sma63")
                    slot_out["sma20"] = item.get("sma20")
                    slot_out["sma50"] = item.get("sma50")
                    slot_out["x4"] = item.get("x4")
                    slot_out["trend_ok"] = item.get("trend_ok")
                    slot_out["rs_ok"] = item.get("rs_ok")
                    slot_out["rank"] = item.get("rank")
                    # P&L
                    if slot.get("entry_price") and item.get("price"):
                        entry = slot["entry_price"]
                        current = item["price"]
                        pnl_pct = (current - entry) / entry * 100
                        slot_out["pnl_pct"] = round(pnl_pct, 2)
                        slot_out["pnl_abs"] = round((current - entry) * (slot.get("qty") or 0), 2)
                    else:
                        slot_out["pnl_pct"] = None
                        slot_out["pnl_abs"] = None
                    exit_sym = [e for e in exit_signals if e["symbol"] == sym]
                    slot_out["exit_alert"] = exit_sym[0]["reasons"] if exit_sym else []
                else:
                    slot_out["current_price"] = None
                    slot_out["exit_alert"] = []
            else:
                slot_out["current_price"] = None
                slot_out["exit_alert"] = []
            slots_result.append(slot_out)

        result = {
            "scan_time": scan_time,
            "watchlist": all_items,
            "slots": slots_result,
            "exit_signals": exit_signals,
            "entry_signals": entry_signals,
            "reentry_signals": reentry_signals,
            "vix": round(vix_val, 2) if vix_val is not None else None,
        }

        # Cache last scan
        state["last_scan"] = scan_time
        # Store a lightweight version (no _closes)
        state["last_scan_result"] = {
            "scan_time": scan_time,
            "exit_signals": exit_signals,
            "entry_signals": entry_signals,
            "reentry_signals": reentry_signals,
        }
        self._save_state(state)

        return result

    # ── Position Management ───────────────────────────────────────────────────

    def enter_position(self, symbol: str, price: float, qty: float, reason: str = "ENTRY") -> tuple:
        """Enter a position in the first free slot."""
        state = self._load_state()

        # Find free slot
        free_slot = None
        for slot in state["slots"]:
            if slot.get("state") in ("LIQUIDBEES", "FREE", None):
                free_slot = slot
                break

        if not free_slot:
            return None, "No free slots available"

        etf_info = ETF_MAP.get(symbol, {})
        entry_date = date.today().isoformat()

        free_slot["state"] = "ACTIVE"
        free_slot["symbol"] = symbol
        free_slot["entry_price"] = round(float(price), 2)
        free_slot["qty"] = round(float(qty), 6)
        free_slot["entry_date"] = entry_date

        # Remove from watchlist if present
        state["watchlist"] = [w for w in state["watchlist"] if w["symbol"] != symbol]

        # Log trade
        log_entry = {
            "date": entry_date,
            "symbol": symbol,
            "name": etf_info.get("name", symbol),
            "action": "ENTRY",
            "price": round(float(price), 2),
            "qty": round(float(qty), 6),
            "capital": free_slot["capital"],
            "reason": reason,
            "slot_id": free_slot["slot_id"],
            "rs63": None,
            "sma63": None,
            "rank": None,
            "swing_high_b": None,
            "pnl": None,
        }
        state["trade_log"].append(log_entry)
        self._save_state(state)
        return free_slot, None

    def exit_position(self, symbol: str, exit_price: float, reason: str = "MANUAL",
                      closes: pd.Series = None) -> tuple:
        """Exit a position, lock SwingHighB, move to watchlist."""
        state = self._load_state()

        # Find the slot
        target_slot = None
        for slot in state["slots"]:
            if slot.get("state") == "ACTIVE" and slot.get("symbol") == symbol:
                target_slot = slot
                break

        if not target_slot:
            return None, f"No active position for {symbol}"

        exit_date = date.today().isoformat()
        entry_price = target_slot.get("entry_price") or 0
        qty = target_slot.get("qty") or 0

        # Compute P&L (gross, no tax/charges for paper tracking)
        gross_pnl = round((float(exit_price) - entry_price) * qty, 2)

        # Lock SwingHighB
        swing_high = None
        if closes is not None and len(closes) > 0:
            swing_high = self._lock_swing_high_b(symbol, exit_date, closes)

        # Add to watchlist
        wl_entry = {
            "symbol": symbol,
            "exit_date": exit_date,
            "exit_price": round(float(exit_price), 2),
            "swing_high_b": swing_high,
            "exit_reason": reason,
        }
        # Remove existing entry for same symbol (in case of duplicate)
        state["watchlist"] = [w for w in state["watchlist"] if w["symbol"] != symbol]
        state["watchlist"].append(wl_entry)

        # Log trade
        etf_info = ETF_MAP.get(symbol, {})
        log_entry = {
            "date": exit_date,
            "symbol": symbol,
            "name": etf_info.get("name", symbol),
            "action": "EXIT",
            "price": round(float(exit_price), 2),
            "qty": qty,
            "capital": target_slot["capital"],
            "reason": reason,
            "slot_id": target_slot["slot_id"],
            "rs63": None,
            "sma63": None,
            "rank": None,
            "swing_high_b": swing_high,
            "pnl": gross_pnl,
            "entry_price": entry_price,
            "entry_date": target_slot.get("entry_date"),
        }
        state["trade_log"].append(log_entry)

        # Reset slot
        target_slot["state"] = "LIQUIDBEES"
        target_slot["symbol"] = None
        target_slot["entry_price"] = None
        target_slot["qty"] = 0
        target_slot["entry_date"] = None

        self._save_state(state)
        return log_entry, None

    def modify_position(self, symbol: str, price: float | None, qty: float | None) -> tuple:
        """Update entry_price and/or qty for an active position."""
        state = self._load_state()
        target_slot = None
        for slot in state["slots"]:
            if slot.get("state") == "ACTIVE" and slot.get("symbol") == symbol:
                target_slot = slot
                break
        if not target_slot:
            return None, f"No active position for {symbol}"
        if price is not None:
            target_slot["entry_price"] = round(float(price), 2)
        if qty is not None:
            target_slot["qty"] = round(float(qty), 6)
        self._save_state(state)
        return target_slot, None

    def get_positions(self) -> list:
        """Return all 5 slots."""
        state = self._load_state()
        return state["slots"]

    def get_history(self) -> list:
        """Return closed trade log."""
        state = self._load_state()
        return state.get("trade_log", [])

    def get_status(self) -> dict:
        """Return full status: slots, watchlist, trade log, last scan."""
        state = self._load_state()
        return {
            "slots": state["slots"],
            "watchlist": state.get("watchlist", []),
            "trade_log": state.get("trade_log", []),
            "last_scan": state.get("last_scan"),
            "last_scan_result": state.get("last_scan_result"),
        }
