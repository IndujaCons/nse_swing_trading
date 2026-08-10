"""
Dividend income tracking — Mom20 tracker.

Two layers:
  1. A shared, cross-user cache of per-ticker dividend history (dividend history
     is per-instrument, not per-user, so many users holding the same stock
     shouldn't each trigger their own yfinance fetch). See get_dividend_history(s).
  2. A per-user ledger that walks each user's holding episodes (open positions +
     closed/historical trades) and credits dividends against the quantity actually
     held on each ex-dividend date. See build_hold_episodes / compute_dividend_ledger.

Entitlement rule (the one easy off-by-one to get backwards): for a dividend with
ex-date d, a holding qualifies if entry_date < d <= (exit_date or today) — strictly
AFTER entry (must have held through the prior close to be entitled), inclusive
THROUGH exit (still entitled on the day you sell).
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import yfinance as yf

from data.user_registry import (
    dividend_cache_path, mom20_dividends_path,
    mom20_history_path, mom20_portfolio_path,
)

IST = timezone(timedelta(hours=5, minutes=30))
DIVIDEND_TTL_HOURS = 24


def _now_ist_iso() -> str:
    return datetime.now(IST).isoformat(timespec="seconds")


def _norm_date(raw) -> str:
    """First 10 chars of a raw date (may carry a time component, e.g. Zerodha's
    'Time' column), validated as YYYY-MM-DD. Returns '' if unparseable."""
    import re
    s = (raw or "")[:10]
    return s if re.match(r"^\d{4}-\d{2}-\d{2}$", s) else ""


def _yf_symbols(ticker: str) -> list:
    """Primary Yahoo symbol first, then a TICKER_ALIASES fallback if one exists
    and differs from the primary — only tried when the primary returns nothing
    (see get_dividend_histories). TICKER_ALIASES is bidirectional and not always
    "the current symbol" (e.g. ETERNAL->ZOMATO.NS is a delisted-symbol fallback
    for historical PIT lookups, not a forward mapping) — hence fallback-only,
    never tried first."""
    primary = f"{ticker}.NS"
    symbols = [primary]
    try:
        from data.momentum_backtest import TICKER_ALIASES
        alias = TICKER_ALIASES.get(ticker)
        if alias:
            alias_sym = alias if "." in alias else f"{alias}.NS"
            if alias_sym != primary:
                symbols.append(alias_sym)
    except Exception:
        pass
    return symbols


def _load_cache() -> dict:
    try:
        with open(dividend_cache_path()) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    path = dividend_cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=2)
    os.replace(tmp, path)


def _is_fresh(entry: dict) -> bool:
    fetched_at = entry.get("fetched_at")
    if not fetched_at:
        return False
    try:
        fetched = datetime.fromisoformat(fetched_at)
    except Exception:
        return False
    return (datetime.now(IST) - fetched) < timedelta(hours=DIVIDEND_TTL_HOURS)


def _fetch_one(ticker: str) -> dict:
    """Returns {"fetched_at", "symbol_used", "dividends": {date_str: per_share}}
    or None on a fetch exception on every symbol tried (caller keeps it out of
    the cache so it's retried next time, rather than caching a transient
    failure as "no dividends")."""
    symbols = _yf_symbols(ticker)
    for i, sym in enumerate(symbols):
        is_last = i == len(symbols) - 1
        try:
            s = yf.Ticker(sym).dividends
        except Exception:
            if is_last:
                return None
            continue
        divs = {d.strftime("%Y-%m-%d"): round(float(v), 4) for d, v in s.items() if v}
        if divs or is_last:
            # Non-empty result, OR this was the last symbol to try (including a
            # genuinely-empty-but-successful fetch — a real non-payer, cache it
            # as {} so it isn't refetched every click).
            return {"fetched_at": _now_ist_iso(), "symbol_used": sym, "dividends": divs}
    return None


def get_dividend_histories(tickers: list, refresh: bool = False) -> tuple:
    """Returns (histories, failed) where histories = {ticker: {date_str: per_share}}
    and failed = [ticker, ...] for tickers whose fetch raised on every symbol tried
    (kept out of the cache entirely so they're retried, not silently treated as
    non-payers)."""
    tickers = sorted(set(t for t in tickers if t))
    cache = _load_cache()
    need_fetch = [t for t in tickers if refresh or t not in cache or not _is_fresh(cache[t])]

    if need_fetch:
        with ThreadPoolExecutor(max_workers=10) as ex:
            results = dict(zip(need_fetch, ex.map(_fetch_one, need_fetch)))
        for t, r in results.items():
            if r is not None:
                cache[t] = r
        _save_cache(cache)

    histories = {}
    failed = []
    for t in tickers:
        entry = cache.get(t)
        if entry is None:
            failed.append(t)
        else:
            histories[t] = entry.get("dividends", {})
    return histories, failed


def get_dividend_history(ticker: str, refresh: bool = False) -> dict:
    histories, _ = get_dividend_histories([ticker], refresh=refresh)
    return histories.get(ticker, {})


# ── Hold-episode reconstruction ──────────────────────────────────────────────

def build_hold_episodes(user_id: str) -> list:
    """One entry per holding episode (a continuous entry-to-exit window for one
    ticker), for both closed trades (from mom20_history.json) and currently-open
    positions (from mom20_portfolio.json). Each episode:

        {"ticker", "qty", "entry_date", "exit_date": str|None,
         "lots": [(date_str, qty), ...], "qty_source": "history"|"flat_fallback",
         "incomplete": bool}

    lots is the chronological tranche timeline (original entry + any top-ups),
    used by compute_dividend_ledger to credit each dividend against the qty
    actually held on that ex-date — a topped-up position's pre-top-up dividends
    only credit the smaller original quantity. qty_source is "flat_fallback"
    when no reliable per-tranche timeline exists (manually seeded positions with
    no trade-book history, or a history/portfolio qty mismatch — see below) —
    in that case the episode's single qty applies across the whole window,
    which can overstate pre-top-up dividends; this is a real, marked limitation,
    not silently blended with real per-lot figures.
    """
    try:
        with open(mom20_history_path(user_id)) as f:
            history = json.load(f)
    except Exception:
        history = []

    episodes = []
    open_book = {}  # ticker -> {"entry_date": str, "lots": [(date, qty), ...]}

    for rb in history:
        rebal_date = rb.get("rebalance_date", "")

        for sell in rb.get("sells", []):
            ticker = sell.get("ticker")
            if not ticker:
                continue
            exit_date = _norm_date(sell.get("trade_date")) or rebal_date
            book = open_book.pop(ticker, None)
            if book is not None:
                episodes.append({
                    "ticker": ticker, "qty": sell.get("qty", 0),
                    "entry_date": book["entry_date"], "exit_date": exit_date,
                    "lots": list(book["lots"]), "qty_source": "history",
                    "incomplete": False,
                })
            else:
                # No prior buy seen in history — position existed before history
                # tracking began. Fall back to the sell's own enriched entry_date
                # (set at upload time, ui/app.py:2972) if present.
                entry_date = _norm_date(sell.get("entry_date"))
                incomplete = not entry_date
                if incomplete:
                    entry_date = exit_date  # zero-length window -> ₹0, not a guess
                episodes.append({
                    "ticker": ticker, "qty": sell.get("qty", 0),
                    "entry_date": entry_date, "exit_date": exit_date,
                    "lots": [(entry_date, sell.get("qty", 0))],
                    "qty_source": "flat_fallback", "incomplete": incomplete,
                })

        for t in rb.get("buys", []) + rb.get("top_ups", []):
            ticker = t.get("ticker")
            qty = t.get("qty", 0)
            if not ticker or qty <= 0:
                continue
            d = _norm_date(t.get("trade_date")) or rebal_date
            if ticker not in open_book:
                open_book[ticker] = {"entry_date": d, "lots": []}
            open_book[ticker]["lots"].append((d, qty))

    # Currently-open positions — mom20_portfolio.json's basket is authoritative
    # for qty/status; history (open_book) supplies the per-tranche timeline when
    # it agrees with that authoritative qty.
    try:
        with open(mom20_portfolio_path(user_id)) as f:
            portfolio = json.load(f)
    except Exception:
        portfolio = {}

    for h in portfolio.get("basket", []) or []:
        ticker = h.get("ticker")
        qty = h.get("qty", 0)
        entry_date = _norm_date(h.get("entry_date"))
        if not ticker or qty <= 0:
            continue
        book = open_book.get(ticker)
        if book is not None and sum(q for _, q in book["lots"]) == qty:
            episodes.append({
                "ticker": ticker, "qty": qty,
                "entry_date": book["entry_date"] or entry_date, "exit_date": None,
                "lots": list(book["lots"]), "qty_source": "history",
                "incomplete": False,
            })
        else:
            # No history for this ticker (manually seeded), or history/portfolio
            # qty disagree (deleted history record, manual edit) — trust the
            # portfolio's qty rather than presenting a mismatched timeline as if
            # it were authoritative.
            incomplete = not entry_date
            episodes.append({
                "ticker": ticker, "qty": qty,
                "entry_date": entry_date or date.today().isoformat(), "exit_date": None,
                "lots": [(entry_date or date.today().isoformat(), qty)],
                "qty_source": "flat_fallback", "incomplete": incomplete,
            })

    return episodes


# ── Ledger computation ───────────────────────────────────────────────────────

_EMPTY_LEDGER = {
    "updated_at": None, "method": "timeline",
    "open": {}, "closed": {}, "totals": {"open": 0.0, "closed": 0.0, "grand": 0.0},
    "coverage": {"requested": 0, "with_data": 0, "failed": []},
}


def compute_dividend_ledger(user_id: str, refresh: bool = True) -> dict:
    """Build and persist the per-user dividend ledger. See module docstring for
    the entitlement rule. Qty at an ex-date is summed from the episode's lots
    strictly before that ex-date (timeline episodes) or the flat episode qty
    (flat_fallback episodes)."""
    episodes = build_hold_episodes(user_id)
    tickers = sorted(set(e["ticker"] for e in episodes))
    histories, failed = get_dividend_histories(tickers, refresh=refresh)

    today_str = date.today().isoformat()
    open_ledger = {}
    closed_ledger = {}

    for ep in episodes:
        ticker = ep["ticker"]
        entry_date = ep["entry_date"]
        exit_date = ep["exit_date"]
        window_end = exit_date or today_str
        div_history = histories.get(ticker, {})

        events = []
        for ex_date, per_share in div_history.items():
            if not (entry_date < ex_date <= window_end):
                continue
            if ep["qty_source"] == "history":
                qty_at_ex = sum(q for lot_date, q in ep["lots"] if lot_date < ex_date)
            else:
                qty_at_ex = ep["qty"]
            if qty_at_ex <= 0:
                continue
            amount = round(qty_at_ex * per_share, 2)
            events.append({"ex_date": ex_date, "per_share": per_share,
                            "qty": qty_at_ex, "amount": amount})

        events.sort(key=lambda e: e["ex_date"])
        total = round(sum(e["amount"] for e in events), 2)
        record = {
            "ticker": ticker, "entry_date": entry_date, "exit_date": exit_date,
            "qty": ep["qty"], "dividend": total, "qty_source": ep["qty_source"],
            "incomplete": ep["incomplete"], "events": events,
        }

        if exit_date is None:
            # An open ticker can only appear once in episodes (one row per
            # basket entry), so no merge needed.
            open_ledger[ticker] = record
        else:
            key = f"{ticker}|{exit_date}"
            closed_ledger[key] = record

    totals = {
        "open": round(sum(r["dividend"] for r in open_ledger.values()), 2),
        "closed": round(sum(r["dividend"] for r in closed_ledger.values()), 2),
    }
    totals["grand"] = round(totals["open"] + totals["closed"], 2)

    ledger = {
        "updated_at": _now_ist_iso(),
        "method": "timeline",
        "open": open_ledger,
        "closed": closed_ledger,
        "totals": totals,
        "coverage": {
            "requested": len(tickers),
            "with_data": len(histories),
            "failed": failed,
        },
    }

    path = mom20_dividends_path(user_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ledger, f, indent=2)
    os.replace(tmp, path)

    return ledger


def load_dividend_ledger(user_id: str) -> dict:
    """Zero-network read of the persisted ledger. Safe empty-shape default so
    callers never need to branch on the file existing."""
    try:
        with open(mom20_dividends_path(user_id)) as f:
            return json.load(f)
    except Exception:
        return dict(_EMPTY_LEDGER)
