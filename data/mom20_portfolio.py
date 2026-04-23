"""
Mom20 Portfolio Manager
=======================
Tracks the Mom20 equal-weight portfolio state, calculates monthly rebalance
diffs, and places CNC orders via Kite on demand.
"""

import os
import json
import math
from datetime import date, timedelta

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORTFOLIO_FILE = os.path.join(_BASE_DIR, "data_store", "mom20_portfolio.json")

_DEFAULT = {
    "capital": 2000000,
    "last_rebalance_date": None,
    "holdings": [],
    "pending_orders": [],
}


def load_portfolio() -> dict:
    if not os.path.exists(PORTFOLIO_FILE):
        return dict(_DEFAULT)
    try:
        with open(PORTFOLIO_FILE) as f:
            data = json.load(f)
        return {**_DEFAULT, **data}
    except Exception:
        return dict(_DEFAULT)


def save_portfolio(state: dict):
    tmp = PORTFOLIO_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, PORTFOLIO_FILE)


def get_next_rebalance_date() -> date:
    """First weekday of next month (approximate — no NSE holiday calendar)."""
    today = date.today()
    if today.month == 12:
        first = date(today.year + 1, 1, 1)
    else:
        first = date(today.year, today.month + 1, 1)
    while first.weekday() >= 5:  # skip Sat/Sun
        first += timedelta(days=1)
    return first


def days_to_rebalance() -> int:
    return (get_next_rebalance_date() - date.today()).days


def calc_rebalance_diff(signals: list, portfolio: dict, regime_on: bool) -> dict:
    """
    signals: mom20_signals from live scan (rank 1..40)
    portfolio: loaded portfolio dict
    regime_on: blocks BUY when False
    Returns: {sell, hold, buy, per_slot, est_charges, regime_on}
    """
    capital = portfolio.get("capital", 2000000)
    holdings = portfolio.get("holdings", [])
    per_slot = capital / 20

    rank_map = {s["ticker"]: s for s in signals}

    sell = []
    hold = []
    hold_tickers = set()

    for h in holdings:
        ticker = h["ticker"]
        sig = rank_map.get(ticker)
        rank = sig["rank"] if sig else 999
        price = sig["price"] if sig else h.get("entry_price", 0)
        entry = h.get("entry_price", 0)
        pnl = round((price / entry - 1) * 100, 1) if entry else 0

        if rank > 40:
            sell.append({
                "ticker": ticker,
                "shares": h["shares"],
                "price": price,
                "amount": round(h["shares"] * price),
                "entry_price": entry,
                "pnl_pct": pnl,
                "rank": rank,
            })
        else:
            hold.append({
                "ticker": ticker,
                "shares": h["shares"],
                "price": price,
                "pnl_pct": pnl,
                "rank": rank,
            })
            hold_tickers.add(ticker)

    free_slots = max(0, 20 - len(hold))
    buy = []
    if regime_on and free_slots > 0:
        for s in signals:
            if s["rank"] > 15:
                break
            if s["ticker"] in hold_tickers:
                continue
            shares = int(math.floor(per_slot / s["price"])) if s["price"] > 0 else 0
            if shares < 1:
                continue
            buy.append({
                "ticker": s["ticker"],
                "shares": shares,
                "price": s["price"],
                "amount": round(shares * s["price"]),
                "rank": s["rank"],
            })
            if len(buy) >= free_slots:
                break

    # Zerodha charges (CLAUDE.md formula)
    buy_tv  = sum(b["amount"] for b in buy)
    sell_tv = sum(s["amount"] for s in sell)
    total_tv = buy_tv + sell_tv
    if total_tv > 0:
        stt      = 0.001    * total_tv
        exchange = 0.0000307 * total_tv
        sebi     = 0.000001  * total_tv
        stamp    = 0.00015   * buy_tv
        gst      = 0.18 * (exchange + sebi)
        est_charges = round(stt + exchange + sebi + stamp + gst)
    else:
        est_charges = 0

    return {
        "sell": sell,
        "hold": hold,
        "buy": buy,
        "per_slot": round(per_slot),
        "est_charges": est_charges,
        "regime_on": regime_on,
        "next_rebalance": str(get_next_rebalance_date()),
        "days_to_rebalance": days_to_rebalance(),
    }
