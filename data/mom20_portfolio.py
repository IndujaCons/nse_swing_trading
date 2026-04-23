"""
Mom20 Portfolio Manager
=======================
Tracks the Mom20 basket with two tracking modes:
  - paper:    tracking from saved prices (not yet invested)
  - invested: tracking from actual Kite avg prices after investment

Portfolio JSON structure:
{
  "capital": 2000000,
  "status": "empty" | "paper" | "invested",
  "tracking_since": "2026-04-23",
  "basket": [
    {"ticker": "TITAN", "weight": 5,
     "saved_price": 4441.20,      # price at Save time
     "invested_price": null}      # avg price from Kite (set after Confirm Investment)
  ],
  "pending_orders": []
}
"""

import os
import json
import math
from datetime import date, timedelta

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORTFOLIO_FILE = os.path.join(_BASE_DIR, "data_store", "mom20_portfolio.json")

_DEFAULT = {
    "capital": 2000000,
    "status": "empty",
    "tracking_since": None,
    "basket": [],
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
    today = date.today()
    first = date(today.year + 1, 1, 1) if today.month == 12 else date(today.year, today.month + 1, 1)
    while first.weekday() >= 5:
        first += timedelta(days=1)
    return first


def days_to_rebalance() -> int:
    return (get_next_rebalance_date() - date.today()).days


def calc_performance(portfolio: dict, current_prices: dict) -> dict:
    """
    Compute portfolio performance vs baseline (saved or invested prices).
    current_prices: {ticker: current_price}
    Returns: {index_value, return_pct, day_return_pct, tracking_since,
              status, min_investment, stocks}
    """
    basket = portfolio.get("basket", [])
    status = portfolio.get("status", "empty")
    capital = portfolio.get("capital", 2000000)

    if not basket or status == "empty":
        return {"status": status, "index_value": 100.0, "return_pct": 0.0}

    use_invested = (status == "invested")

    total_weight = sum(s["weight"] for s in basket)
    if total_weight == 0:
        return {"status": status, "index_value": 100.0, "return_pct": 0.0}

    weighted_return = 0.0
    stocks = []
    for s in basket:
        baseline = s.get("invested_price") if use_invested else s.get("saved_price")
        current  = current_prices.get(s["ticker"], baseline)
        if not baseline or not current:
            ret = 0.0
        else:
            ret = current / baseline - 1
        w_frac = s["weight"] / total_weight
        weighted_return += w_frac * ret

        # Min investment: capital such that floor(cap × w% / price) >= 1
        # => cap >= price / (w/100)
        min_cap_this = (current / (s["weight"] / 100)) if current and s["weight"] else 0
        shares = int(math.floor(capital * (s["weight"] / 100) / current)) if current else 0

        stocks.append({
            "ticker": s["ticker"],
            "weight": s["weight"],
            "baseline": round(baseline, 2) if baseline else None,
            "current": round(current, 2) if current else None,
            "return_pct": round(ret * 100, 2),
            "shares": shares,
            "min_cap": round(min_cap_this),
        })

    index_value = round(100 * (1 + weighted_return), 2)
    return_pct  = round(weighted_return * 100, 2)
    min_investment = max((s["min_cap"] for s in stocks), default=0)

    return {
        "status": status,
        "index_value": index_value,
        "return_pct": return_pct,
        "tracking_since": portfolio.get("tracking_since"),
        "min_investment": min_investment,
        "capital": capital,
        "stocks": stocks,
    }
