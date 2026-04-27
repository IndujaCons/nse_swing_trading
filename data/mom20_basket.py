"""
Mom20 Basket Generator
======================
Generates Zerodha basket order CSV for a user's Mom20 rebalance.

Basket CSV format (Zerodha):
  Tradingsymbol, Exchange, Transaction type, Order type, Quantity, Price, Product
  BHARATFORG, NSE, BUY, MARKET, 12, 0, CNC

Sizing:
  capital_per_slot = total_capital / n_slots   (equal weight)
  qty = floor(capital_per_slot / current_price)
"""

import csv
import io
import math
import os
import json
from datetime import date

N_SLOTS    = 20   # Mom20 top-N
BUFFER_OUT = 40   # hold if rank <= this


def generate_basket(user: dict, signals: list, current_portfolio: dict) -> dict:
    """
    Compute exits + entries for a Mom20 rebalance and return basket data.

    Args:
        user:              user dict from user_registry (has strategies.mom20.capital)
        signals:           list of mom20_signals dicts from live_signals_engine
                           each has: ticker, rank, price, momentum_score
        current_portfolio: user's mom20_portfolio.json dict (has basket list)

    Returns dict with:
        exits:        [{ticker, current_price, qty, reason}]
        entries:      [{ticker, rank, price, qty, capital_allocated, score}]
        holds:        [{ticker, rank, price}]
        min_capital:  minimum capital to buy 1 share of each entry stock
        capital_per_slot: total_capital / N_SLOTS
        total_capital: from user config
        basket_date:  today
        summary:      human-readable string
    """
    total_capital  = user["strategies"]["mom20"]["capital"]
    capital_per_slot = total_capital / N_SLOTS

    # Build rank map from signals
    rank_map  = {s["ticker"]: s["rank"]  for s in signals}
    price_map = {s["ticker"]: s.get("price", 0) for s in signals}
    score_map = {s["ticker"]: s.get("momentum_score", 0) for s in signals}

    current_tickers = {item["ticker"] for item in current_portfolio.get("basket", [])}
    current_qty_map = {item["ticker"]: item.get("qty", 0)
                       for item in current_portfolio.get("basket", [])}

    # Top-15 entry universe (buffer_in=15), hold if rank <= 40
    top15 = {s["ticker"] for s in signals if s["rank"] <= 15}
    hold_set = {s["ticker"] for s in signals if s["rank"] <= BUFFER_OUT}

    exits  = []
    holds  = []
    entries = []

    # Exits: currently held but rank > buffer_out OR not in signals at all
    for ticker in current_tickers:
        rank = rank_map.get(ticker, 999)
        price = price_map.get(ticker, 0)
        qty = current_qty_map.get(ticker, 0)
        if rank > BUFFER_OUT:
            exits.append({
                "ticker": ticker,
                "rank": rank,
                "current_price": price,
                "qty": qty,
                "reason": f"rank {rank} > {BUFFER_OUT}",
            })
        else:
            holds.append({"ticker": ticker, "rank": rank, "price": price})

    # Entries: top-15 not already held
    for ticker in top15:
        if ticker not in current_tickers:
            price = price_map.get(ticker, 0)
            qty = int(math.floor(capital_per_slot / price)) if price > 0 else 0
            entries.append({
                "ticker": ticker,
                "rank": rank_map[ticker],
                "price": price,
                "qty": qty,
                "capital_allocated": round(qty * price, 2),
                "score": round(score_map.get(ticker, 0), 3),
            })

    entries.sort(key=lambda x: x["rank"])

    min_capital = sum(e["price"] for e in entries if e["price"] > 0)
    total_deployed = sum(e["capital_allocated"] for e in entries)

    return {
        "basket_date":     date.today().isoformat(),
        "total_capital":   total_capital,
        "capital_per_slot": round(capital_per_slot, 2),
        "min_capital":     round(min_capital, 2),
        "total_deployed":  round(total_deployed, 2),
        "n_exits":         len(exits),
        "n_entries":       len(entries),
        "n_holds":         len(holds),
        "exits":           exits,
        "entries":         entries,
        "holds":           holds,
    }


def to_zerodha_csv(basket_data: dict) -> str:
    """
    Convert basket data to Zerodha basket order CSV string.
    Exits first (SELL), then entries (BUY). Market order + CNC.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Tradingsymbol", "Exchange", "Transaction type",
        "Order type", "Quantity", "Price", "Product"
    ])

    for e in basket_data["exits"]:
        if e["qty"] > 0:
            writer.writerow([e["ticker"], "NSE", "SELL", "MARKET", e["qty"], 0, "CNC"])

    for e in basket_data["entries"]:
        if e["qty"] > 0:
            writer.writerow([e["ticker"], "NSE", "BUY", "MARKET", e["qty"], 0, "CNC"])

    return output.getvalue()


def parse_trade_book(csv_content: str) -> list:
    """
    Parse Zerodha trade book CSV export.
    Returns list of trade dicts with: ticker, action, qty, price, trade_date
    """
    trades = []
    reader = csv.DictReader(io.StringIO(csv_content))
    for row in reader:
        # Zerodha trade book columns vary — handle common formats
        ticker = (row.get("tradingsymbol") or row.get("Tradingsymbol") or
                  row.get("Symbol") or "").strip().upper()
        action = (row.get("trade_type") or row.get("Transaction type") or
                  row.get("Type") or "").strip().upper()
        qty_raw   = row.get("quantity") or row.get("Quantity") or "0"
        price_raw = row.get("price") or row.get("Price") or "0"
        date_raw  = (row.get("trade_date") or row.get("Order Date") or
                     row.get("Date") or "").strip()

        try:
            qty   = int(float(qty_raw))
            price = float(price_raw)
        except (ValueError, TypeError):
            continue

        if not ticker or not action or qty <= 0:
            continue

        trades.append({
            "ticker":     ticker,
            "action":     "BUY" if "BUY" in action else "SELL",
            "qty":        qty,
            "price":      round(price, 2),
            "trade_date": date_raw,
        })
    return trades


def sync_portfolio_from_trades(portfolio: dict, trades: list, basket_data: dict) -> dict:
    """
    Apply trade book fills to portfolio.
    - SELL trades: remove from basket
    - BUY trades: add/update in basket with actual fill price

    Returns updated portfolio dict.
    """
    basket = {item["ticker"]: item for item in portfolio.get("basket", [])}

    # Group trades by ticker+action (handle partial fills)
    from collections import defaultdict
    grouped = defaultdict(list)
    for t in trades:
        grouped[(t["ticker"], t["action"])].append(t)

    # Process sells
    for (ticker, action), trade_list in grouped.items():
        if action == "SELL" and ticker in basket:
            del basket[ticker]

    # Process buys — weighted average price if multiple fills
    for (ticker, action), trade_list in grouped.items():
        if action == "BUY":
            total_qty   = sum(t["qty"] for t in trade_list)
            avg_price   = sum(t["qty"] * t["price"] for t in trade_list) / total_qty
            trade_date  = trade_list[0]["trade_date"]
            # Find entry rank/score from basket_data
            entry_info  = next((e for e in basket_data.get("entries", [])
                                if e["ticker"] == ticker), {})
            basket[ticker] = {
                "ticker":       ticker,
                "qty":          total_qty,
                "entry_price":  round(avg_price, 2),
                "entry_date":   trade_date,
                "weight":       round(100 / N_SLOTS, 2),
                "rank_at_entry": entry_info.get("rank"),
                "score_at_entry": entry_info.get("score"),
            }

    portfolio["basket"]  = list(basket.values())
    portfolio["status"]  = "invested" if basket else "empty"
    portfolio["last_synced"] = date.today().isoformat()
    return portfolio
