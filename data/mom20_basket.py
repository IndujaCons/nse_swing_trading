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


def generate_basket(user: dict, signals: list, current_portfolio: dict,
                    unfiltered_ranks: dict = None, all_prices: dict = None,
                    sector_map: dict = None,
                    top5_sectors: list = None,
                    etf_prices: dict = None,
                    mom20_overflow: list = None) -> dict:
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
    total_capital  = (user.get("strategies") or {}).get("mom20", {}).get("capital") or 20_00_000
    capital_per_slot = total_capital / N_SLOTS

    # Build rank map from signals (beta-capped filtered ranks — used for display and entries)
    rank_map        = {s["ticker"]: s["rank"]  for s in signals}
    filtered_rank_map = rank_map  # alias — same source, explicit name for exit display
    price_map = {s["ticker"]: s.get("price", 0) for s in signals}
    score_map = {s["ticker"]: s.get("momentum_score", 0) for s in signals}
    # all_prices contains user's freshly-fetched live prices (highest priority)
    # plus scanner prices for the rest of N200. Override signals prices so
    # exits and entries always reflect the most recent ₹ the user fetched.
    if all_prices:
        for ticker, price in all_prices.items():
            if price:
                price_map[ticker] = price

    current_tickers = {item["ticker"] for item in current_portfolio.get("basket", [])}
    current_qty_map = {item["ticker"]: item.get("qty", 0)
                       for item in current_portfolio.get("basket", [])}

    # Entry universe: top-N_SLOTS if fresh (no positions), else buffer_in=15 for new adds
    BUFFER_IN = N_SLOTS if not current_tickers else 15
    entry_universe = {s["ticker"] for s in signals if s["rank"] <= BUFFER_IN}
    hold_set = {s["ticker"] for s in signals if s["rank"] <= BUFFER_OUT}

    exits  = []
    holds  = []
    entries = []

    # Exits use the unfiltered N200 rank (full universe, no beta cap).
    # Strategy spec: "beta cap ≤ 1.2 at entry; no floor" — once held, exit is
    # purely rank-based on the full N200 universe. Using the beta-capped rank
    # inflates held stocks' apparent rank (high-beta names above them are
    # excluded, pushing them up) and masks true exit signals.
    # If absent from unfiltered_ranks the stock left the N200 (reconstitution).
    fallback_ranks = unfiltered_ranks or {}

    # ETF tickers (e.g. MODEFENCE, HEALTHIETF) bought via the Q2 sector
    # top-up land in current_tickers after a tradebook sync but are NOT in
    # the N200 rank maps — without this branch they'd be misclassified as
    # "removed from Nifty 200". Exit semantics for ETFs: hold while the
    # underlying sector is in today's top-5; exit when it drops past 5.
    from data.sector_etf_map import KNOWN_ETF_SYMBOLS, ETF_TO_SECTOR
    top5_list = list(top5_sectors or [])
    top5_set = set(top5_list)

    def _etf_rank_str(sec):
        """ETF#<n> when sector is in top-5; plain 'ETF' otherwise."""
        try:
            return f"ETF#{top5_list.index(sec) + 1}"
        except (ValueError, AttributeError):
            return "ETF"

    for ticker in current_tickers:
        if ticker in KNOWN_ETF_SYMBOLS:
            sec = ETF_TO_SECTOR.get(ticker)
            price = price_map.get(ticker) or (all_prices or {}).get(ticker, 0)
            qty = current_qty_map.get(ticker, 0)
            if sec and sec in top5_set:
                holds.append({
                    "ticker":         ticker,
                    "rank":           _etf_rank_str(sec),
                    "price":          price,
                    "is_etf":         True,
                    "etf_for_sector": sec,
                })
            else:
                exits.append({
                    "ticker":         ticker,
                    "rank":           "ETF",
                    "current_price":  price,
                    "qty":            qty,
                    "reason":         f"sector {sec or '?'} dropped below top-5",
                    "is_etf":         True,
                    "etf_for_sector": sec,
                })
            continue

        # Exit rank source:
        #   β ≤ 1.2 stocks → filtered rank (rank_map): matches backtest methodology.
        #   β > 1.2 stocks (intentional high-beta buys, e.g. MCX, BHEL) → unfiltered
        #     rank (fallback_ranks): filtered universe excludes them, so rank_map would
        #     return None and cause a false "removed from N200" exit.
        #   Dropped from N200 → neither map has them → force exit.
        if ticker in rank_map:
            rank = rank_map[ticker]           # filtered — matches backtest
            display_rank = rank
        elif ticker in fallback_ranks:
            rank = fallback_ranks[ticker]     # unfiltered — high-beta intentional buy
            display_rank = rank
        else:
            rank = None
            display_rank = None

        price = price_map.get(ticker) or (all_prices or {}).get(ticker, 0)
        qty = current_qty_map.get(ticker, 0)

        if rank is None:
            exits.append({
                "ticker":        ticker,
                "rank":          None,
                "display_rank":  display_rank,
                "current_price": price,
                "qty":           qty,
                "reason":        "removed from Nifty 200",
            })
        elif rank > BUFFER_OUT:
            exits.append({
                "ticker":        ticker,
                "rank":          rank,
                "display_rank":  display_rank,
                "current_price": price,
                "qty":           qty,
                "reason":        f"rank {rank} > {BUFFER_OUT}",
            })
        else:
            holds.append({"ticker": ticker, "rank": rank, "price": price})

    # Entries: rank-ordered fill with sector cap of 4.
    #
    # Two flavours:
    #   • Fresh portfolio (no holds): scan signals top-down, fill up to
    #     N_SLOTS picks. May reach past rank-15 if sector caps force
    #     replacements (signals contains top-40 β-capped, plenty of headroom).
    #   • Existing portfolio: only consider stocks ranked ≤ BUFFER_IN (=15)
    #     for new entries. Stocks ranked 16-40 that are already held stay
    #     as holds (per BUFFER_OUT=40 logic above) but are NOT eligible to
    #     be added fresh — that would churn the portfolio against the
    #     hysteresis spec (entry ≤ 15, hold ≤ 40, exit > 40).
    SECTOR_CAP = 4
    is_fresh = not current_tickers
    sector_count = {}
    # Seed sector_count from holds — both stocks (via sector_map) and held
    # ETFs (via etf_for_sector) — so existing portfolio respects the cap.
    for h in holds:
        sec = (h.get("etf_for_sector") if h.get("is_etf")
               else (sector_map or {}).get(h["ticker"]))
        if sec:
            sector_count[sec] = sector_count.get(sec, 0) + 1

    for s in sorted(signals, key=lambda x: x["rank"]):
        # Existing-portfolio gate: stop at BUFFER_IN. Fresh portfolio: stop
        # once we've collected N_SLOTS picks (may scan past rank-15).
        if not is_fresh and s["rank"] > BUFFER_IN:
            break
        if is_fresh and len(entries) >= N_SLOTS:
            break
        ticker = s["ticker"]
        if ticker in current_tickers:
            continue
        sec = (sector_map or {}).get(ticker)
        if sec and sector_count.get(sec, 0) >= SECTOR_CAP:
            continue   # sector saturated — try next ranked stock
        price = price_map.get(ticker, 0)
        qty = int(math.floor(capital_per_slot / price)) if price > 0 else 0
        entries.append({
            "ticker": ticker,
            "rank":   rank_map[ticker],
            "price":  price,
            "qty":    qty,
            "capital_allocated": round(qty * price, 2),
            "score":  round(score_map.get(ticker, 0), 3),
        })
        if sec:
            sector_count[sec] = sector_count.get(sec, 0) + 1

    entries.sort(key=lambda x: x["rank"])

    # ── Sector-ETF top-up (Q2) ────────────────────────────────────────────
    # For each top-5 sector with <4 stocks among entries+holds, append one
    # sector-ETF entry. Mom20 stock picks themselves are unchanged.
    etf_entries = []
    if sector_map and top5_sectors:
        from data.sector_etf_map import SECTOR_TO_ETF
        # Count stocks per sector across both entries (new) and holds (kept).
        # Held ETFs already cover their sector — count them too via
        # `etf_for_sector` (sector_map only knows N200 stocks).
        sector_count = {}
        held_etf_syms = set()
        for item in list(entries) + list(holds):
            if item.get("is_etf") or item.get("is_etf_topup"):
                sec = item.get("etf_for_sector")
                held_etf_syms.add(item["ticker"])
            else:
                sec = sector_map.get(item["ticker"])
            if sec:
                sector_count[sec] = sector_count.get(sec, 0) + 1
        for sec in top5_sectors:
            mapping = SECTOR_TO_ETF.get(sec)
            if not mapping:
                continue
            etf_sym, etf_name = mapping
            # Belt-and-braces: never re-add an ETF the user already holds,
            # regardless of sector count.
            if etf_sym in held_etf_syms:
                continue
            if sector_count.get(sec, 0) >= 4:
                continue
            etf_price = (etf_prices or {}).get(etf_sym, 0)
            if etf_price <= 0:
                continue   # skip if price unavailable
            qty = int(math.floor(capital_per_slot / etf_price))
            if qty <= 0:
                continue
            etf_entries.append({
                "ticker":            etf_sym,
                "rank":              _etf_rank_str(sec),
                "price":             round(etf_price, 2),
                "qty":               qty,
                "capital_allocated": round(qty * etf_price, 2),
                "score":             None,
                "is_etf_topup":      True,
                "etf_for_sector":    sec,
                "etf_name":          etf_name,
            })
    entries.extend(etf_entries)

    # Min capital = need at least 1 share per slot for the most expensive entry
    max_entry_price = max((e["price"] for e in entries if e["price"] > 0), default=0)
    min_capital = round(N_SLOTS * max_entry_price, 2)
    total_to_deploy = sum(e["capital_allocated"] for e in entries)

    return {
        "basket_date":     date.today().isoformat(),
        "total_capital":   total_capital,
        "capital_per_slot": round(capital_per_slot, 2),
        "min_capital":     min_capital,
        "total_to_deploy": round(total_to_deploy, 2),
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


def _basket_item(ticker: str, transaction_type: str, qty: int, weight: int) -> dict:
    """Build a single Zerodha basket JSON item."""
    return {
        "id": 0,
        "instrument": {
            "tradingsymbol": ticker,
            "type": "EQ",
            "symbol": ticker,
            "segment": "NSE",
            "exchange": "NSE",
            "tickSize": 0.05,
            "lotSize": 1,
            "company": ticker,
            "tradable": True,
            "precision": 2,
            "fullName": ticker,
            "niceName": ticker,
            "niceNameHTML": ticker,
            "stockWidget": True,
            "exchangeToken": 0,
            "instrumentToken": 0,
            "isin": "",
            "related": [],
            "underlying": None,
            "auctionNumber": None,
            "isEquity": True,
            "isWeekly": False,
        },
        "weight": weight,
        "params": {
            "transactionType": transaction_type,
            "product": "CNC",
            "orderType": "MARKET",
            "validity": "DAY",
            "validityTTL": 1,
            "quantity": qty,
            "price": 0,
            "triggerPrice": 0,
            "disclosedQuantity": 0,
            "lastPrice": 0,
            "variety": "regular",
            "tags": [],
        },
    }


def to_zerodha_json(basket_data: dict) -> str:
    """
    Convert basket data to Zerodha basket order JSON string.
    Exits first (SELL), then entries (BUY). Market order + CNC.
    """
    items = []
    weight = 0
    for e in basket_data["exits"]:
        if e["qty"] > 0:
            items.append(_basket_item(e["ticker"], "SELL", e["qty"], weight))
            weight += 1
    for e in basket_data["entries"]:
        if e["qty"] > 0:
            items.append(_basket_item(e["ticker"], "BUY", e["qty"], weight))
            weight += 1
    return json.dumps(items, indent=2)


def _detect_positions_csv(fieldnames: list) -> bool:
    """Return True if this looks like a Zerodha holdings/positions export."""
    cols = {f.strip().lower() for f in (fieldnames or [])}
    has_action = any(c in cols for c in ("type", "trade_type", "transaction type"))
    has_instrument = any(c in cols for c in ("instrument", "tradingsymbol", "symbol"))
    has_avg = any(c in cols for c in ("avg.", "avg. price", "average price"))
    return has_instrument and has_avg and not has_action


def parse_trade_book(csv_content: str) -> list:
    """
    Parse Zerodha order book / trade book CSV export, or a holdings/positions
    snapshot.  Returns list of trade dicts:
        {ticker, action, qty, price, trade_date}

    Positions/holdings format (no BUY/SELL column):
        Product, Instrument, Qty., Avg., LTP, P&L, Chg.
    Each row is treated as a BUY at the average price — useful for seeding
    an existing portfolio from a Zerodha holdings snapshot.
    """
    trades = []
    lines = csv_content.splitlines()
    # Skip leading title/blank rows to find the real header
    header_keywords = {"time", "instrument", "tradingsymbol", "type", "qty",
                       "quantity", "avg.", "product"}
    start = 0
    for i, line in enumerate(lines):
        if any(k in line.lower() for k in header_keywords):
            start = i
            break
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    rows = list(reader)
    if not rows:
        return trades

    # ── Detect format ────────────────────────────────────────────────────────
    if _detect_positions_csv(reader.fieldnames):
        # Holdings / positions snapshot — treat every row as a BUY
        today = date.today().isoformat()
        for row in rows:
            ticker = (row.get("Instrument") or row.get("Symbol") or "").strip().upper()
            qty_raw   = row.get("Qty.") or row.get("Quantity") or "0"
            price_raw = row.get("Avg.") or row.get("Avg. price") or row.get("Average Price") or "0"
            qty_str = str(qty_raw).split("/")[0].strip()
            try:
                qty   = int(float(qty_str))
                price = float(str(price_raw).replace(",", ""))
            except (ValueError, TypeError):
                continue
            if not ticker or qty <= 0 or price <= 0:
                continue
            trades.append({
                "ticker":     ticker,
                "action":     "BUY",
                "qty":        qty,
                "price":      round(price, 2),
                "trade_date": today,
            })
        return trades

    # ── Trade book / order book format ──────────────────────────────────────
    for row in rows:
        # Zerodha order book: Instrument, Type, Qty. (e.g. "35/35"), Avg. price, Time, Status
        # Zerodha trade book: tradingsymbol, trade_type, quantity, price, trade_date
        ticker = (row.get("Instrument") or row.get("tradingsymbol") or
                  row.get("Tradingsymbol") or row.get("Symbol") or "").strip().upper()
        action = (row.get("Type") or row.get("trade_type") or
                  row.get("Transaction type") or "").strip().upper()
        qty_raw   = row.get("Qty.") or row.get("quantity") or row.get("Quantity") or "0"
        price_raw = row.get("Avg. price") or row.get("price") or row.get("Price") or "0"
        date_raw  = (row.get("Time") or row.get("trade_date") or
                     row.get("Order Date") or row.get("Date") or "").strip()
        status    = row.get("Status", "").strip().upper()

        # Skip non-complete orders (order book has pending/rejected rows)
        if status and status not in ("COMPLETE", "COMPLETED"):
            continue

        # Qty. in order book is "filled/total" e.g. "35/35" — take filled portion
        qty_str = str(qty_raw).split("/")[0].strip()
        try:
            qty   = int(float(qty_str))
            price = float(str(price_raw).replace(",", ""))
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

    # Process buys — weighted average price if multiple fills.
    # If the ticker is already held (added in a prior rebalance), accumulate
    # qty and compute a weighted average entry price rather than overwriting.
    for (ticker, action), trade_list in grouped.items():
        if action == "BUY":
            new_qty   = sum(t["qty"] for t in trade_list)
            new_price = sum(t["qty"] * t["price"] for t in trade_list) / new_qty
            trade_date = trade_list[0]["trade_date"]
            entry_info = next((e for e in basket_data.get("entries", [])
                               if e["ticker"] == ticker), {})
            if ticker in basket:
                # Already held — add to position with weighted average price
                old_qty   = basket[ticker].get("qty", 0)
                old_price = basket[ticker].get("entry_price", 0)
                total_qty = old_qty + new_qty
                wavg      = ((old_qty * old_price + new_qty * new_price) / total_qty
                             if total_qty > 0 else new_price)
                basket[ticker] = {
                    **basket[ticker],
                    "qty":              total_qty,
                    "entry_price":      round(wavg, 2),
                    "last_added_date":  trade_date,
                    "last_added_price": round(new_price, 2),
                    "last_added_qty":   new_qty,
                }
            else:
                basket[ticker] = {
                    "ticker":        ticker,
                    "qty":           new_qty,
                    "entry_price":   round(new_price, 2),
                    "entry_date":    trade_date,
                    "weight":        round(100 / N_SLOTS, 2),
                    "rank_at_entry": entry_info.get("rank"),
                    "score_at_entry": entry_info.get("score"),
                }

    portfolio["basket"]  = list(basket.values())
    portfolio["status"]  = "invested" if basket else "empty"
    portfolio["last_synced"] = date.today().isoformat()
    return portfolio
