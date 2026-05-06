#!/usr/bin/env python3
"""
oi_walls.py — Ad-hoc NSE option-chain OI walls printer (via Kite Connect).

NSE's public option-chain API blocks all known scraping methods (Akamai TLS
fingerprinting + JS challenge). This script uses the existing dashboard's
Kite session to pull option contracts, current OI/LTP via kite.quote(), and
prev-day close OI for the top walls via kite.historical_data(..., oi=True).

Requires: an active Kite session (logged in via the dashboard at least once;
the access token is persisted in data_store/kite_session_*.json).

Usage:
  python3 oi_walls.py POWERINDIA
  python3 oi_walls.py POWERINDIA --expiry 28-May-2026
  python3 oi_walls.py POWERINDIA RELIANCE TCS

Output: spot/ATM/expiry header, Max Pain, PCR, top 5 call/put OI walls, and
(when the short-call-writing-into-rally setup fires) a SIGNAL block with
bear put spread suggestions.
"""

import argparse
import csv
import datetime as dt
import os
import sys
import time
from datetime import datetime, timedelta


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CACHE_DIR = os.path.join(HERE, "data", "cache")

# Load .env so KITE_USER*_API_KEY etc. are available when running standalone
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(HERE, ".env"))
except Exception:
    pass


# ── Kite session + instruments ────────────────────────────────────────────────

def _connect_kite():
    """Find an active Kite session from the configured users.
    is_connected() only checks the token *exists*, not whether it's still
    valid — so probe each user with a cheap kite.profile() call and pick
    the first one that actually responds."""
    from broker.kite_broker import KiteBroker
    from config.settings import get_kite_users

    users = get_kite_users()
    if not users:
        raise RuntimeError(
            "No Kite users configured. Set KITE_USER1_API_KEY etc. in .env.")

    failures = []
    for u in users:
        broker = KiteBroker(
            user_id=u["id"], name=u["name"],
            api_key=u["api_key"], api_secret=u["api_secret"])
        if not broker.is_connected() or broker._kite is None:
            failures.append(f"{u['name']}: no session file")
            continue
        try:
            broker._kite.profile()  # cheap auth check
            return broker._kite, u["name"]
        except Exception as e:
            failures.append(f"{u['name']}: {str(e).splitlines()[0]}")
            continue

    raise RuntimeError(
        "No active Kite session. Open the dashboard and re-login on one of "
        "the Kite users, then re-run.\n  Tried: " + " | ".join(failures))


def _load_nfo_instruments(kite):
    """Load NFO instruments. Cached to data/cache/nfo_instruments_YYYY-MM-DD.csv
    so we hit Kite once per day."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    today = dt.date.today().isoformat()
    cache_file = os.path.join(CACHE_DIR, f"nfo_instruments_{today}.csv")

    if os.path.exists(cache_file):
        rows = []
        with open(cache_file, newline="") as f:
            for row in csv.DictReader(f):
                row["instrument_token"] = int(row["instrument_token"])
                row["strike"] = float(row["strike"]) if row["strike"] else 0.0
                row["lot_size"] = int(row["lot_size"]) if row["lot_size"] else 0
                row["expiry"] = (dt.date.fromisoformat(row["expiry"])
                                 if row["expiry"] else None)
                rows.append(row)
        return rows

    instruments = kite.instruments("NFO")
    keys = ["instrument_token", "tradingsymbol", "name", "expiry", "strike",
            "lot_size", "instrument_type", "segment"]
    with open(cache_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for inst in instruments:
            row = {k: inst.get(k, "") for k in keys}
            if isinstance(row["expiry"], dt.date):
                row["expiry"] = row["expiry"].isoformat()
            w.writerow(row)
    return instruments


def _resolve_expiry(matches, expiry_arg):
    """Pick the chosen expiry as a datetime.date. expiry_arg can be
    'DD-Mon-YYYY' or 'YYYY-MM-DD' or None (= nearest)."""
    all_exp = sorted({m["expiry"] for m in matches if m["expiry"]})
    if not all_exp:
        return None, []
    if expiry_arg is None:
        return all_exp[0], all_exp
    target = None
    for fmt in ("%d-%b-%Y", "%Y-%m-%d"):
        try:
            target = datetime.strptime(expiry_arg, fmt).date()
            break
        except ValueError:
            continue
    if target is None or target not in all_exp:
        return None, all_exp
    return target, all_exp


# ── Chain assembly via kite.quote() + historical_data() ───────────────────────

def _empty_leg():
    return {"oi": 0, "oi_chg": 0, "vol": 0, "ltp": 0, "iv": 0,
            "_token": None, "_tsym": None}


def _fill_oi_change(kite, strikes, side, top_n=10):
    """For top-N strikes by current OI on this side, fetch prev-day close OI
    via historical_data and set oi_chg = current_oi - prev_close_oi.
    Rate-limit: ~3/sec. ~3-5 sec total per side."""
    cands = sorted(
        ((K, leg[side]) for K, leg in strikes.items() if leg[side]["oi"] > 0),
        key=lambda kv: -kv[1]["oi"])[:top_n]

    end = datetime.now()
    start = end - timedelta(days=10)

    for K, leg in cands:
        token = leg.get("_token")
        if not token:
            continue
        try:
            candles = kite.historical_data(token, start, end, "day", oi=True)
            if len(candles) >= 2:
                prev_oi = candles[-2].get("oi", 0) or 0
                strikes[K][side]["oi_chg"] = leg["oi"] - prev_oi
        except Exception:
            pass
        time.sleep(0.34)


def fetch_chain_kite(symbol, expiry_arg):
    """Return (expiry_str, spot, strikes_dict, all_expiries_str, lot_size).
    On miss, returns (None, None, None, None, None)."""
    kite, _ = _connect_kite()
    instruments = _load_nfo_instruments(kite)

    matches = [i for i in instruments
               if i.get("name") == symbol
               and i.get("instrument_type") in ("CE", "PE")]
    if not matches:
        return None, None, None, None, None

    chosen, all_exp = _resolve_expiry(matches, expiry_arg)
    if chosen is None:
        avail = ", ".join(e.strftime("%d-%b-%Y") for e in all_exp[:5])
        print(f"Expiry not available. Available: {avail}", file=sys.stderr)
        return None, None, None, None, None

    matches = [m for m in matches if m["expiry"] == chosen]
    lot_size = matches[0]["lot_size"]

    # Spot from NSE LTP
    try:
        ltp_resp = kite.ltp([f"NSE:{symbol}"])
        spot = float(ltp_resp.get(f"NSE:{symbol}", {}).get("last_price", 0))
    except Exception:
        spot = 0.0

    # quote() in batches of 500 trading symbols
    quote_keys = [f"NFO:{m['tradingsymbol']}" for m in matches]
    quotes = {}
    for i in range(0, len(quote_keys), 500):
        batch = quote_keys[i:i + 500]
        try:
            quotes.update(kite.quote(batch))
        except Exception as e:
            msg = str(e)
            if "access_token" in msg.lower() or "api_key" in msg.lower():
                raise RuntimeError(
                    "Kite session expired. Re-login on the dashboard "
                    "(/api/login on the configured Kite user) and re-run.")
            print(f"  WARN: kite.quote() failed on batch: {e}", file=sys.stderr)

    strikes = {}
    for m in matches:
        K = float(m["strike"])
        side = "ce" if m["instrument_type"] == "CE" else "pe"
        if K not in strikes:
            strikes[K] = {"ce": _empty_leg(), "pe": _empty_leg()}
        q = quotes.get(f"NFO:{m['tradingsymbol']}", {}) or {}
        strikes[K][side] = {
            "oi":     q.get("oi", 0) or 0,
            "oi_chg": 0,
            "vol":    q.get("volume", 0) or 0,
            "ltp":    q.get("last_price", 0) or 0,
            "iv":     0,
            "_token": m["instrument_token"],
            "_tsym":  m["tradingsymbol"],
        }

    _fill_oi_change(kite, strikes, "ce", top_n=10)
    _fill_oi_change(kite, strikes, "pe", top_n=10)

    return chosen.strftime("%d-%b-%Y"), spot, strikes, [
        e.strftime("%d-%b-%Y") for e in all_exp], lot_size


def get_prev_close(kite, symbol):
    """Stock prev close from Kite historical_data."""
    try:
        # Need the equity instrument token; cache via kite.ltp() trick won't
        # give it. Use a cheap fallback: kite.quote() includes ohlc.
        q = kite.quote([f"NSE:{symbol}"])
        ohlc = q.get(f"NSE:{symbol}", {}).get("ohlc", {}) or {}
        prev = ohlc.get("close", 0) or 0
        return float(prev) if prev else None
    except Exception:
        return None


# ── Analytics ─────────────────────────────────────────────────────────────────

def compute_max_pain(strikes):
    if not strikes:
        return 0
    ks = sorted(strikes.keys())
    best_K, best_loss = ks[0], float("inf")
    for S in ks:
        loss = 0.0
        for K, leg in strikes.items():
            loss += leg["ce"]["oi"] * max(S - K, 0)
            loss += leg["pe"]["oi"] * max(K - S, 0)
        if loss < best_loss:
            best_loss, best_K = loss, S
    return best_K


def days_to_expiry(expiry_str):
    try:
        d = datetime.strptime(expiry_str, "%d-%b-%Y").date()
        return (d - dt.date.today()).days
    except Exception:
        return -1


def _is_fresh_buildup(side_data):
    chg = side_data["oi_chg"]
    prior = side_data["oi"] - chg
    return chg > max(0.25 * prior, 1000)


# ── Output ────────────────────────────────────────────────────────────────────

def print_walls(strikes, spot, side, label):
    rows = sorted(strikes.items(), key=lambda kv: -kv[1][side]["oi"])[:5]
    print(f"\n{label}")
    print(f"  {'Strike':<8} {'OI':>10}  {'Δ today':>10}  {'Distance':>10}")
    for i, (K, leg) in enumerate(rows):
        L = leg[side]
        oi = L["oi"]
        chg = L["oi_chg"]
        dist = (K - spot) / spot * 100 if spot else 0
        markers = []
        if i == 0:
            markers.append("★ strongest")
        if _is_fresh_buildup(L) and abs(dist) < 5:
            markers.append("⚠ fresh buildup")
        marker_str = ("  " + " + ".join(markers)) if markers else ""
        chg_str = f"{chg:>+10,}" if chg else f"{'(no Δ)':>10}"
        print(f"  {int(K):<8} {oi:>10,}  {chg_str}  {dist:>+9.2f}%{marker_str}")


def detect_signal(strikes, spot, spot_chg_pct):
    if spot_chg_pct is None or spot_chg_pct < 0.5:
        return []
    top3 = {K for K, _ in sorted(
        strikes.items(), key=lambda kv: -kv[1]["ce"]["oi"])[:3]}
    fires = []
    for K, leg in strikes.items():
        if K not in top3:
            continue
        if not spot or abs(K - spot) / spot >= 0.03:
            continue
        ce = leg["ce"]
        prior_oi = ce["oi"] - ce["oi_chg"]
        threshold = max(0.25 * prior_oi, 1000)
        if ce["oi_chg"] > threshold:
            fires.append((K, prior_oi, ce["oi_chg"]))
    return fires


def suggest_spreads(strikes, spot):
    ks = sorted(strikes.keys())
    below = [K for K in ks if K <= spot]
    if not below:
        return []
    long_K = max(below)
    long_idx = ks.index(long_K)
    out = []
    for offset, label in [(1, "TIGHT  (1-strike)"), (2, "WIDER  (2-strike)")]:
        if long_idx - offset < 0:
            continue
        short_K = ks[long_idx - offset]
        long_ltp = strikes[long_K]["pe"]["ltp"]
        short_ltp = strikes[short_K]["pe"]["ltp"]
        if long_ltp <= 0 or short_ltp <= 0:
            continue
        out.append((label, long_K, short_K, long_ltp, short_ltp))
    return out


def print_signal(fires, strikes, spot, spot_chg_pct, lot_size):
    print("\n" + "─" * 64)
    for K, prior_oi, oi_chg in fires:
        dist_pct = (K - spot) / spot * 100
        print(f"SIGNAL: short-call-writing-into-rally @ {int(K)}")
        print(f"  Stock up {spot_chg_pct:+.2f}% AND fresh call OI Δ {oi_chg:+,} "
              f"(was {prior_oi:,}) at {int(K)},")
        print(f"  which is also a top-3 call wall and only {dist_pct:+.2f}% from spot.")
        print(f"  Working hypothesis: writers fading the rally; near-term downside skew.")

    spreads = suggest_spreads(strikes, spot)
    if not spreads:
        print("\n  (no liquid bear put spread suggestions available)")
        print("─" * 64)
        return

    print(f"\n  HEDGE — bear put spread suggestions  (lot {lot_size}):")
    for label, long_K, short_K, long_ltp, short_ltp in spreads:
        debit = long_ltp - short_ltp
        if debit <= 0:
            continue
        width = long_K - short_K
        max_profit = width - debit
        breakeven = long_K - debit
        rr = max_profit / debit
        print(f"\n  {label}  →  {int(long_K)}P / {int(short_K)}P")
        print(f"    Long  {int(long_K)} PE @ ₹{long_ltp:.2f}    "
              f"Short {int(short_K)} PE @ ₹{short_ltp:.2f}")
        print(f"    Net debit  ₹{debit:.2f} / sh   = ₹{debit*lot_size:,.0f} / lot")
        print(f"    Max profit ₹{max_profit:.2f} / sh   = ₹{max_profit*lot_size:,.0f} / lot")
        print(f"    Breakeven  ₹{breakeven:,.0f}    R:R {rr:.2f}×")
    print("─" * 64)


# ── Top-level ─────────────────────────────────────────────────────────────────

def analyze(symbol, expiry_arg, kite_for_prev_close=None, summary_mode=False):
    """Run analysis on one symbol. Returns a metrics dict for summary mode.
    Also prints the detailed per-stock block unless summary_mode=True."""
    metrics = {"ticker": symbol, "ok": False, "error": None}

    if not summary_mode:
        print(f"\n{'═' * 64}")
        print(f" {symbol}  Option Chain Analysis")
        print(f"{'═' * 64}")

    try:
        expiry, spot, strikes, _, lot_size = fetch_chain_kite(symbol, expiry_arg)
    except Exception as e:
        metrics["error"] = str(e).splitlines()[0]
        if not summary_mode:
            print(f"  ERROR fetching {symbol}: {e}", file=sys.stderr)
        return metrics

    if not strikes:
        metrics["error"] = "not in F&O"
        if not summary_mode:
            print(f"  {symbol} not in F&O segment (or no data for chosen expiry).",
                  file=sys.stderr)
        return metrics

    if kite_for_prev_close is None:
        kite_for_prev_close, _ = _connect_kite()
    prev_close = get_prev_close(kite_for_prev_close, symbol)
    spot_chg_pct = ((spot - prev_close) / prev_close * 100) if prev_close else None

    ks = sorted(strikes.keys())
    atm = min(ks, key=lambda K: abs(K - spot))
    dte = days_to_expiry(expiry)

    mp = compute_max_pain(strikes)
    mp_dist = (mp - spot) / spot * 100 if spot else 0

    total_ce_oi  = sum(s["ce"]["oi"]  for s in strikes.values())
    total_pe_oi  = sum(s["pe"]["oi"]  for s in strikes.values())
    total_ce_vol = sum(s["ce"]["vol"] for s in strikes.values())
    total_pe_vol = sum(s["pe"]["vol"] for s in strikes.values())
    pcr_oi  = total_pe_oi  / total_ce_oi  if total_ce_oi  else 0
    pcr_vol = total_pe_vol / total_ce_vol if total_ce_vol else 0

    fires = detect_signal(strikes, spot, spot_chg_pct)
    top_call_K = max(strikes.items(), key=lambda kv: kv[1]["ce"]["oi"])[0]
    top_put_K  = max(strikes.items(), key=lambda kv: kv[1]["pe"]["oi"])[0]

    metrics.update({
        "ok": True,
        "spot": spot,
        "spot_chg_pct": spot_chg_pct,
        "expiry": expiry,
        "dte": dte,
        "lot_size": lot_size,
        "max_pain": mp,
        "max_pain_dist_pct": mp_dist,
        "pcr_oi": pcr_oi,
        "pcr_vol": pcr_vol,
        "top_call_K": top_call_K,
        "top_put_K": top_put_K,
        "signal_strike": fires[0][0] if fires else None,
    })

    if summary_mode:
        return metrics

    chg_str = f"  ({spot_chg_pct:+.2f}% today)" if spot_chg_pct is not None else ""
    print(f"\nExpiry {expiry}")
    print(f"Spot ₹{spot:,.2f}{chg_str}    ATM {int(atm)}    "
          f"Days to expiry {dte}    Lot {lot_size}")

    print(f"\nMax Pain  : ₹{int(mp):,}   ({mp_dist:+.2f}% from spot)")
    print(f"PCR (OI)  : {pcr_oi:.2f}")
    print(f"PCR (Vol) : {pcr_vol:.2f}")

    print_walls(strikes, spot, "ce", "CALL OI WALLS (resistance)")
    print_walls(strikes, spot, "pe", "PUT OI WALLS (support)")

    if fires:
        print_signal(fires, strikes, spot, spot_chg_pct, lot_size)

    bias = "downward" if mp_dist < 0 else "upward"
    print("\nInference")
    print(f"  Range bracket : {int(top_put_K):,} (put wall) ↔ "
          f"{int(top_call_K):,} (call wall)")
    print(f"  Pinning bias  : Max Pain at ₹{int(mp):,} → "
          f"mild {bias} pull into expiry")
    return metrics


def print_summary_table(results):
    """One row per symbol, fixed-width columns."""
    print(f"\n{'═' * 90}")
    print(f" Portfolio OI-Walls Summary  ({len(results)} symbols)")
    print(f"{'═' * 90}\n")

    hdr = (f"{'Ticker':<14} {'Spot':>10} {'Δ%':>6} {'MaxPain':>10} "
           f"{'PCR':>5} {'TopCall':>9} {'TopPut':>9}  Signal")
    print(hdr)
    print("─" * len(hdr))

    # Failures last; OK rows sorted by ticker
    ok_rows = sorted([r for r in results if r["ok"]],
                     key=lambda r: r["ticker"])
    err_rows = sorted([r for r in results if not r["ok"]],
                      key=lambda r: r["ticker"])

    for r in ok_rows:
        spot = r["spot"]
        chg = r["spot_chg_pct"]
        chg_str = f"{chg:+.1f}" if chg is not None else "—"
        sig = (f"SHORT-CALL@{int(r['signal_strike'])}"
               if r["signal_strike"] else "—")
        print(f"{r['ticker']:<14} {spot:>10,.0f} {chg_str:>6} "
              f"{int(r['max_pain']):>10,} {r['pcr_oi']:>5.2f} "
              f"{int(r['top_call_K']):>9,} {int(r['top_put_K']):>9,}  {sig}")

    if err_rows:
        print()
        for r in err_rows:
            print(f"{r['ticker']:<14}  ({r.get('error', 'unknown error')})")

    # Quick portfolio-level read
    n_signals = sum(1 for r in ok_rows if r["signal_strike"])
    n_below_mp = sum(1 for r in ok_rows
                     if r["max_pain_dist_pct"] < -1.0)
    n_above_mp = sum(1 for r in ok_rows
                     if r["max_pain_dist_pct"] >  1.0)
    print(f"\nPortfolio read:")
    print(f"  {n_signals} signal(s) firing across {len(ok_rows)} F&O names")
    print(f"  {n_below_mp} stocks with Max Pain >1% below spot (downward gravity)")
    print(f"  {n_above_mp} stocks with Max Pain >1% above spot (upward gravity)")


def main():
    p = argparse.ArgumentParser(description="NSE option-chain OI walls (via Kite)")
    p.add_argument("symbols", nargs="+",
                   help="NSE symbols (e.g. POWERINDIA RELIANCE)")
    p.add_argument("--expiry", default=None,
                   help="Expiry (DD-Mon-YYYY or YYYY-MM-DD); defaults to nearest")
    p.add_argument("--summary", action="store_true",
                   help="Print one-row-per-stock summary table only "
                        "(no per-stock detail / signal / spread blocks)")
    args = p.parse_args()

    try:
        kite, user_name = _connect_kite()
        print(f"\n[Kite session: {user_name}]", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Brief progress in summary mode so the user knows it's working
    if args.summary:
        print(f"\nProcessing {len(args.symbols)} symbols...", file=sys.stderr)

    results = []
    for i, sym in enumerate(args.symbols):
        if i > 0:
            time.sleep(0.5)
        sym = sym.upper().strip()
        if args.summary:
            print(f"  [{i+1}/{len(args.symbols)}] {sym}", file=sys.stderr)
        r = analyze(sym, args.expiry, kite_for_prev_close=kite,
                    summary_mode=args.summary)
        results.append(r)

    if args.summary:
        print_summary_table(results)
    print()


if __name__ == "__main__":
    main()
