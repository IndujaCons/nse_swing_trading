"""
ETF Core Strategy — Backtest
Period : 2021-01-01 → today
Capital: ₹10L, 5 slots × ₹2L
Entry   : RS63 > 2% (2 consec Fridays, not falling >3% wk/wk) AND price > SMA63 AND rank ≤ 5  (Friday EOD → Monday open)
Exit    : X1 RS63<0 | X2 price<SMA63 | X3 price < 20dHigh×0.85 | X4 price>SMA20×1.20 or price>SMA50×1.30
Re-entry: same entry conditions as fresh entry (no SwingHighB gate)
Buffer  : rank 6-9 hold; rank 10+ hold until exit fires
Benchmark: Nifty 200 (^CNX200)
"""

import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta

# ── Symbol map: ETF name → yfinance ticker ─────────────────────────────────

UNIVERSE = [
    ("JUNIORBEES",  "Nifty Next 50",         "^NSEMDCP50"),
    ("MID150BEES",  "Nifty Midcap 150",      "MID150BEES.NS"),
    ("HDFCSMALL",   "Nifty Smallcap 250",    "HDFCSML250.NS"),
    ("BANKBEES",    "Nifty Bank",            "^NSEBANK"),
    ("PSUBNKBEES",  "Nifty PSU Bank",        "^CNXPSUBANK"),
    ("ITBEES",      "Nifty IT",              "^CNXIT"),
    ("HEALTHIETF",  "Nifty Healthcare",      "HEALTHIETF.NS"),
    ("AUTOBEES",    "Nifty Auto",            "^CNXAUTO"),
    ("METALIETF",   "Nifty Metal",           "^CNXMETAL"),
    ("CONSUMBEES",  "Nifty Consumption",     "CONSUMBEES.NS"),
    ("INFRABEES",   "Nifty Infra",           "^CNXINFRA"),
    ("OILIETF",     "Nifty Oil & Gas",       "OILIETF.NS"),
    ("MOREALTY",    "Nifty Realty",          "^CNXREALTY"),
    ("CPSEETF",     "Nifty CPSE",            "CPSEETF.NS"),
    ("SETFMOMET",   "Nifty200 Mom30",        "MOM30IETF.NS"),
    ("QUAL30IETF",  "Nifty100 Quality30",    "QUAL30IETF.NS"),
    ("KOTAKLOWV",   "Nifty100 LowVol30",     "LOWVOLIETF.NS"),
    ("ALPL30IETF",  "Nifty Alpha LowVol30",  "ALPL30IETF.NS"),
    ("MODEFENCE",   "Nifty India Defence",   "MODEFENCE.NS"),
    ("MOMOMENTUM",  "Nifty500 Mom50",        "MOMOMENTUM.NS"),
    ("MONIFTY500",  "Nifty 500",             "MONIFTY500.NS"),
    ("GOLDBEES",    "Gold",                  "GOLDBEES.NS"),
    ("SILVERBEES",  "Silver",                "SILVERBEES.NS"),
    ("MON100",      "Nasdaq 100",            "^NDX"),
    ("SETFNN50",    "Nifty Next 50 ETF",     "SETFNN50.NS"),
    # MASPTOP50 removed — duplicate US exposure (MON100 already covers it), -₹21K, 36% WR
    ("FRDM",        "Freedom 100 EM",        "FRDM"),
    ("SOXX",        "Semiconductors",        "SOXX"),
    ("BOTZ",        "Robotics / AI",         "BOTZ"),
    ("EWY",         "South Korea",           "EWY"),
    ("EMXC",        "EM ex-China",           "EMXC"),
    ("AVDV",        "Intl SmallCap Value",   "AVDV"),
    ("ILF",         "Latin America 40",      "ILF"),
    ("XLE",         "US Energy",             "XLE"),
    ("GDX",         "Gold Miners",           "GDX"),
    ("XME",         "Metals & Mining",       "XME"),
    ("VGK",         "Europe (broad)",        "VGK"),
    ("XLK",         "US Technology",         "XLK"),
    ("XLP",         "US Cons Staples",       "XLP"),
    ("TLT",         "US LT Treasuries",      "TLT"),
    ("XLV",         "US Healthcare",         "XLV"),
    ("ITA",         "US Aerospace/Defence",  "ITA"),
    # UCITS equivalents (LSE-listed, Irish/Lux domicile — estate-tax friendly)
    ("GDGB",        "Gold Miners UCITS",     "GDGB.L"),   # GDX equiv
    ("VEUR",        "Europe UCITS",          "VEUR.L"),   # VGK equiv
    ("ISF",         "UK FTSE100 UCITS",      "ISF.L"),    # EWU equiv
    ("EMXU",        "EM ex-China UCITS",     "EMXU.L"),   # EMXC equiv
    ("LTAM",        "Latin America UCITS",   "LTAM.L"),   # ILF equiv
    ("IUES",        "US Energy UCITS",       "IUES.L"),   # XLE equiv
    ("COPX",        "Copper Miners UCITS",   "COPX.L"),   # XME approx
    ("WSML",        "World SmCap UCITS",     "WSML.L"),   # AVDV approx
]

BENCHMARK_SYM  = "^CNX200"
START_DATE     = "2016-01-01"
FETCH_FROM     = "2015-03-01"   # extra history for RS63 warm-up
END_DATE       = date.today().isoformat()
CAPITAL        = 200_000        # per slot
N_SLOTS_MIN    = 5              # starting slots; grows as profits compound
MAX_SLOTS      = 10             # hard cap on slots; beyond ₹20L increase capital per slot instead
ENTRY_RANK_MAX = 5              # hard cap — no entries beyond rank 5
ENTRY_RS63_MIN = 0.02           # minimum RS63 for entry (2% threshold)
LIQUIDBEES_PA  = 0.065          # 6.5% p.a. return on idle slots (LIQUIDBEES proxy)
MAX_SLOT_CAPITAL = 300_000      # ₹3L cap per slot; excess compounds in LIQUIDBEES
RS63_DECAY_MAX = 0.03           # allow RS63 to dip up to 3% below prev week before blocking entry
MAX_INTL_SLOTS = 5              # max slots in international ETFs at any time
INTL_ETFS = {
    "XLE", "GDX", "XME", "VGK",
    "AVDV", "ILF", "FRDM", "EMXC",
    "MON100", "SOXX", "BOTZ", "EWY",
    "XLK", "XLP", "TLT", "XLV", "ITA",
    # UCITS (LSE-listed)
    "GDGB", "VEUR", "ISF", "EMXU", "LTAM", "IUES", "COPX", "WSML",
}
UCITS_ETFS = {"GDGB", "VEUR", "ISF", "EMXU", "LTAM", "IUES", "COPX", "WSML"}

# Correlation groups — at most 1 ETF per group held/entered at any time (pick highest-ranked)
CORR_GROUPS = [
    {"GOLDBEES", "GDX"},       # physical gold vs gold miners
]

def corr_group_blocked(sym, active_syms, pending_syms, rank_map):
    """Return True if a higher-ranked corr-group peer is already active or pending."""
    for group in CORR_GROUPS:
        if sym not in group:
            continue
        my_rank = rank_map.get(sym, 9999)
        for peer in group:
            if peer == sym:
                continue
            if peer in active_syms or peer in pending_syms:
                peer_rank = rank_map.get(peer, 9999)
                if peer_rank <= my_rank:  # peer is higher-ranked (or equal) → block this one
                    return True
    return False

def zerodha_charges(buy_amt, sell_amt):
    """Zerodha equity delivery charges (from CLAUDE.md)."""
    T = buy_amt + sell_amt
    stt      = 0.001     * T
    exchange = 0.0000307 * T
    sebi     = 0.000001  * T
    stamp    = 0.00015   * buy_amt
    gst      = 0.18      * (exchange + sebi)
    return stt + exchange + sebi + stamp + gst
MONTHLY_REBALANCE = False       # True = check only on last Friday of each month

# ── Fetch data ──────────────────────────────────────────────────────────────

def fetch_all():
    print("Fetching benchmark...", end=" ", flush=True)
    bench_df = yf.download(BENCHMARK_SYM, start=FETCH_FROM, end=END_DATE,
                           progress=False, auto_adjust=True)
    bench = bench_df["Close"].squeeze().dropna()
    bench.index = pd.to_datetime(bench.index).tz_localize(None)
    print(f"{len(bench)} rows")

    closes = {}
    for sym, name, yf_sym in UNIVERSE:
        df = yf.download(yf_sym, start=FETCH_FROM, end=END_DATE,
                         progress=False, auto_adjust=True)
        if df.empty:
            print(f"  WARN: no data for {sym} ({yf_sym})")
            closes[sym] = pd.Series(dtype=float)
            continue
        s = df["Close"].squeeze().dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        closes[sym] = s
        print(f"  {sym:12s} {yf_sym:30s} {len(s)} rows")
    return closes, bench

# ── Kalman filter ────────────────────────────────────────────────────────────

def kalman_rs_slope(log_rs_series, q1=0.01, q2=0.001, r=1.0):
    """Kalman filter on log(ETF/Benchmark) — returns daily slope estimates.
    State: [level, slope]. Slope = rate of relative outperformance per day.
    Entry threshold equivalent to RS63>2%: slope > log(1.02)/63 ≈ 0.000314
    """
    n = len(log_rs_series)
    slopes = np.zeros(n)

    x = np.array([log_rs_series[0], 0.0])   # [level, slope]
    P = np.eye(2)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])  # level = level+slope; slope = slope
    H = np.array([[1.0, 0.0]])              # observe level only
    Q = np.array([[q1, 0.0], [0.0, q2]])    # process noise
    I2 = np.eye(2)

    for t in range(n):
        z = log_rs_series[t]
        # Predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        # Update
        y = z - (H @ x_pred)[0]
        S = (H @ P_pred @ H.T)[0, 0] + r
        K = (P_pred @ H.T) / S
        x = x_pred + K.flatten() * y
        P = (I2 - K @ H) @ P_pred
        slopes[t] = x[1]

    return slopes

# ── Indicator computation ────────────────────────────────────────────────────

def compute_indicators(closes_dict, bench, kalman_params=None):
    """For each date, compute rs63, sma63, high20 for all ETFs.
    kalman_params: dict with q1, q2, r — if provided, also compute kalman_slope.
    """
    all_dates = sorted(set.union(*[set(s.index) for s in closes_dict.values() if len(s) > 0]))
    all_dates = [d for d in all_dates if d >= pd.Timestamp(FETCH_FROM)]

    result = {}  # date → {sym: {price, rs63, sma63, high20, kalman_slope}}
    for sym, _, _ in UNIVERSE:
        s = closes_dict.get(sym, pd.Series(dtype=float))
        if len(s) < 68:
            continue

        bench_aligned = bench.reindex(s.index, method="ffill")
        raw_rs63 = (s / s.shift(63)) / (bench_aligned / bench_aligned.shift(63)) - 1
        rs63     = raw_rs63.rolling(5, min_periods=3).mean()
        raw_rs21 = (s / s.shift(21)) / (bench_aligned / bench_aligned.shift(21)) - 1
        rs21     = raw_rs21.rolling(5, min_periods=3).mean()
        sma63  = s.rolling(63, min_periods=63).mean()
        sma50  = s.rolling(50, min_periods=50).mean()
        sma20  = s.rolling(20, min_periods=20).mean()
        high20 = s.rolling(20, min_periods=10).max()
        # Prior week's Friday close: shift back 5 trading bars (last Friday's close)
        prev_wk_close = s.shift(5)

        # Kalman slope (optional)
        kalman_slope_series = None
        if kalman_params:
            log_rs = np.log(s.values / bench_aligned.reindex(s.index, method="ffill").values)
            valid = ~np.isnan(log_rs)
            if valid.sum() > 10:
                ks = kalman_rs_slope(log_rs[valid],
                                     q1=kalman_params.get("q1", 0.01),
                                     q2=kalman_params.get("q2", 0.001),
                                     r=kalman_params.get("r", 1.0))
                full_ks = np.full(len(log_rs), np.nan)
                full_ks[valid] = ks
                kalman_slope_series = pd.Series(full_ks, index=s.index)

        for dt in s.index:
            if dt not in result:
                result[dt] = {}
            entry = {
                "price":  float(s[dt])       if not pd.isna(s[dt])      else None,
                "rs63":   float(rs63[dt])    if not pd.isna(rs63[dt])   else None,
                "rs21":   float(rs21[dt])    if not pd.isna(rs21[dt])   else None,
                "sma63":  float(sma63[dt])   if not pd.isna(sma63[dt])  else None,
                "sma50":  float(sma50[dt])   if not pd.isna(sma50[dt])  else None,
                "sma20":  float(sma20[dt])   if not pd.isna(sma20[dt])  else None,
                "high20": float(high20[dt])  if not pd.isna(high20[dt]) else None,
                "prev_wk_close": float(prev_wk_close[dt]) if not pd.isna(prev_wk_close[dt]) else None,
            }
            if kalman_slope_series is not None:
                v = kalman_slope_series[dt]
                entry["kalman_slope"] = float(v) if not pd.isna(v) else None
            result[dt][sym] = entry
    return result

# ── Helpers ──────────────────────────────────────────────────────────────────

def rank_by_rs63(date_data, use_kalman=False):
    """Return list of (sym, signal_val) sorted desc."""
    key = "kalman_slope" if use_kalman else "rs63"
    ranked = [(sym, d[key]) for sym, d in date_data.items()
              if d.get(key) is not None]
    # Non-UCITS always rank above UCITS (secondary sort by RS63 within each group)
    ranked.sort(key=lambda x: (1 if x[0] in UCITS_ETFS else 0, -x[1]))
    return ranked

# ── Simulation ───────────────────────────────────────────────────────────────

def run_backtest(closes_dict, bench, rs63_decay_allowance=RS63_DECAY_MAX,
                 use_kalman=False, q1=0.01, q2=0.001, kalman_threshold=0.0003):
    """Run ETF Core backtest.
    use_kalman=True: replace RS63 ranking/entry/X1 with Kalman slope.
    kalman_threshold: slope > threshold for entry (default 0.0003 ≈ RS63>2%).
    """
    kalman_params = {"q1": q1, "q2": q2, "r": 1.0} if use_kalman else None
    sig_key = "kalman_slope" if use_kalman else "rs63"
    entry_min = kalman_threshold if use_kalman else ENTRY_RS63_MIN

    print(f"\nComputing indicators {'+ Kalman (q1={q1}, q2={q2})' if use_kalman else ''}...", flush=True)
    ind = compute_indicators(closes_dict, bench, kalman_params=kalman_params)

    # Get all trading days >= START_DATE
    all_days = sorted(d for d in ind.keys() if d >= pd.Timestamp(START_DATE))
    if not all_days:
        print("No trading days found!")
        return

    # Portfolio state
    # slot: {sym, entry_price, entry_date, swing_high_b (locked at exit)}
    slots = [None] * N_SLOTS_MIN      # None = LIQUIDBEES/free; list grows as profits compound
    slot_capital = CAPITAL            # ₹2L per slot initially; rises after portfolio ≥ ₹20L
    watchlist = {}                     # sym → {swing_high_b, exit_date, exit_reason}
    x4_cooloff = {}                    # sym → exit_date; blocks re-entry for 6 weeks after X4
    X4_COOLOFF_WEEKS = 6
    pending_exits  = []                # (sym, slot_idx, reason) → execute next open
    pending_entries = []               # [(sym, reason)] → execute next open

    trades = []       # completed trades
    weekly_log = []   # weekly summary entries
    realized_pnl    = 0.0  # running total of closed-trade P&L
    liquidbees_income = 0.0  # cumulative LIQUIDBEES interest on idle slots

    current_week_exits   = []
    current_week_entries = []
    current_week_holdings = []
    last_friday = None
    exited_this_week = set()   # symbols that exited Mon–Fri; cleared at week end
    prev_friday_rs63 = {}      # sym → rs63 from the previous Friday (2-week confirmation)

    def active_symbols():
        return {s["sym"]: i for i, s in enumerate(slots) if s is not None}

    def free_slot_idx():
        for i, s in enumerate(slots):
            if s is None:
                return i
        return None

    # ── Execute pending actions (at open) ──
    def execute_pending(today_data, today):
        nonlocal pending_exits, pending_entries, current_week_exits, current_week_entries, realized_pnl, slot_capital, x4_cooloff

        # Execute exits first
        for sym, slot_idx, reason in pending_exits:
            slot = slots[slot_idx]
            if slot is None or slot["sym"] != sym:
                continue
            exit_price = today_data.get(sym, {}).get("price")
            if exit_price is None:
                # try yesterday's close
                exit_price = slot.get("last_price", slot["entry_price"])

            sc = slot.get("capital", CAPITAL)
            gross_pnl_pct = (exit_price - slot["entry_price"]) / slot["entry_price"] * 100
            gross_pnl_abs = (exit_price - slot["entry_price"]) / slot["entry_price"] * sc
            sell_amt = sc * (exit_price / slot["entry_price"])
            txcost = zerodha_charges(buy_amt=sc, sell_amt=sell_amt)
            pnl_abs = gross_pnl_abs - txcost
            pnl_pct = pnl_abs / sc * 100

            watchlist[sym] = {"exit_date": today, "exit_reason": reason}
            exited_this_week.add(sym)  # block same-week re-entry
            if reason.startswith("X4"):
                x4_cooloff[sym] = today  # 6-week cooloff before re-entry

            realized_pnl += pnl_abs
            trades.append({
                "sym": sym, "entry_date": slot["entry_date"],
                "exit_date": today, "entry_price": slot["entry_price"],
                "exit_price": exit_price, "reason": reason,
                "pnl_pct": pnl_pct, "pnl_abs": pnl_abs,
                "txcost": txcost,
            })
            current_week_exits.append({
                "sym": sym, "reason": reason,
                "entry_price": slot["entry_price"], "exit_price": exit_price,
                "pnl_pct": pnl_pct, "exec_date": today,
            })
            slots[slot_idx] = None

        pending_exits.clear()

        # Execute entries / re-entries
        for sym, reason, signal_rank in pending_entries:
            if sym in active_symbols():
                continue
            fidx = free_slot_idx()
            if fidx is None:
                continue
            entry_price = today_data.get(sym, {}).get("price")
            if entry_price is None:
                continue

            if sym in watchlist:
                del watchlist[sym]

            slots[fidx] = {
                "sym": sym, "entry_price": entry_price,
                "entry_date": today, "last_price": entry_price,
                "entry_rank": signal_rank,
                "capital": slot_capital,   # capital deployed for this position
            }
            current_week_entries.append({
                "sym": sym, "reason": reason, "price": entry_price,
                "exec_date": today, "entry_rank": signal_rank,
            })

        pending_entries.clear()

    # ── Main day loop ──
    for i, today in enumerate(all_days):
        today_data = ind.get(today, {})

        # Execute pending from yesterday
        execute_pending(today_data, today)

        # Update last_price every day (for MTM tracking)
        for slot in slots:
            if slot and slot["sym"] in today_data:
                p = today_data[slot["sym"]].get("price")
                if p:
                    slot["last_price"] = p

        # Rankings updated daily (needed for Friday decisions + holdings display)
        ranked = rank_by_rs63(today_data, use_kalman=use_kalman)
        rank_map = {sym: r+1 for r, (sym, _) in enumerate(ranked)}

        # ── Daily X4 check: overextended exit (any day, next-open execution) ──
        for slot_idx, slot in enumerate(slots):
            if slot is None:
                continue
            if any(slot_idx == si for _, si, _ in pending_exits):
                continue  # already flagged for exit
            sym = slot["sym"]
            d = today_data.get(sym, {})
            price = d.get("price")
            sma20 = d.get("sma20")
            sma50 = d.get("sma50")
            if price is None:
                continue
            reason = None
            if sma20 is not None and price > sma20 * 1.20:
                reason = f"X4: >20% above SMA20 ({price:.1f} vs {sma20:.1f})"
            elif sma50 is not None and price > sma50 * 1.30:
                reason = f"X4: >30% above SMA50 ({price:.1f} vs {sma50:.1f})"
            if reason:
                pending_exits.append((sym, slot_idx, reason))

        # ── Friday: exit + entry checks (EOD signal → Monday open execution) ──
        is_friday = (today.weekday() == 4)
        is_rebalance = is_friday and (
            not MONTHLY_REBALANCE or
            (today + timedelta(7)).month != today.month  # last Friday of month
        )
        if is_rebalance:
            # Compute current unrealized P&L for slot-expansion check
            unrealized_for_expand = 0.0
            for slot in slots:
                if slot is not None:
                    p = today_data.get(slot["sym"], {}).get("price") or slot.get("last_price", slot["entry_price"])
                    if p:
                        unrealized_for_expand += (p / slot["entry_price"] - 1) * slot.get("capital", CAPITAL)

            portfolio_val_check = CAPITAL * N_SLOTS_MIN + realized_pnl + unrealized_for_expand

            if len(slots) < MAX_SLOTS:
                # Phase 1: unlock a new ₹2L slot for each ₹2L of portfolio growth, up to MAX_SLOTS
                target_slots = max(N_SLOTS_MIN, min(MAX_SLOTS, int(portfolio_val_check // CAPITAL)))
                if target_slots > len(slots):
                    added = target_slots - len(slots)
                    slots.extend([None] * added)
                    print(f"  ★ SLOT UNLOCKED [{today.strftime('%Y-%m-%d')}]: "
                          f"{len(slots)-added} → {len(slots)} slots  "
                          f"(portfolio ₹{portfolio_val_check/1e5:.1f}L)")
            elif portfolio_val_check >= 10 * CAPITAL:
                # Phase 2: at MAX_SLOTS, grow capital per slot once portfolio ≥ ₹20L
                # Capped at MAX_SLOT_CAPITAL; excess compounds in LIQUIDBEES
                new_sc = min(portfolio_val_check / MAX_SLOTS, MAX_SLOT_CAPITAL)
                if new_sc > slot_capital * 1.01:   # update only when >1% change
                    print(f"  ↑ SLOT CAPITAL [{today.strftime('%Y-%m-%d')}]: "
                          f"₹{slot_capital/1e5:.2f}L → ₹{new_sc/1e5:.2f}L  "
                          f"(portfolio ₹{portfolio_val_check/1e5:.1f}L)")
                    slot_capital = new_sc

            # Exit checks — checked Friday EOD only, executed Monday open
            for slot_idx, slot in enumerate(slots):
                if slot is None:
                    continue
                sym = slot["sym"]
                d = today_data.get(sym, {})
                price  = d.get("price")
                sig    = d.get(sig_key)   # rs63 or kalman_slope
                sma63  = d.get("sma63")
                high20 = d.get("high20")
                if price is None:
                    continue
                reason = None
                if sig is not None and sig < 0:
                    reason = f"X1: {sig_key}<0"
                elif sma63 is not None and price < sma63:
                    reason = "X2: price<SMA63"
                elif high20 is not None and price < high20 * 0.85:
                    reason = f"X3: -15% from 20dHigh ({high20:.1f}→{price:.1f})"
                # X4 is checked daily (not here) — already in pending_exits if triggered
                if reason:
                    pending_exits.append((sym, slot_idx, reason))

            active = active_symbols()
            # Symbols already flagged for exit this Friday + any that exited earlier this week
            pending_exit_syms = {sym for sym, _, _ in pending_exits}
            blocked_syms = pending_exit_syms | exited_this_week

            # Count active intl slots (excluding pending exits)
            intl_active = sum(1 for s in slots if s and s["sym"] in INTL_ETFS and s["sym"] not in pending_exit_syms)

            # Re-entry checks (watchlist = previously exited, now eligible for re-entry)
            for sym, wl in list(watchlist.items()):
                if sym in active or sym in blocked_syms:
                    continue
                # X4 cooloff: block for 6 weeks after overextension exit
                x4_date = x4_cooloff.get(sym)
                if x4_date and (today - x4_date).days < X4_COOLOFF_WEEKS * 7:
                    continue
                d = today_data.get(sym, {})
                price  = d.get("price")
                sig    = d.get(sig_key)
                sma63  = d.get("sma63")
                rank   = rank_map.get(sym)
                if None in (price, sig, sma63, rank):
                    continue
                if rank > ENTRY_RANK_MAX:
                    continue
                if sig < entry_min or price <= sma63:
                    continue
                prev_rs = prev_friday_rs63.get(sym, 0)
                if prev_rs < entry_min:  # 2nd week confirmation
                    continue
                if rs63_decay_allowance is not None and not use_kalman and sig < prev_rs - rs63_decay_allowance:
                    continue  # decay check (RS63 only; Kalman already smooths)
                prev_wk_c = d.get("prev_wk_close")
                if prev_wk_c is not None and price <= prev_wk_c:
                    continue  # this Friday close must exceed last Friday close (weekly momentum)
                if sym in INTL_ETFS and intl_active >= MAX_INTL_SLOTS:
                    continue  # geography cap: max 3 intl slots
                pending_syms = {s for s, _, _ in pending_entries}
                if corr_group_blocked(sym, active, pending_syms, rank_map):
                    continue  # corr group: higher-ranked peer already active/pending
                pending_entries.append((sym, f"RE-ENTRY rank#{rank}", rank))

            # Fresh entry checks
            re_entry_syms = {sym for sym, _, _ in pending_entries}
            for sym, sig_val in ranked[:ENTRY_RANK_MAX + 1]:
                if sym in active or sym in blocked_syms or sym in re_entry_syms:
                    continue
                # X4 cooloff: block fresh entries too
                x4_date = x4_cooloff.get(sym)
                if x4_date and (today - x4_date).days < X4_COOLOFF_WEEKS * 7:
                    continue
                rank = rank_map.get(sym)
                if rank is None or rank > ENTRY_RANK_MAX:
                    continue  # hard gate: never enter beyond rank 5
                d = today_data.get(sym, {})
                price = d.get("price")
                sma63 = d.get("sma63")
                sig   = d.get(sig_key)
                if None in (price, sma63, sig):
                    continue
                if sig < entry_min or price <= sma63:
                    continue
                prev_rs = prev_friday_rs63.get(sym, 0)
                if prev_rs < entry_min:  # 2nd week confirmation
                    continue
                if rs63_decay_allowance is not None and not use_kalman and sig < prev_rs - rs63_decay_allowance:
                    continue  # decay check (RS63 only; Kalman already smooths)
                prev_wk_c = d.get("prev_wk_close")
                if prev_wk_c is not None and price <= prev_wk_c:
                    continue  # this Friday close must exceed last Friday close (weekly momentum)
                if sym in INTL_ETFS and intl_active >= MAX_INTL_SLOTS:
                    continue  # geography cap: max 3 intl slots
                pending_syms = {s for s, _, _ in pending_entries}
                if corr_group_blocked(sym, active, pending_syms, rank_map):
                    continue  # corr group: higher-ranked peer already active/pending
                if sym not in watchlist:  # watchlist → re-entry only
                    pending_entries.append((sym, f"ENTRY rank#{rank}", rank))

            # Save this Friday's signal for next week's 2-week check
            prev_friday_rs63 = {
                sym: d.get(sig_key, 0) or 0
                for sym, d in today_data.items()
            }

        # ── Weekly summary on Friday ──
        if is_friday:
            active = active_symbols()
            holdings = []
            # Base: initial capital + all realized gains + unrealized MTM on open positions
            unrealized_pnl = 0.0
            for sidx, slot in enumerate(slots):
                if slot is not None:
                    sym = slot["sym"]
                    d = today_data.get(sym, {})
                    price = d.get("price") or slot.get("last_price", slot["entry_price"])
                    sc = slot.get("capital", CAPITAL)
                    mtm = (price / slot["entry_price"] - 1) * sc if price else 0.0
                    unrealized_pnl += mtm
                    pnl_pct = mtm / sc * 100
                    weeks_held = (today - slot["entry_date"]).days // 7
                    holdings.append({
                        "sym": sym,
                        "rank": rank_map.get(sym),
                        "entry_rank": slot.get("entry_rank"),
                        "rs63": d.get("rs63"),
                        "pnl_pct": pnl_pct,
                        "entry_date": slot["entry_date"],
                        "weeks_held": weeks_held,
                    })
            portfolio_val = CAPITAL * N_SLOTS_MIN + realized_pnl + unrealized_pnl

            # LIQUIDBEES income on idle slots + excess above MAX_SLOT_CAPITAL cap
            free_slots   = sum(1 for s in slots if s is None)
            max_deployed = len(slots) * slot_capital
            excess_lb    = max(0.0, portfolio_val - max_deployed)   # sits in LIQUIDBEES when slot cap hit
            idle_capital = free_slots * slot_capital + excess_lb
            weekly_lb    = idle_capital * ((1 + LIQUIDBEES_PA) ** (1/52) - 1)
            liquidbees_income += weekly_lb
            realized_pnl      += weekly_lb   # counts as income in portfolio value

            weekly_log.append({
                "date": today,
                "exits":    list(current_week_exits),
                "entries":  list(current_week_entries),
                "holdings": holdings,
                "portfolio_val": portfolio_val,
                "n_slots": len(slots),
                "free_slots": free_slots,
                "liquidbees_wk": weekly_lb,
            })
            current_week_exits   = []
            current_week_entries = []
            exited_this_week.clear()  # reset for next week

        last_friday = today if is_friday else last_friday

    # ── Close all open positions at last price ──
    last_day = all_days[-1]
    last_data = ind.get(last_day, {})
    for slot_idx, slot in enumerate(slots):
        if slot is None:
            continue
        sym = slot["sym"]
        price = last_data.get(sym, {}).get("price") or slot.get("last_price", slot["entry_price"])
        sc = slot.get("capital", CAPITAL)
        gross_pnl_abs = (price - slot["entry_price"]) / slot["entry_price"] * sc
        sell_amt = sc * (price / slot["entry_price"])
        txcost = zerodha_charges(buy_amt=sc, sell_amt=sell_amt)
        pnl_abs = gross_pnl_abs - txcost
        pnl_pct = pnl_abs / sc * 100
        trades.append({
            "sym": sym, "entry_date": slot["entry_date"],
            "exit_date": last_day, "entry_price": slot["entry_price"],
            "exit_price": price, "reason": "OPEN (MTM)",
            "pnl_pct": pnl_pct, "pnl_abs": pnl_abs, "txcost": txcost,
        })

    return weekly_log, trades, bench, liquidbees_income

# ── Print results ─────────────────────────────────────────────────────────────

def print_results(weekly_log, trades, bench=None, liquidbees_income=0.0):
    print("\n" + "="*80)
    print("  ETF CORE BACKTEST — 2025-01-01 to " + END_DATE)
    print("  5 slots × ₹2L | RS63 vs Nifty200 | X1/X2/X3 exits")
    print("="*80)

    for week in weekly_log:
        dt  = week["date"].strftime("%Y-%m-%d")
        wday = week["date"].strftime("%a")
        exits    = week["exits"]
        entries  = week["entries"]
        holdings = week["holdings"]

        if not exits and not entries and not holdings:
            continue

        print(f"\n{'─'*70}")
        print(f"  WEEK: {dt} ({wday})")

        if exits:
            print(f"\n  EXITS ({len(exits)})")
            for e in exits:
                sign = "+" if e["pnl_pct"] >= 0 else ""
                exec_dt = e["exec_date"].strftime("%b%d") if e.get("exec_date") else ""
                print(f"    ✗ {e['sym']:12s}  {exec_dt}  {sign}{e['pnl_pct']:+.1f}%  "
                      f"entry ₹{e['entry_price']:.1f} → exit ₹{e['exit_price']:.1f}  "
                      f"[{e['reason']}]")

        if entries:
            real_entries = [e for e in entries if not e.get("skipped")]
            skipped     = [e for e in entries if e.get("skipped")]
            if entries:
                print(f"\n  ENTRIES ({len(entries)})")
                for e in entries:
                    exec_dt = e["exec_date"].strftime("%b%d") if e.get("exec_date") else ""
                    print(f"    ✓ {e['sym']:12s}  {exec_dt}  ₹{e['price']:.1f}  [{e['reason']}]")

        n_slots   = week.get("n_slots", N_SLOTS_MIN)
        free_slots = week.get("free_slots", 0)
        lb_wk      = week.get("liquidbees_wk", 0)

        if holdings:
            print(f"\n  HOLDINGS ({len(holdings)})")
            active_sorted = sorted(holdings, key=lambda x: x.get("rank") or 99)
            for h in active_sorted:
                rs = h["rs63"]
                rs_str = f"RS63:{rs*100:+.1f}%" if rs is not None else "RS63:—"
                cur_rank = h["rank"]
                ent_rank = h.get("entry_rank")
                if cur_rank and ent_rank and cur_rank != ent_rank:
                    rank_str = f"#{cur_rank}(in@#{ent_rank})"
                elif cur_rank:
                    rank_str = f"#{cur_rank}"
                else:
                    rank_str = "#?"
                sign = "+" if h["pnl_pct"] >= 0 else ""
                entry_dt = h["entry_date"].strftime("%b%d") if h["entry_date"] else "?"
                weeks_held = h.get("weeks_held", 0)
                print(f"    · {h['sym']:12s}  rank{rank_str:12s}  {rs_str:15s}  "
                      f"MTM:{sign}{h['pnl_pct']:+.1f}%  ({weeks_held}w since {entry_dt})")

        if free_slots > 0:
            print(f"\n  IDLE  ({free_slots}/{n_slots} slots in LIQUIDBEES)  +₹{lb_wk:,.0f} this week  (6.5% p.a.)")

    # ── Trade summary ──
    print(f"\n{'='*70}")
    print(f"  TRADE SUMMARY")
    print(f"{'─'*70}")
    print(f"  {'Symbol':12s}  {'Entry':10s}  {'Exit':10s}  {'Entry₹':>8s}  {'Exit₹':>8s}  {'P&L%':>7s}  {'P&L₹':>9s}  Reason")
    print(f"  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*9}  {'─'*20}")
    closed = [t for t in trades if t["reason"] != "OPEN (MTM)"]
    open_  = [t for t in trades if t["reason"] == "OPEN (MTM)"]

    total_closed = 0
    total_open = 0
    for t in sorted(closed, key=lambda x: x["exit_date"]):
        sign = "+" if t["pnl_pct"] >= 0 else ""
        print(f"  {t['sym']:12s}  {t['entry_date'].strftime('%Y-%m-%d')}  "
              f"{t['exit_date'].strftime('%Y-%m-%d')}  "
              f"₹{t['entry_price']:>7.1f}  ₹{t['exit_price']:>7.1f}  "
              f"{sign}{t['pnl_pct']:>6.1f}%  "
              f"₹{t['pnl_abs']:>+8.0f}  {t['reason']}")
        total_closed += t["pnl_abs"]

    if open_:
        print(f"\n  OPEN POSITIONS (MTM)")
        for t in open_:
            sign = "+" if t["pnl_pct"] >= 0 else ""
            print(f"  {t['sym']:12s}  {t['entry_date'].strftime('%Y-%m-%d')}  "
                  f"{'still open':10s}  "
                  f"₹{t['entry_price']:>7.1f}  ₹{t['exit_price']:>7.1f}  "
                  f"{sign}{t['pnl_pct']:>6.1f}%  "
                  f"₹{t['pnl_abs']:>+8.0f}  {t['reason']}")
            total_open += t["pnl_abs"]

    total_deployed = CAPITAL * N_SLOTS_MIN
    total_pnl = total_closed + total_open
    total_pct = total_pnl / total_deployed * 100
    total_txcost = sum(t.get("txcost", 0) for t in trades)

    wins   = [t for t in closed if t["pnl_pct"] > 0]
    losses = [t for t in closed if t["pnl_pct"] <= 0]
    wr = len(wins) / len(closed) * 100 if closed else 0

    gross_closed = total_closed + total_txcost   # add back costs to get gross
    gross_total  = total_pnl   + total_txcost

    print(f"\n{'─'*70}")
    print(f"  Total trades  : {len(closed)} closed + {len(open_)} open")
    print(f"  Win rate      : {wr:.1f}%  ({len(wins)}W / {len(losses)}L)  [closed trades only]")
    if wins:   print(f"  Avg win       : {np.mean([t['pnl_pct'] for t in wins]):+.1f}%")
    if losses: print(f"  Avg loss      : {np.mean([t['pnl_pct'] for t in losses]):+.1f}%")
    print(f"{'─'*70}")
    print(f"  Gross closed  : ₹{gross_closed:+,.0f}")
    avg_per_trade = total_txcost / len(closed) if closed else 0
    print(f"  Txn costs     : ₹{-total_txcost:+,.0f}  (Zerodha formula, avg ₹{avg_per_trade:,.0f}/trade × {len(closed)} closed trades)")
    print(f"  ─── net ─────────────────────────────────────────────────────────")
    print(f"  Net closed P&L: ₹{total_closed:+,.0f}")
    print(f"  LIQUIDBEES    : ₹{liquidbees_income:+,.0f}  (6.5% p.a. on idle slots)")
    print(f"  Net open  P&L : ₹{total_open:+,.0f}  (MTM, gross)")
    grand_total = total_closed + liquidbees_income + total_open
    grand_pct   = grand_total / total_deployed * 100
    print(f"  Net total P&L : ₹{grand_total:+,.0f}  ({grand_pct:+.1f}% on ₹{total_deployed/1e5:.0f}L deployed)")

    # ── Weekly P&L table ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  WEEKLY P&L  (starts: {N_SLOTS_MIN} slots × ₹2L = ₹{CAPITAL*N_SLOTS_MIN//100000:.0f}L; new slot per ₹2L profit)")
    print(f"{'─'*70}")
    print(f"  {'Week':10s}  {'Port.Val':>10s}  {'Wk P&L':>9s}  {'Wk%':>6s}  {'Cum P&L':>10s}  {'Cum%':>6s}  {'Slots':>5s}  {'LiqBees':>8s}  Activity")
    print(f"  {'─'*10}  {'─'*10}  {'─'*9}  {'─'*6}  {'─'*10}  {'─'*6}  {'─'*5}  {'─'*8}  {'─'*20}")

    initial_val = CAPITAL * N_SLOTS_MIN
    prev_val = initial_val
    for week in weekly_log:
        pval = week.get("portfolio_val")
        if pval is None:
            continue
        wk_pnl = pval - prev_val
        wk_pct = wk_pnl / prev_val * 100
        cum_pnl = pval - initial_val
        cum_pct = cum_pnl / initial_val * 100
        n_active = len(week["holdings"])
        dt = week["date"].strftime("%Y-%m-%d")

        # Activity summary
        acts = []
        for e in week["exits"]:
            acts.append(f"OUT:{e['sym']}({e['pnl_pct']:+.1f}%)")
        for e in week["entries"]:
            acts.append(f"IN:{e['sym']}")
        act_str = " ".join(acts) if acts else "—"

        wk_sign  = "+" if wk_pnl >= 0 else ""
        cum_sign = "+" if cum_pnl >= 0 else ""
        lb_wk    = week.get("liquidbees_wk", 0)
        lb_str   = f"₹{lb_wk:>5,.0f}" if lb_wk > 0 else "      —"
        print(f"  {dt}  ₹{pval:>9,.0f}  {wk_sign}₹{wk_pnl:>7,.0f}  {wk_sign}{wk_pct:>5.1f}%  "
              f"{cum_sign}₹{cum_pnl:>8,.0f}  {cum_sign}{cum_pct:>5.1f}%  {n_active:>2}/{week.get('n_slots', N_SLOTS_MIN)}  {lb_str}  {act_str}")
        prev_val = pval

    # Monthly aggregation
    print(f"\n{'='*70}")
    print(f"  MONTHLY P&L SUMMARY  (vs Nifty 200 TRI benchmark)")
    print(f"{'─'*70}")
    print(f"  {'Month':8s}  {'Start Val':>10s}  {'End Val':>10s}  {'P&L':>9s}  {'%':>6s}  {'Cum%':>7s}  {'N200%':>7s}  {'Alpha':>7s}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*9}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*7}")

    weeks_with_val = [(w["date"], w["portfolio_val"])
                      for w in weekly_log if w.get("portfolio_val") is not None]

    # Build benchmark monthly returns
    bench_monthly = {}
    if bench is not None:
        bench_m = bench[bench.index >= pd.Timestamp(START_DATE)]
        for dt, val in bench_m.items():
            key = dt.strftime("%Y-%m")
            if key not in bench_monthly:
                bench_monthly[key] = {"first": val, "last": val}
            bench_monthly[key]["last"] = val

    if weeks_with_val:
        months = {}
        for dt, pval in weeks_with_val:
            key = dt.strftime("%Y-%m")
            if key not in months:
                months[key] = {"first": pval, "last": pval}
            months[key]["last"] = pval

        for month_key in sorted(months.keys()):
            m = months[month_key]
            prev_keys = [k for k in sorted(months.keys()) if k < month_key]
            prev_month_end = months[prev_keys[-1]]["last"] if prev_keys else initial_val
            month_pnl = m["last"] - prev_month_end
            month_pct = month_pnl / prev_month_end * 100
            cum_pct = (m["last"] - initial_val) / initial_val * 100
            sign = "+" if month_pnl >= 0 else ""

            # Benchmark return for this month
            bm = bench_monthly.get(month_key)
            if bm and bm["first"]:
                bm_pct = (bm["last"] / bm["first"] - 1) * 100
                alpha  = month_pct - bm_pct
                bm_str = f"{bm_pct:>+6.1f}%"
                al_str = f"{alpha:>+6.1f}%"
            else:
                bm_str = "    n/a"
                al_str = "    n/a"

            print(f"  {month_key}  ₹{prev_month_end:>9,.0f}  ₹{m['last']:>9,.0f}  "
                  f"{sign}₹{month_pnl:>7,.0f}  {sign}{month_pct:>5.1f}%  {cum_pct:>+6.1f}%  "
                  f"{bm_str}  {al_str}")

    print()


def print_compact(weekly_log, trades, bench=None):
    """Compact per-week trade log — one row per week, easy to scan."""
    initial_val = CAPITAL * N_SLOTS_MIN

    # Build benchmark cumulative for weekly dates
    bench_by_date = {}
    if bench is not None:
        bench_start = bench[bench.index >= pd.Timestamp(START_DATE)]
        first_val   = bench_start.iloc[0] if len(bench_start) else None
        for dt, val in bench_start.items():
            bench_by_date[dt] = (val / first_val - 1) * 100 if first_val else 0

    print("\n" + "="*120)
    print("  ETF CORE — COMPACT WEEKLY TRADE LOG  (2025-01-01 → " + END_DATE + ")")
    print("="*120)
    hdr = (f"  {'Week':10s}  {'Entries':35s}  {'Exits':45s}  "
           f"{'Holdings':30s}  {'Val':>8s}  {'Cum%':>6s}  {'N200':>6s}")
    print(hdr)
    print("  " + "─"*118)

    prev_val = initial_val
    for week in weekly_log:
        dt  = week["date"].strftime("%Y-%m-%d")
        pval = week.get("portfolio_val", prev_val)
        cum_pct = (pval - initial_val) / initial_val * 100

        # Nifty 200 cumulative
        bench_dt = min(bench_by_date.keys(), key=lambda d: abs((d - week["date"]).days)) if bench_by_date else None
        n200_str = f"{bench_by_date[bench_dt]:+.1f}%" if bench_dt else "  n/a"

        # Entries column
        real_entries = [e for e in week["entries"] if not e.get("skipped")]
        skipped      = [e for e in week["entries"] if e.get("skipped")]
        ent_parts = []
        for e in real_entries:
            rk = e.get("entry_rank", "?")
            is_re = "RE-ENTRY" in e.get("reason", "")
            tag = "↺" if is_re else "+"
            ent_parts.append(f"{tag}{e['sym']}#{rk}@{e['price']:.0f}")
        for e in skipped:
            ent_parts.append(f"⊘{e['sym']}(gap)")
        ent_str = " ".join(ent_parts) if ent_parts else "—"

        # Exits column
        exit_parts = []
        for e in week["exits"]:
            pnl = e["pnl_pct"]
            sign = "+" if pnl >= 0 else ""
            # shorten reason to code
            reason = e["reason"]
            if "X1" in reason:   code = "X1"
            elif "X2" in reason: code = "X2"
            elif "X3" in reason: code = "X3"
            else:                code = reason[:4]
            exec_d = e["exec_date"].strftime("%d") if e.get("exec_date") else ""
            exit_parts.append(f"{e['sym']}({exec_d}){sign}{pnl:.1f}%[{code}]")
        exit_str = " ".join(exit_parts) if exit_parts else "—"

        # Holdings column
        hold_parts = []
        sorted_h = sorted(week["holdings"], key=lambda x: x.get("rank") or 99)
        for h in sorted_h:
            cur_r = h["rank"]
            ent_r = h.get("entry_rank")
            rk_str = f"#{cur_r}" if cur_r else "#?"
            if ent_r and cur_r and ent_r != cur_r:
                rk_str = f"#{cur_r}↓#{ent_r}"
            pnl_s = f"{h['pnl_pct']:+.0f}%"
            hold_parts.append(f"{h['sym']}{rk_str}{pnl_s}")
        hold_str = " ".join(hold_parts) if hold_parts else "—"

        val_str = f"₹{pval/1000:.0f}K"
        cum_str = f"{cum_pct:+.1f}%"

        # Truncate columns to fit
        print(f"  {dt}  {ent_str:<35s}  {exit_str:<45s}  {hold_str:<30s}  {val_str:>8s}  {cum_str:>6s}  {n200_str:>6s}")
        prev_val = pval

    # Monthly P&L summary (compact)
    print("\n" + "="*120)
    print("  MONTHLY  (strategy vs Nifty 200)")
    print(f"  {'Month':8s}  {'P&L':>9s}  {'%':>6s}  {'Cum%':>7s}  {'N200%':>7s}  {'Alpha':>7s}")
    print("  " + "─"*60)
    weeks_with_val = [(w["date"], w["portfolio_val"])
                      for w in weekly_log if w.get("portfolio_val") is not None]
    bench_monthly = {}
    if bench is not None:
        bench_m = bench[bench.index >= pd.Timestamp(START_DATE)]
        for dt2, val in bench_m.items():
            key = dt2.strftime("%Y-%m")
            if key not in bench_monthly:
                bench_monthly[key] = {"first": val, "last": val}
            bench_monthly[key]["last"] = val
    if weeks_with_val:
        months = {}
        for dt2, pval in weeks_with_val:
            key = dt2.strftime("%Y-%m")
            if key not in months:
                months[key] = {"first": pval, "last": pval}
            months[key]["last"] = pval
        for mk in sorted(months.keys()):
            m = months[mk]
            prev_keys = [k for k in sorted(months.keys()) if k < mk]
            prev_end = months[prev_keys[-1]]["last"] if prev_keys else initial_val
            mpnl = m["last"] - prev_end
            mpct = mpnl / prev_end * 100
            cpct = (m["last"] - initial_val) / initial_val * 100
            bm = bench_monthly.get(mk)
            bm_pct = (bm["last"] / bm["first"] - 1) * 100 if bm and bm["first"] else None
            alpha  = mpct - bm_pct if bm_pct is not None else None
            bm_s = f"{bm_pct:>+6.1f}%" if bm_pct is not None else "    n/a"
            al_s = f"{alpha:>+6.1f}%" if alpha is not None else "    n/a"
            sign = "+" if mpnl >= 0 else ""
            print(f"  {mk}  {sign}₹{mpnl:>7,.0f}  {sign}{mpct:>5.1f}%  {cpct:>+6.1f}%  {bm_s}  {al_s}")

    # Trade list
    print("\n" + "="*120)
    print("  CLOSED TRADES")
    print(f"  {'#':3s}  {'Sym':12s}  {'Entry':10s}  {'Exit':10s}  "
          f"{'Entry₹':>8s}  {'Exit₹':>8s}  {'Hold':>5s}  {'P&L%':>7s}  {'P&L₹':>9s}  Code")
    print("  " + "─"*108)
    closed = [t for t in trades if t["reason"] != "OPEN (MTM)"]
    for i, t in enumerate(sorted(closed, key=lambda x: x["entry_date"]), 1):
        hold_days = (t["exit_date"] - t["entry_date"]).days
        sign = "+" if t["pnl_pct"] >= 0 else ""
        reason = t["reason"]
        if "X1" in reason:   code = "X1"
        elif "X2" in reason: code = "X2"
        elif "X3" in reason: code = "X3"
        else:                code = reason[:6]
        print(f"  {i:3d}  {t['sym']:12s}  {t['entry_date'].strftime('%Y-%m-%d')}  "
              f"{t['exit_date'].strftime('%Y-%m-%d')}  "
              f"₹{t['entry_price']:>7.1f}  ₹{t['exit_price']:>7.1f}  "
              f"{hold_days:>4d}d  {sign}{t['pnl_pct']:>6.1f}%  "
              f"₹{t['pnl_abs']:>+8.0f}  {code}")
    open_ = [t for t in trades if t["reason"] == "OPEN (MTM)"]
    if open_:
        print(f"\n  OPEN (MTM as of {END_DATE})")
        for t in open_:
            hold_days = (t["exit_date"] - t["entry_date"]).days
            sign = "+" if t["pnl_pct"] >= 0 else ""
            print(f"       {t['sym']:12s}  {t['entry_date'].strftime('%Y-%m-%d')}  "
                  f"{'open':10s}  "
                  f"₹{t['entry_price']:>7.1f}  ₹{t['exit_price']:>7.1f}  "
                  f"{hold_days:>4d}d  {sign}{t['pnl_pct']:>6.1f}%  ₹{t['pnl_abs']:>+8.0f}")

    # Stats footer
    wins   = [t for t in closed if t["pnl_pct"] > 0]
    losses = [t for t in closed if t["pnl_pct"] <= 0]
    total_pnl = sum(t["pnl_abs"] for t in trades)
    total_txc  = sum(t.get("txcost", 0) for t in trades)
    print(f"\n  {len(closed)} closed trades  |  WR {len(wins)/len(closed)*100:.1f}%  "
          f"({len(wins)}W/{len(losses)}L)  |  AvgW {np.mean([t['pnl_pct'] for t in wins]):+.1f}%  "
          f"AvgL {np.mean([t['pnl_pct'] for t in losses]):+.1f}%  |  "
          f"Total P&L ₹{total_pnl:+,.0f}  ({total_pnl/initial_val*100:+.1f}%)  |  "
          f"Txn costs ₹{total_txc:,.0f}")
    print()


def run_kalman_grid(closes_dict, bench):
    """Grid search over Kalman q1/q2 and entry threshold. Compare vs RS63 baseline."""
    import itertools

    # Baseline
    print("\n" + "="*90)
    print("  KALMAN GRID SEARCH vs RS63 BASELINE")
    print("="*90)

    wlog, trades, _, lb = run_backtest(closes_dict, bench)
    closed = [t for t in trades if t["reason"] != "OPEN (MTM)"]
    wins   = [t for t in closed if t["pnl_pct"] > 0]
    total  = sum(t["pnl_abs"] for t in trades) + lb
    neg_years = _count_neg_years(wlog)
    print(f"\n  {'Config':45s}  {'Trades':>6s}  {'WR':>5s}  {'Net%':>7s}  {'NegYrs':>6s}")
    print(f"  {'─'*45}  {'─'*6}  {'─'*5}  {'─'*7}  {'─'*6}")
    print(f"  {'RS63 BASELINE':45s}  {len(closed):>6d}  {len(wins)/len(closed)*100:>4.1f}%  "
          f"{total/1e6*100:>+6.1f}%  {neg_years:>6d}")

    q1_vals    = [0.001, 0.01, 0.05, 0.1]
    q2_vals    = [0.0001, 0.001, 0.005, 0.01]
    thresholds = [0.0001, 0.0003, 0.0005, 0.001]

    best = {"net": total, "cfg": "RS63 baseline"}
    for q1, q2 in itertools.product(q1_vals, q2_vals):
        for thr in thresholds:
            wlog_k, trades_k, _, lb_k = run_backtest(
                closes_dict, bench,
                use_kalman=True, q1=q1, q2=q2, kalman_threshold=thr
            )
            closed_k = [t for t in trades_k if t["reason"] != "OPEN (MTM)"]
            if not closed_k:
                continue
            wins_k  = [t for t in closed_k if t["pnl_pct"] > 0]
            total_k = sum(t["pnl_abs"] for t in trades_k) + lb_k
            neg_k   = _count_neg_years(wlog_k)
            label   = f"Kalman q1={q1} q2={q2} thr={thr}"
            marker  = " ◄ BEST" if total_k > best["net"] else ""
            print(f"  {label:45s}  {len(closed_k):>6d}  {len(wins_k)/len(closed_k)*100:>4.1f}%  "
                  f"{total_k/1e6*100:>+6.1f}%  {neg_k:>6d}{marker}")
            if total_k > best["net"]:
                best = {"net": total_k, "cfg": label}

    print(f"\n  Best config: {best['cfg']}  ({best['net']/1e6*100:+.1f}%)")


def _count_neg_years(weekly_log):
    """Count calendar years with negative portfolio return."""
    if not weekly_log:
        return 0
    initial = CAPITAL * N_SLOTS_MIN
    year_vals = {}
    for w in weekly_log:
        yr = w["date"].year
        pv = w.get("portfolio_val")
        if pv is None:
            continue
        if yr not in year_vals:
            year_vals[yr] = {"first": pv, "last": pv}
        year_vals[yr]["last"] = pv
    neg = 0
    years = sorted(year_vals.keys())
    for i, yr in enumerate(years):
        prev = year_vals[years[i-1]]["last"] if i > 0 else initial
        if year_vals[yr]["last"] < prev:
            neg += 1
    return neg


if __name__ == "__main__":
    import sys
    closes_dict, bench = fetch_all()
    if "--kalman-grid" in sys.argv:
        run_kalman_grid(closes_dict, bench)
    else:
        weekly_log, trades, bench, liquidbees_income = run_backtest(closes_dict, bench)
        print_results(weekly_log, trades, bench, liquidbees_income)

        # Export month-end NAV for portfolio correlation analysis
        import os as _os
        wl_df = pd.DataFrame(weekly_log)
        if not wl_df.empty and 'portfolio_val' in wl_df.columns:
            wl_df['date'] = pd.to_datetime(wl_df['date'])
            wl_df = wl_df.set_index('date').sort_index()
            monthly_etf = wl_df['portfolio_val'].resample('ME').last().dropna()
            nav_csv = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'etf_monthly.csv')
            monthly_etf.reset_index().rename(columns={'portfolio_val': 'nav'}).to_csv(nav_csv, index=False)
            print(f"\n  Monthly NAV exported → etf_monthly.csv ({len(monthly_etf)} rows)")
