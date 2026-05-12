#!/usr/bin/env python3
"""
AI Universe Backtest — Monthly Rebalance
=========================================
Universe : 94 AI-ecosystem stocks across 16 sectors (max 2 per sector in portfolio)
Selection: Top 10 by momentum score, monthly rebalance
Buffer   : entry rank ≤ 7, exit rank > 20
Scoring  : MR_12 (50%) + MR_3 (50%), Z-scored, Normalised Score
Charges  : US SEC fee 0.00278% on sell, zero commission
Benchmark: QQQ (Nasdaq-100)

Sectors:
 1  Compute Silicon / AI Accelerators   7  Server OEMs & Contract Mfg     13 Defense & Drones
 2  Memory & Storage                    8  AI Data Centers / Neoclouds     14 Space & Satellites
 3  Semiconductor Equipment             9  Power & Cooling                 15 Materials & Rare Earths
 4  Packaging & Foundry                10  Energy / AI Power               16 Frontier AI Models
 5  Photonics / Optical                11  Power Electronics
 6  Networking & Connectivity          12  Robotics & Autonomy

Usage:
    python3 ai_universe_backtest.py
    python3 ai_universe_backtest.py --refresh
    python3 ai_universe_backtest.py --no-regime
    python3 ai_universe_backtest.py --start 2024-05-01
"""

import os, pickle, warnings, argparse
from datetime import date, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
START_DATE     = date(2025, 5, 1)
MAX_SLOTS      = 10
BUFFER_IN      = 7
BUFFER_OUT     = 20
MAX_PER_SECTOR = 2
W12, W3        = 0.50, 0.50
RET12M_CAP     = None   # hard-exclude new entries above this (e.g. 5.0 = 500%); None = off
HALF_SLOT_CAP  = None   # half-slot new entries above this (e.g. 3.0 = 300%); None = off
WARN_THRESHOLD = 3.0    # ⚠ marker on Ret12m in entry table (always-on at 300%)
DECAY_FILTER      = False  # skip entries where Ret12m > 300% AND Ret3m < 20%
PARABOLIC_FILTER  = False  # skip entries where Ret3m/Ret12m > 0.5 (blowoff — any Ret12m level)
LONG_PD        = 252    # 12-month lookback
SHORT_PD       = 63     # 3-month lookback
WARMUP_DAYS    = 520    # calendar days before START_DATE for scoring warmup
CAPITAL        = 100_000.0

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, 'data', 'cache', 'ai_universe_daily.pkl')

# ── FULL AI UNIVERSE — 16 SECTORS ─────────────────────────────────────────────
AI_UNIVERSE = {

    # ── 1. Compute Silicon / AI Accelerators ──────────────────────────────────
    "NVDA":  "Compute Silicon",    # Nvidia — GPUs, AI accelerators
    "AMD":   "Compute Silicon",    # Advanced Micro Devices — GPUs, CPUs
    "AVGO":  "Compute Silicon",    # Broadcom — AI ASICs, networking chips
    "INTC":  "Compute Silicon",    # Intel — CPUs, Gaudi AI accelerator
    "ARM":   "Compute Silicon",    # Arm Holdings — IP architecture licensor
    "MRVL":  "Compute Silicon",    # Marvell Technology — custom AI silicon, DPUs
    "SNPS":  "Compute Silicon",    # Synopsys — EDA tools (chip design software)

    # ── 2. Memory & Storage ───────────────────────────────────────────────────
    "MU":    "Memory & Storage",   # Micron — HBM3E, DRAM, NAND
    "WDC":   "Memory & Storage",   # Western Digital
    "SNDK":  "Memory & Storage",   # SanDisk (WDC flash spinoff, Feb 2025)
    "STX":   "Memory & Storage",   # Seagate — enterprise HDDs
    "NTAP":  "Memory & Storage",   # NetApp — storage software

    # ── 3. Semiconductor Equipment ────────────────────────────────────────────
    "ASML":  "Semicon Equipment",  # ASML — EUV/DUV lithography
    "LRCX":  "Semicon Equipment",  # Lam Research — etch & deposition
    "KLAC":  "Semicon Equipment",  # KLA Corporation — process control & inspection
    "KEYS":  "Semicon Equipment",  # Keysight — test & measurement
    "CAMT":  "Semicon Equipment",  # Camtek — advanced packaging inspection

    # ── 4. Packaging & Foundry ────────────────────────────────────────────────
    "TSM":   "Packaging & Foundry", # TSMC — leading-edge foundry, CoWoS packaging
    "ASX":   "Packaging & Foundry", # ASE Technology — OSAT
    "AMKR":  "Packaging & Foundry", # Amkor Technology — OSAT

    # ── 5. Photonics / Optical ────────────────────────────────────────────────
    "COHR":  "Photonics / Optical", # Coherent Corp — optical transceivers, lasers
    "LITE":  "Photonics / Optical", # Lumentum — optical components
    "GLW":   "Photonics / Optical", # Corning — fiber optic cable
    "FN":    "Photonics / Optical", # Fabrinet — optical module contract mfg
    "NOK":   "Photonics / Optical", # Nokia — optical networking
    "CIEN":  "Photonics / Optical", # Ciena — optical networking
    "AAOI":  "Photonics / Optical", # Applied Optoelectronics — transceivers

    # ── 6. Networking & Connectivity ─────────────────────────────────────────
    "ANET":  "Networking",         # Arista Networks — data center switching
    "CSCO":  "Networking",         # Cisco Systems
    "CRDO":  "Networking",         # Credo Technology — high-speed SerDes
    "APH":   "Networking",         # Amphenol — connectors & interconnects

    # ── 7. Server OEMs & Contract Manufacturing ───────────────────────────────
    "SMCI":  "Server OEMs",        # Super Micro Computer — GPU servers
    "DELL":  "Server OEMs",        # Dell Technologies
    "HPE":   "Server OEMs",        # Hewlett Packard Enterprise
    "JBL":   "Server OEMs",        # Jabil — EMS / contract manufacturing
    "FLEX":  "Server OEMs",        # Flex Ltd — contract electronics

    # ── 8. AI Data Centers / Neoclouds ───────────────────────────────────────
    "CRWV":  "AI Neoclouds",       # CoreWeave — dedicated AI cloud
    "NBIS":  "AI Neoclouds",       # Nebius — AI cloud, ex-Yandex
    "IREN":  "AI Neoclouds",       # Iris Energy — AI compute data centers
    "APLD":  "AI Neoclouds",       # Applied Digital — AI data center hosting
    "WULF":  "AI Neoclouds",       # TeraWulf — AI compute
    "CORZ":  "AI Neoclouds",       # Core Scientific — AI HPC data centers
    "CIFR":  "AI Neoclouds",       # Cipher Mining — AI compute

    # ── 9. Power & Cooling Infrastructure ────────────────────────────────────
    "VRT":   "Power & Cooling",    # Vertiv — power/thermal mgmt for data centers
    "ETN":   "Power & Cooling",    # Eaton — power management, UPS
    "GEV":   "Power & Cooling",    # GE Vernova — power generation, grid equipment
    "PWR":   "Power & Cooling",    # Quanta Services — electrical infrastructure
    "HUBB":  "Power & Cooling",    # Hubbell — electrical products
    "MOD":   "Power & Cooling",    # Modine Manufacturing — thermal management

    # ── 10. Energy / AI Power Supply ─────────────────────────────────────────
    "CEG":   "Energy / AI Power",  # Constellation Energy — nuclear
    "VST":   "Energy / AI Power",  # Vistra — gas + nuclear
    "NEE":   "Energy / AI Power",  # NextEra Energy — wind + solar
    "SMR":   "Energy / AI Power",  # NuScale Power — small modular reactors
    "OKLO":  "Energy / AI Power",  # Oklo — microreactors
    "EOSE":  "Energy / AI Power",  # Eos Energy — grid-scale battery storage
    "EQT":   "Energy / AI Power",  # EQT Corporation — natural gas

    # ── 11. Power Electronics ─────────────────────────────────────────────────
    "STM":   "Power Electronics",  # STMicroelectronics
    "ADI":   "Power Electronics",  # Analog Devices
    "MPWR":  "Power Electronics",  # Monolithic Power Systems
    "NVTS":  "Power Electronics",  # Navitas Semiconductor — GaN/SiC
    "ON":    "Power Electronics",  # ON Semiconductor

    # ── 12. Robotics & Autonomy ───────────────────────────────────────────────
    "TSLA":  "Robotics & Autonomy", # Tesla — autonomous vehicles, Optimus
    "PATH":  "Robotics & Autonomy", # UiPath — RPA, enterprise automation
    "SYM":   "Robotics & Autonomy", # Symbotic — warehouse robotics
    "SERV":  "Robotics & Autonomy", # Serve Robotics — autonomous delivery
    "TER":   "Robotics & Autonomy", # Teradyne — industrial robotics + test
    "ISRG":  "Robotics & Autonomy", # Intuitive Surgical — surgical robotics

    # ── 13. Defense & Drones ──────────────────────────────────────────────────
    "KTOS":  "Defense & Drones",   # Kratos Defense — drones, defense AI
    "AVAV":  "Defense & Drones",   # AeroVironment — tactical drones
    "ONDS":  "Defense & Drones",   # Ondas Holdings — industrial/defense drones
    "RCAT":  "Defense & Drones",   # Red Cat Holdings — defense drones
    "OSIS":  "Defense & Drones",   # OSI Systems — security/defense
    "LMT":   "Defense & Drones",   # Lockheed Martin
    "NOC":   "Defense & Drones",   # Northrop Grumman

    # ── 14. Space & Satellites ────────────────────────────────────────────────
    "ASTS":  "Space & Satellites", # AST SpaceMobile — space-based cellular
    "RKLB":  "Space & Satellites", # Rocket Lab — launch vehicles
    "LUNR":  "Space & Satellites", # Intuitive Machines — lunar landers
    "PL":    "Space & Satellites", # Planet Labs — earth observation
    "BKSY":  "Space & Satellites", # BlackSky — satellite imagery/analytics
    "IRDM":  "Space & Satellites", # Iridium Communications — satellite IoT

    # ── 15. Materials & Rare Earths ───────────────────────────────────────────
    "MP":    "Materials",          # MP Materials — rare earth mining
    "UUUU":  "Materials",          # Energy Fuels — uranium + rare earths
    "FCX":   "Materials",          # Freeport-McMoRan — copper
    "AA":    "Materials",          # Alcoa — aluminum
    "TECK":  "Materials",          # Teck Resources — base metals

    # ── 16. Frontier AI Models ────────────────────────────────────────────────
    "MSFT":  "Frontier AI Models", # Microsoft — Azure + OpenAI partnership
    "GOOGL": "Frontier AI Models", # Alphabet — Gemini / DeepMind
    "META":  "Frontier AI Models", # Meta — LLaMA, AI research
    "AMZN":  "Frontier AI Models", # Amazon — AWS Bedrock + AGI lab
}

SECTOR_ORDER = [
    "Compute Silicon", "Memory & Storage", "Semicon Equipment",
    "Packaging & Foundry", "Photonics / Optical", "Networking",
    "Server OEMs", "AI Neoclouds", "Power & Cooling",
    "Energy / AI Power", "Power Electronics", "Robotics & Autonomy",
    "Defense & Drones", "Space & Satellites", "Materials",
    "Frontier AI Models",
]

# ── CHARGES ───────────────────────────────────────────────────────────────────
def calc_charges(buy_val, sell_val):
    """US zero-commission broker: SEC fee 0.00278% on sell side only."""
    return 0.0000278 * sell_val

# ── FORMATTING ────────────────────────────────────────────────────────────────
def usd(v):
    return f"${v:,.2f}"

def pct(v):
    sign = '+' if v >= 0 else ''
    return f"{sign}{v:.1f}%"

def print_table(headers, rows, col_widths):
    sep = "  ".join("─" * w for w in col_widths)
    hdr = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(f"  {hdr}")
    print(f"  {sep}")
    for row in rows:
        print("  " + "  ".join(str(c).ljust(w) for c, w in zip(row, col_widths)))

# ── DATA LOADING ──────────────────────────────────────────────────────────────
def fetch_ticker(ticker, start, end):
    try:
        df = yf.Ticker(ticker).history(start=start, end=end)
    except Exception:
        df = pd.DataFrame()
    if not df.empty:
        df.index = df.index.tz_localize(None) if df.index.tzinfo else df.index
    return df

def load_or_fetch_data(tickers, fetch_start, fetch_end, refresh=False):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    if not refresh and os.path.exists(CACHE_FILE):
        print(f"Loading cached data from {CACHE_FILE}...")
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    print(f"Fetching {len(tickers)} tickers from yfinance...")
    stock_data = {}
    failed = []
    for i, ticker in enumerate(sorted(tickers)):
        df = fetch_ticker(ticker, fetch_start, fetch_end)
        if not df.empty and len(df) >= 30:
            stock_data[ticker] = df
        else:
            failed.append(ticker)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(tickers)} fetched — {len(stock_data)} OK")
    print(f"  Done. {len(stock_data)}/{len(tickers)} loaded.")
    if failed:
        print(f"  No data: {', '.join(failed)}")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(stock_data, f)
    return stock_data

# ── SCORING ───────────────────────────────────────────────────────────────────
def compute_scores(day, stock_data, date_to_iloc):
    raw = {}
    for ticker in AI_UNIVERSE:
        if ticker not in stock_data:
            continue
        df = stock_data[ticker]
        idx_map = date_to_iloc.get(ticker, {})
        ci = idx_map.get(day)
        if ci is None:
            for off in range(1, 6):
                prev = day - timedelta(days=off)
                if prev in idx_map:
                    ci = idx_map[prev]
                    break
        if ci is None or ci < LONG_PD + 20:
            continue
        closes = df["Close"].values.astype(float)
        p_now = closes[ci]
        p_12m = closes[ci - LONG_PD]
        p_3m  = closes[ci - SHORT_PD]
        if p_now <= 0 or p_12m <= 0 or p_3m <= 0:
            continue
        ret_12 = p_now / p_12m - 1
        ret_3  = p_now / p_3m  - 1
        log_r  = np.diff(np.log(np.maximum(closes[ci - LONG_PD:ci + 1], 0.01)))
        sigma  = float(np.std(log_r)) * np.sqrt(252)
        if sigma < 0.01:
            continue
        raw[ticker] = {
            "price":   p_now,
            "mr_12":   ret_12 / sigma,
            "mr_3":    ret_3  / sigma,
            "ret_12m": ret_12,
            "ret_3m":  ret_3,
            "sigma":   sigma,
        }

    if len(raw) < 10:
        return {}

    mr12v = np.array([s["mr_12"] for s in raw.values()])
    mr3v  = np.array([s["mr_3"]  for s in raw.values()])
    mu12, sd12 = np.mean(mr12v), np.std(mr12v)
    mu3,  sd3  = np.mean(mr3v),  np.std(mr3v)
    if sd12 < 0.001 or sd3 < 0.001:
        return {}

    for t, s in raw.items():
        z12 = (s["mr_12"] - mu12) / sd12
        z3  = (s["mr_3"]  - mu3)  / sd3
        wz  = W12 * z12 + W3 * z3
        s["wt_z"]       = wz
        s["norm_score"] = (1 + wz) if wz >= 0 else 1.0 / (1 - wz)

    return raw

# ── REBALANCE DATES ───────────────────────────────────────────────────────────
def get_rebal_dates(trading_days):
    """First trading day of each calendar month from START_DATE."""
    seen = set()
    result = []
    for d in trading_days:
        ym = (d.year, d.month)
        if ym not in seen and d >= START_DATE:
            seen.add(ym)
            result.append(d)
    return result

# ── PORTFOLIO SELECTION WITH SECTOR CONSTRAINT ───────────────────────────────
def select_portfolio(ranked, ticker_rank, current_set, scores):
    """
    Buffer rule + hard sector cap (MAX_PER_SECTOR) + optional Ret12m cap:
     1. Keeps  — held stocks with rank ≤ BUFFER_OUT, top-ranked first per sector
     2. Entries — new stocks with rank ≤ BUFFER_IN if sector has room
                  (RET12M_CAP excludes new entries with excessive 12m run; keeps are unaffected)
     3. Fill   — best remaining stocks until MAX_SLOTS reached
    Returns (new_set, skipped_cap) where skipped_cap = [(ticker, ret12m), ...].
    """
    sector_count = {}

    def can_add(t):
        return sector_count.get(AI_UNIVERSE.get(t, "?"), 0) < MAX_PER_SECTOR

    def mark(t):
        s = AI_UNIVERSE.get(t, "?")
        sector_count[s] = sector_count.get(s, 0) + 1

    # Step 1: keeps (existing holdings within buffer_out — never capped)
    keeps = set()
    for t in sorted(current_set, key=lambda x: ticker_rank.get(x, 9999)):
        if ticker_rank.get(t, 9999) <= BUFFER_OUT and can_add(t):
            keeps.add(t)
            mark(t)

    new_set = set(keeps)

    def _is_filtered(t):
        """Return (filter_name, ret12, ret3) if entry should be skipped, else None."""
        ret12 = scores[t].get("ret_12m", 0)
        ret3  = scores[t].get("ret_3m", 0)
        if RET12M_CAP is not None and ret12 > RET12M_CAP:
            return ("cap", ret12, ret3)
        if DECAY_FILTER and ret12 > 3.0 and ret3 < 0.20:
            return ("decay", ret12, ret3)
        if PARABOLIC_FILTER and ret12 > 3.0 and ret3 > 0 and (ret3 / ret12) > 0.5:
            return ("parabolic", ret12, ret3)
        return None

    # Step 2: new entries within buffer_in (apply filters to new entries only)
    skipped_cap       = []
    skipped_decay     = []
    skipped_parabolic = []
    for r, (t, _) in enumerate(ranked):
        if r + 1 > BUFFER_IN:
            break
        if t in current_set or t in new_set or not can_add(t):
            continue
        hit = _is_filtered(t)
        if hit:
            name, r12, r3 = hit
            if name == "cap":        skipped_cap.append((t, r12))
            elif name == "decay":    skipped_decay.append((t, r12, r3))
            elif name == "parabolic": skipped_parabolic.append((t, r12, r3))
            continue
        new_set.add(t)
        mark(t)

    # Step 3: fill remaining slots (filters still apply; safety net for under-full portfolios)
    for _, (t, _) in enumerate(ranked):
        if len(new_set) >= MAX_SLOTS:
            break
        if t in new_set or not can_add(t):
            continue
        if _is_filtered(t):
            continue
        new_set.add(t)
        mark(t)

    return new_set, skipped_cap, skipped_decay, skipped_parabolic

# ── MAIN BACKTEST ─────────────────────────────────────────────────────────────
def run(refresh=False, use_regime=True, start_override=None,
        ret12m_cap=None, half_slot_cap=None, decay_filter=False, parabolic_filter=False):
    global START_DATE, RET12M_CAP, HALF_SLOT_CAP, DECAY_FILTER, PARABOLIC_FILTER
    if start_override:
        START_DATE = date.fromisoformat(start_override)
    RET12M_CAP       = ret12m_cap
    HALF_SLOT_CAP    = half_slot_cap
    DECAY_FILTER     = decay_filter
    PARABOLIC_FILTER = parabolic_filter

    n_sectors = len(SECTOR_ORDER)
    regime_label = "Regime ON [QQQ SMA200]" if use_regime else "Regime OFF"
    cap_label = ""
    if RET12M_CAP is not None and HALF_SLOT_CAP is not None:
        cap_label = f" | Cap: >{RET12M_CAP*100:.0f}%=skip >{HALF_SLOT_CAP*100:.0f}%=½slot"
    elif RET12M_CAP is not None:
        cap_label = f" | Ret12m cap: >{RET12M_CAP*100:.0f}%=skip"
    elif HALF_SLOT_CAP is not None:
        cap_label = f" | Half-slot: >{HALF_SLOT_CAP*100:.0f}%"
    if DECAY_FILTER:
        cap_label += " | Decay filter: 12m>300%+3m<20%=skip"
    if PARABOLIC_FILTER:
        cap_label += " | Parabolic filter: 3m/12m>0.5=skip"
    print(f"=== AI Universe | Top{MAX_SLOTS} | Monthly | Max {MAX_PER_SECTOR}/Sector | {regime_label}{cap_label} ===")
    print(f"    top_n={MAX_SLOTS}  buffer_in={BUFFER_IN}  buffer_out={BUFFER_OUT}  "
          f"max_per_sector={MAX_PER_SECTOR}")
    print(f"    universe={len(AI_UNIVERSE)} stocks across {n_sectors} sectors")

    tickers     = list(AI_UNIVERSE.keys())
    fetch_start = START_DATE - timedelta(days=WARMUP_DAYS)
    fetch_end   = date.today()

    stock_data = load_or_fetch_data(tickers, fetch_start, fetch_end, refresh)

    # QQQ for benchmark + regime
    import time as _time
    print("Fetching QQQ (benchmark + regime)...")
    qqq_raw = pd.DataFrame()
    for _attempt in range(3):
        try:
            qqq_raw = yf.Ticker("QQQ").history(start=fetch_start, end=fetch_end)
            if not qqq_raw.empty:
                break
        except Exception as e:
            print(f"  Attempt {_attempt+1} failed: {e}")
        _time.sleep(2)
    if qqq_raw.empty:
        print("  WARNING: Could not fetch QQQ — regime filter disabled")
        use_regime = False
    else:
        qqq_raw.index = qqq_raw.index.tz_localize(None) if qqq_raw.index.tzinfo else qqq_raw.index
        qqq_raw["sma200"] = qqq_raw["Close"].rolling(200).mean()
        print(f"  QQQ: {len(qqq_raw)} bars")
    qqq_iloc = ({qqq_raw.index[i].date(): i for i in range(len(qqq_raw))}
                if not qqq_raw.empty else {})

    # date → iloc maps
    date_to_iloc = {t: {df.index[i].date(): i for i in range(len(df))}
                    for t, df in stock_data.items()}

    # US trading days (≥ 5 of our stocks have data)
    day_counts = {}
    for df in stock_data.values():
        for d in df.index:
            dt = d.date()
            day_counts[dt] = day_counts.get(dt, 0) + 1
    trading_days = sorted(d for d, c in day_counts.items()
                          if c >= 5 and d >= START_DATE)
    print(f"  Trading days: {len(trading_days)} ({trading_days[0]} → {trading_days[-1]})")

    rebal_dates = get_rebal_dates(trading_days)
    print(f"  Rebalance dates: {len(rebal_dates)}")

    # ── Portfolio state ───────────────────────────────────────────────────────
    cash          = CAPITAL
    _start_cap    = cash
    portfolio     = {}
    all_trades    = []
    total_charges = 0.0
    rebal_nav     = []

    banner = (f"  AI UNIVERSE BACKTEST  |  ${CAPITAL:,.0f}  |  Top{MAX_SLOTS}  |"
              f"  Monthly  |  Max {MAX_PER_SECTOR}/Sector  |  {n_sectors} sectors  |  {regime_label}")
    print()
    print("=" * len(banner))
    print(banner)
    print("=" * len(banner))

    for rebal_idx, rebal_day in enumerate(rebal_dates):
        # MTM
        port_value = cash
        for t, pos in portfolio.items():
            idx_map = date_to_iloc.get(t, {})
            ci = idx_map.get(rebal_day)
            if ci is None:
                for off in range(1, 6):
                    prev = rebal_day - timedelta(days=off)
                    if prev in idx_map:
                        ci = idx_map[prev]
                        break
            pos["curr_price"] = (float(stock_data[t]["Close"].iloc[ci])
                                 if ci is not None else pos["entry_price"])
            port_value += pos["shares"] * pos["curr_price"]

        rebal_nav.append({"date": rebal_day, "nav": port_value})
        per_slot = port_value / MAX_SLOTS

        print()
        print("=" * 72)
        print(f"  REBALANCE #{rebal_idx+1:02d}  —  {rebal_day.strftime('%d %b %Y')}")
        print(f"  NAV: {usd(port_value)}  |  Slot: {usd(per_slot)}  |  Cash: {usd(cash)}")
        print("=" * 72)

        scores = compute_scores(rebal_day, stock_data, date_to_iloc)
        scored_n = len(scores)
        if not scores:
            print(f"  ⚠ Insufficient scored stocks — skipping rebalance")
            continue
        print(f"  [{scored_n} of {len(AI_UNIVERSE)} stocks scored]")

        ranked      = sorted(scores.items(), key=lambda x: -x[1]["norm_score"])
        ticker_rank = {t: r + 1 for r, (t, _) in enumerate(ranked)}
        current_set = set(portfolio.keys())

        # Regime check
        regime_off = False
        if use_regime and qqq_iloc:
            qqq_ci = qqq_iloc.get(rebal_day)
            if qqq_ci is None:
                for off in range(1, 6):
                    qqq_ci = qqq_iloc.get(rebal_day - timedelta(days=off))
                    if qqq_ci is not None:
                        break
            if qqq_ci is not None:
                qqq_close = float(qqq_raw["Close"].iloc[qqq_ci])
                qqq_sma   = qqq_raw["sma200"].iloc[qqq_ci]
                if not pd.isna(qqq_sma) and qqq_close < float(qqq_sma):
                    regime_off = True
                    print(f"\n  [REGIME OFF] QQQ {qqq_close:,.2f} < SMA200 "
                          f"{float(qqq_sma):,.2f} — holding all, no exits/entries")

        new_set, skipped_cap, skipped_decay, skipped_parabolic = select_portfolio(ranked, ticker_rank, current_set, scores)

        # ── EXITS ─────────────────────────────────────────────────────────────
        to_sell   = current_set - new_set
        exit_rows = []
        for t in sorted(to_sell):
            if regime_off:
                break
            pos       = portfolio[t]
            ep        = pos.get("curr_price", pos["entry_price"])
            gross_pnl = (ep - pos["entry_price"]) * pos["shares"]
            buy_val   = pos["entry_price"] * pos["shares"]
            sell_val  = ep * pos["shares"]
            chg       = calc_charges(buy_val, sell_val)
            net_pnl   = gross_pnl - chg
            pnl_pct   = (ep / pos["entry_price"] - 1) * 100
            hold_days = (rebal_day - pos["entry_date"]).days

            cash += sell_val
            total_charges += chg
            all_trades.append({
                "ticker": t, "entry": pos["entry_date"], "exit": rebal_day,
                "entry_price": pos["entry_price"], "exit_price": ep,
                "shares": pos["shares"], "gross_pnl": gross_pnl,
                "charges": chg, "net_pnl": net_pnl, "hold_days": hold_days,
            })
            exit_rows.append((
                t,
                AI_UNIVERSE.get(t, ""),
                ticker_rank.get(t, "—"),
                pos["entry_date"].strftime("%d-%b-%y"),
                f"{pos['entry_price']:,.2f}",
                f"{ep:,.2f}",
                pos["shares"],
                usd(gross_pnl),
                pct(pnl_pct),
                f"{hold_days}d",
            ))
            del portfolio[t]

        print(f"\n  EXITS ({len(exit_rows)})")
        if exit_rows:
            print_table(
                ["Ticker", "Sector", "Rank", "Entry", "Entry$", "Exit$",
                 "Qty", "Gross P&L", "P&L%", "Hold"],
                sorted(exit_rows,
                       key=lambda r: float(r[8].replace('+', '').replace('%', '')),
                       reverse=True),
                [6, 20, 5, 10, 10, 10, 6, 12, 8, 6],
            )
        else:
            print("    —")

        # ── ENTRIES ────────────────────────────────────────────────────────────
        to_buy     = new_set - current_set
        entry_rows = []
        for t in sorted(to_buy, key=lambda x: ticker_rank.get(x, 9999)):
            if regime_off:
                break
            s      = scores[t]
            ep     = s["price"]
            ret12  = s["ret_12m"]
            half_sized     = HALF_SLOT_CAP is not None and ret12 > HALF_SLOT_CAP
            effective_slot = per_slot / 2 if half_sized else per_slot
            shares = int(effective_slot // ep)
            if shares == 0:
                continue
            cost = ep * shares
            if cost > cash:
                shares = int(cash // ep)
                cost   = ep * shares
            if shares == 0:
                continue
            chg   = calc_charges(cost, 0)
            cash -= cost + chg
            total_charges += chg
            portfolio[t] = {"entry_date": rebal_day, "entry_price": ep, "shares": shares}
            ret12_str = f"{ret12*100:+.1f}%" + (" ⚠" if ret12 > WARN_THRESHOLD else "")
            entry_rows.append((
                t,
                AI_UNIVERSE.get(t, ""),
                ticker_rank[t],
                f"{s['norm_score']:.3f}",
                ret12_str,
                f"{s['ret_3m']*100:+.1f}%",
                f"{ep:,.2f}",
                shares,
                usd(cost),
                "½" if half_sized else "  ",
            ))

        print(f"\n  ENTRIES ({len(entry_rows)})")
        if entry_rows:
            print_table(
                ["Ticker", "Sector", "Rank", "Score", "Ret12m", "Ret3m",
                 "Entry$", "Qty", "Capital", "Cap"],
                entry_rows,
                [6, 20, 5, 7, 10, 8, 10, 6, 12, 3],
            )
        else:
            print("    —")

        if skipped_cap:
            parts = [f"{t}({r*100:+.0f}%)" for t, r in skipped_cap]
            print(f"  [RET12M CAP >{RET12M_CAP*100:.0f}%] skipped {len(skipped_cap)}: "
                  f"{', '.join(parts)}")
        if skipped_decay:
            parts = [f"{t}(12m:{r12*100:+.0f}%,3m:{r3*100:+.0f}%)"
                     for t, r12, r3 in skipped_decay]
            print(f"  [DECAY FILTER 12m>300%+3m<20%] skipped {len(skipped_decay)}: "
                  f"{', '.join(parts)}")
        if skipped_parabolic:
            parts = [f"{t}(12m:{r12*100:+.0f}%,3m:{r3*100:+.0f}%,ratio:{r3/r12:.2f})"
                     for t, r12, r3 in skipped_parabolic]
            print(f"  [PARABOLIC FILTER 3m/12m>0.5] skipped {len(skipped_parabolic)}: "
                  f"{', '.join(parts)}")

        # ── HOLDS ──────────────────────────────────────────────────────────────
        holds     = (current_set if regime_off else (current_set & new_set))
        hold_rows = []
        warn_syms = []
        for t in sorted(holds, key=lambda x: ticker_rank.get(x, 9999)):
            pos    = portfolio[t]
            cp     = pos.get("curr_price", pos["entry_price"])
            unreal = (cp - pos["entry_price"]) * pos["shares"]
            pp     = (cp / pos["entry_price"] - 1) * 100
            sc     = scores.get(t, {}).get("norm_score", 1.0)
            warn   = " ⚠" if sc < 1.0 else ""
            if sc < 1.0:
                warn_syms.append(t)
            hold_rows.append((
                t,
                AI_UNIVERSE.get(t, ""),
                ticker_rank.get(t, "—"),
                pos["entry_date"].strftime("%d-%b-%y"),
                f"{pos['entry_price']:,.2f}",
                f"{cp:,.2f}",
                pos["shares"],
                usd(unreal),
                f"{pct(pp)}{warn}",
            ))

        print(f"\n  HOLDS ({len(hold_rows)})")
        if hold_rows:
            print_table(
                ["Ticker", "Sector", "Rank", "Since", "Entry$", "Now$",
                 "Qty", "Unreal P&L", "P&L%"],
                sorted(hold_rows,
                       key=lambda r: float(
                           r[8].replace('+', '').replace('%', '').replace(' ⚠', '')),
                       reverse=True),
                [6, 20, 5, 10, 10, 10, 6, 12, 10],
            )
            if warn_syms:
                print(f"  ⚠  WAZ < 0 (momentum below mean): {', '.join(warn_syms)}")
        else:
            print("    —")

        # Portfolio summary with sector breakdown
        invested  = sum(pos["shares"] * pos.get("curr_price", pos["entry_price"])
                        for pos in portfolio.values())
        total_now = cash + invested
        sec_count = {}
        for t in portfolio:
            s = AI_UNIVERSE.get(t, "?")
            sec_count[s] = sec_count.get(s, 0) + 1
        sec_summary = " | ".join(
            f"{s[:8]}×{n}" for s, n in sorted(sec_count.items()))
        print(f"\n  AFTER: Invested {usd(invested)} | Cash {usd(cash)} | "
              f"Total {usd(total_now)} | Positions {len(portfolio)}/{MAX_SLOTS}")
        print(f"  Sectors: {sec_summary}")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    open_pnl = 0.0
    for t, pos in portfolio.items():
        idx_map = date_to_iloc.get(t, {})
        last_ci = max(idx_map.values()) if idx_map else None
        lp = (float(stock_data[t]["Close"].iloc[last_ci])
              if last_ci is not None else pos["entry_price"])
        open_pnl += (lp - pos["entry_price"]) * pos["shares"]

    closed_pnl = sum(tr["net_pnl"]    for tr in all_trades)
    winners    = [tr for tr in all_trades if tr["gross_pnl"] > 0]
    losers     = [tr for tr in all_trades if tr["gross_pnl"] <= 0]
    avg_hold   = (sum(tr["hold_days"] for tr in all_trades) / len(all_trades)
                  if all_trades else 0)

    final_value = cash + sum(
        pos["shares"] * (
            float(stock_data[t]["Close"].iloc[max(date_to_iloc[t].values())])
            if t in date_to_iloc else pos["entry_price"]
        )
        for t, pos in portfolio.items()
    )
    total_return = (final_value - _start_cap) / _start_cap * 100
    years = (trading_days[-1] - trading_days[0]).days / 365.25
    cagr  = ((final_value / _start_cap) ** (1 / years) - 1) * 100 if years > 0 else 0

    gross_win  = sum(tr["gross_pnl"] for tr in winners)
    gross_loss = abs(sum(tr["gross_pnl"] for tr in losers))
    pf         = gross_win / gross_loss if gross_loss > 0 else float('inf')
    wr         = len(winners) / len(all_trades) * 100 if all_trades else 0

    print()
    print("=" * 72)
    print("  FINAL SUMMARY")
    print("=" * 72)
    print(f"  Period        : {trading_days[0]} → {trading_days[-1]}  ({years:.1f} years)")
    print(f"  Universe      : {len(AI_UNIVERSE)} stocks | {n_sectors} sectors | "
          f"max {MAX_PER_SECTOR}/sector")
    print(f"  Starting Cap  : {usd(_start_cap)}")
    print(f"  Final Value   : {usd(final_value)}")
    print(f"  Total Return  : {pct(total_return)}")
    print(f"  CAGR          : {pct(cagr)}")
    print()
    print(f"  Closed Trades : {len(all_trades)}  |  Open: {len(portfolio)}")
    print(f"  Win Rate      : {wr:.1f}%  ({len(winners)}W / {len(losers)}L)")
    print(f"  Profit Factor : {pf:.2f}")
    print(f"  Avg hold      : {avg_hold:.0f} days")
    print(f"  Total charges : {usd(total_charges)}")
    print(f"  Closed net P&L: {usd(closed_pnl)}")
    print(f"  Open unreal   : {usd(open_pnl)}")

    # Per-year
    print()
    print("  YEAR-BY-YEAR:")
    year_last_nav = {}
    for row in rebal_nav:
        year_last_nav[row["date"].year] = row["nav"]
    prev_nav  = _start_cap
    neg_years = 0
    for yr in sorted(year_last_nav):
        end_nav = year_last_nav[yr]
        ret_y   = (end_nav / prev_nav - 1) * 100
        if ret_y < 0:
            neg_years += 1
        bar = ("█" * min(int(abs(ret_y)), 40)) if ret_y > 0 else ("░" * min(int(abs(ret_y)), 40))
        print(f"  {yr}  {'+' if ret_y >= 0 else '-'}{abs(ret_y):5.1f}%  {bar}")
        prev_nav = end_nav
    print(f"  Negative years: {neg_years}")

    # Full sector legend
    print()
    print("  SECTOR LEGEND:")
    for i, sector in enumerate(SECTOR_ORDER, 1):
        stocks = [t for t, s in AI_UNIVERSE.items() if s == sector]
        print(f"  {i:2d}. {sector:26s}: {', '.join(stocks)}")

    # Export CSV
    if rebal_nav:
        csv_path = os.path.join(BASE_DIR, 'ai_universe_rebal.csv')
        pd.DataFrame(rebal_nav).to_csv(csv_path, index=False)
        print(f"\n  Rebalance NAV → ai_universe_rebal.csv ({len(rebal_nav)} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Universe Momentum Backtest (94 stocks, 16 sectors)")
    parser.add_argument("--refresh",   action="store_true", help="Re-download price data")
    parser.add_argument("--no-regime", action="store_true", help="Disable QQQ regime filter")
    parser.add_argument("--start", default=None,
                        help="Override start date YYYY-MM-DD (default 2025-05-01)")
    parser.add_argument("--ret12m-cap", type=float, default=None,
                        help="Exclude new entries with Ret12m above this multiple (e.g. 5.0 = 500%%)")
    parser.add_argument("--half-slot-cap", type=float, default=None,
                        help="Half-size new entries with Ret12m above this multiple (e.g. 3.0 = 300%%)")
    parser.add_argument("--decay-filter", action="store_true",
                        help="Skip entries where Ret12m > 300%% AND Ret3m < 20%% (deceleration)")
    parser.add_argument("--parabolic-filter", action="store_true",
                        help="Skip entries where Ret12m > 300%% AND Ret3m/Ret12m > 0.5 (blowoff)")
    args = parser.parse_args()
    run(refresh=args.refresh,
        use_regime=not args.no_regime,
        start_override=args.start,
        ret12m_cap=args.ret12m_cap,
        half_slot_cap=args.half_slot_cap,
        decay_filter=args.decay_filter,
        parabolic_filter=args.parabolic_filter)
