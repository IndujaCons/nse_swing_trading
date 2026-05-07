#!/usr/bin/env python3
"""
build_sector_map_pit.py — Build a PIT (point-in-time) sector mapping.

For each NIFTY 200 reconstitution date in nse_const/nifty200_pit.json,
emit the sector classification of every member stock at that date.

Approach: classify each historical ticker against the *current* NSE
sectoral index constituent CSVs (Bank, IT, Pharma, etc.). The current
CSVs include ALL stocks in each index — typically 10-80 each, ~500 stocks
total — far more than the 200 currently in N200. This means a stock that
was in N200 in 2015 but isn't today (e.g. ABBOTINDIA, APOLLOTYRE,
BANDHANBNK, ZEEL) still gets classified.

Why this is OK as PIT-correct: sector identity is stable over time
(HDFC Bank has always been a private bank; TCS has always been IT). The
NSE sectoral indices have evolved slightly with reconstitutions, but the
*membership* of each stock has stayed in the same sector — because that's
the sector identity of the company, not just the index.

Edge cases handled via:
 - TICKER_ALIASES (data/momentum_backtest.py) — for renames (HDFC→HDFCBANK,
   ZOMATO→ETERNAL, NIITTECH→COFORGE, etc.)
 - HISTORICAL_OVERRIDES (this file) — for stocks no NSE sector index
   currently lists (delisted, merged, banktrupt) where we still want a
   classification

Output: nse_const/nifty200_sector_map_pit.json
Schema: { "<reconstitution_date>": { "<ticker>": {"primary_sector",
                                                   "listed_in_indices",
                                                   "resolved_via_alias?"} }, ... }
"""

import csv
import json
import os
import sys
import time
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

# Same niftyindices CSV URLs as build_sector_map.py
NIFTYINDICES_URLS = {
    "NIFTY BANK":              "ind_niftybanklist.csv",
    "NIFTY PSU BANK":          "ind_niftypsubanklist.csv",
    "NIFTY IT":                "ind_niftyitlist.csv",
    "NIFTY AUTO":              "ind_niftyautolist.csv",
    "NIFTY METAL":             "ind_niftymetallist.csv",
    "NIFTY ENERGY":            "ind_niftyenergylist.csv",
    "NIFTY HEALTHCARE":        "ind_niftyhealthcarelist.csv",
    "NIFTY FMCG":              "ind_niftyfmcglist.csv",
    "NIFTY CONSUMER DURABLES": "ind_niftyconsumerdurableslist.csv",
    "NIFTY REALTY":            "ind_niftyrealtylist.csv",
    "NIFTY MEDIA":             "ind_niftymedialist.csv",
    "NIFTY CONSUMPTION":       "ind_niftyconsumptionlist.csv",
    "NIFTY INFRA":             "ind_niftyinfralist.csv",
    "NIFTY INDIA MFG":         "ind_niftyindiamanufacturing_list.csv",
    "NIFTY INDIA DEFENCE":     "ind_niftyindiadefence_list.csv",
    "NIFTY MNC":               "ind_niftymnclist.csv",
    "NIFTY PSE":               "ind_niftypselist.csv",
}

OIL_GAS_HARDCODED = {"RELIANCE", "ONGC", "IOC", "BPCL", "GAIL", "HINDPETRO",
                     "PETRONET", "OIL", "ATGL", "MGL", "IGL", "GUJGASLTD"}

# Hardcoded fallbacks (mirrors build_sector_map.py)
HARDCODED_FALLBACKS = {
    "NIFTY BANK": {"AUBANK", "AXISBANK", "BANKBARODA", "CANBK", "FEDERALBNK",
                   "HDFCBANK", "ICICIBANK", "IDFCFIRSTB", "INDUSINDBK",
                   "KOTAKBANK", "PNB", "SBIN", "UNIONBANK", "YESBANK"},
    "NIFTY PSU BANK": {"BANKBARODA", "BANKINDIA", "CANBK", "CENTRALBK",
                       "INDIANB", "IOB", "MAHABANK", "PNB", "PSB", "SBIN",
                       "UCOBANK", "UNIONBANK"},
    "NIFTY INDIA DEFENCE": {"AXISCADES", "AEQUS", "APOLLO", "ASTRAMICRO",
                            "BEML", "BDL", "BEL", "BHARATFORG", "COCHINSHIP",
                            "DATAPATTNS", "DYNAMATECH", "GRSE", "HAL",
                            "MTARTECH", "MAZDOCK", "MIDHANI", "PARAS",
                            "SOLARINDS", "ZENTEC"},
}

# Precedence: most specific → most general (mirrors build_sector_map.py)
SECTOR_PRECEDENCE = [
    "NIFTY PVT BANK", "NIFTY PSU BANK", "NIFTY INDIA DEFENCE",
    "NIFTY OIL & GAS", "NIFTY IT", "NIFTY AUTO", "NIFTY METAL",
    "NIFTY ENERGY", "NIFTY HEALTHCARE", "NIFTY CONSUMER DURABLES",
    "NIFTY FMCG", "NIFTY REALTY", "NIFTY MEDIA", "NIFTY CONSUMPTION",
    "NIFTY INFRA", "NIFTY INDIA MFG", "NIFTY FIN SERVICE", "NIFTY BANK",
    "NIFTY MNC", "NIFTY PSE",
]

# Manual primary-sector overrides (mirrors build_sector_map.py + adds known
# historical-only tickers — delisted/merged/no-current-sector-index — that
# we still want classified. ALL of these apply at every PIT date.
MANUAL_OVERRIDES = {
    # From build_sector_map.py:
    "DMART":      "NIFTY FMCG",
    "GROWW":      "NIFTY FIN SERVICE",
    "ICICIAMC":   "NIFTY FIN SERVICE",
    "TATACAP":    "NIFTY FIN SERVICE",
    "KPITTECH":   "NIFTY IT",
    "TATAELXSI":  "NIFTY IT",
    "NAUKRI":     "NIFTY IT",
    "JUBLFOOD":   "NIFTY CONSUMPTION",
    "IDEA":       "NIFTY CONSUMPTION",
    "TATACOMM":   "NIFTY CONSUMPTION",
    "GODFRYPHLP": "NIFTY FMCG",
    "GMRAIRPORT": "NIFTY INFRA",
    "PREMIERENE": "NIFTY ENERGY",
    "SWIGGY":     "NIFTY CONSUMPTION",
    "VMM":        "NIFTY FMCG",
    "LENSKART":   "NIFTY CONSUMPTION",
    "IREDA":      "NIFTY ENERGY",
    "WAAREEENER": "NIFTY ENERGY",
    "NYKAA":      "NIFTY CONSUMPTION",
    # PIT historical-only — known-defunct or moved indices:
    "ABIRLANUVO":  "NIFTY INFRA",       # Aditya Birla Nuvo (demerged into Grasim)
    "ALSTOMT&D":   "NIFTY ENERGY",      # Alstom T&D (now GE Power India)
    "AMTEKAUTO":   "NIFTY AUTO",        # Amtek Auto (bankrupt)
    "BHARTIHEXA":  "NIFTY IT",          # Bharti Hexacom
    "COX&KINGS":   "NIFTY CONSUMPTION", # travel/hospitality (fraud, delisted)
    "DHANI":       "NIFTY FIN SERVICE", # ex-Indiabulls Ventures (delisted)
    "DHFL":        "NIFTY FIN SERVICE", # Dewan Housing (bankrupt)
    "FCONSUMER":   "NIFTY CONSUMPTION", # Future Consumer (bankrupt)
    "FRETAIL":     "NIFTY CONSUMPTION", # Future Retail (bankrupt)
    "GDL":         "NIFTY INFRA",       # Gateway Distriparks (privatized)
    "GSKCONS":     "NIFTY FMCG",        # GSK Consumer (merged HUL)
    "GUJFLUORO":   "NIFTY METAL",       # Gujarat Fluorochem (chemicals)
    "HDIL":        "NIFTY REALTY",      # Housing Development & Infrastructure
    "HEXAWARE":    "NIFTY IT",          # Hexaware (delisted by Carlyle)
    "IBREALEST":   "NIFTY REALTY",      # Indiabulls Real Estate
    "IBULHSGFIN":  "NIFTY FIN SERVICE", # Indiabulls Housing Finance
    "IBVENTURES":  "NIFTY FIN SERVICE", # Indiabulls Ventures (→DHANI)
    "INOXWIND":    "NIFTY ENERGY",      # Inox Wind
    "ISEC":        "NIFTY FIN SERVICE", # ICICI Securities (delisted)
    "JETAIRWAYS":  "NIFTY CONSUMPTION", # Jet Airways (bankrupt)
    "JISLJALEQS":  "NIFTY METAL",       # Jain Irrigation (chemicals)
    "JPASSOCIAT":  "NIFTY INFRA",       # Jaiprakash Associates
    "JPPOWER":     "NIFTY ENERGY",      # Jaiprakash Power
    "JUSTDIAL":    "NIFTY IT",          # local search
    "KSCL":        "NIFTY FMCG",        # Kaveri Seed
    "MMTC":        "NIFTY METAL",       # MMTC (commodity trader)
    "PCJEWELLER":  "NIFTY CONSUMER DURABLES",  # PC Jeweller
    "PIPAVAVDOC":  "NIFTY INFRA",       # Pipavav Defence (acquired)
    "RAYMOND":     "NIFTY CONSUMPTION", # Raymond (apparel)
    "RCOM":        "NIFTY CONSUMPTION", # Reliance Communications (bankrupt)
    "RELCAPITAL":  "NIFTY FIN SERVICE", # Reliance Capital (bankrupt)
    "RELINFRA":    "NIFTY INFRA",       # Reliance Infrastructure
    "RPOWER":      "NIFTY ENERGY",      # Reliance Power
    "SADBHAV":     "NIFTY INFRA",       # Sadbhav Engineering
    "SINTEX":      "NIFTY METAL",       # Sintex Industries (bankrupt)
    "SUNTV":       "NIFTY MEDIA",       # Sun TV
    "TATAMTRDVR":  "NIFTY AUTO",        # Tata Motors DVR (delisted)
    "TV18BRDCST":  "NIFTY MEDIA",       # TV18 Broadcast (merged Network18)
    "UNITECH":     "NIFTY REALTY",      # Unitech (bankrupt)
    "VAKRANGEE":   "NIFTY IT",          # Vakrangee
    "VIDEOIND":    "NIFTY CONSUMER DURABLES",  # Videocon
    "ZEEL":        "NIFTY MEDIA",       # Zee Entertainment
}

PIT_JSON      = os.path.join(HERE, "nifty200_pit.json")
OUTPUT_JSON   = os.path.join(HERE, "nifty200_sector_map_pit.json")


def fetch_constituents(sector: str, fname: str, retries: int = 3) -> set:
    url = f"https://www.niftyindices.com/IndexConstituent/{fname}"
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code != 200 or r.text[:50].startswith("<"):
                last_err = f"HTTP {r.status_code} (URL not a CSV)"
                continue
            rdr = csv.DictReader(r.text.splitlines())
            return {row["Symbol"].strip() for row in rdr if row.get("Symbol")}
        except Exception as e:
            last_err = str(e)
        time.sleep(2 * (attempt + 1))
    print(f"  ERR  {sector:18s} {last_err} (after {retries} retries)", file=sys.stderr)
    return set()


def fin_service_set() -> set:
    """Pull NIFTY FIN SERVICE membership from the curated dict (niftyindices
    doesn't expose this index's CSV publicly)."""
    from sector_mapping import STOCK_SECTOR_MAP
    return {t for t, s in STOCK_SECTOR_MAP.items() if s == "NIFTY FIN SERVICE"}


def load_aliases():
    """Pull TICKER_ALIASES from data/momentum_backtest.py and reduce each
    yfinance symbol back to its bare ticker (strip .NS / .BO suffix)."""
    from data.momentum_backtest import TICKER_ALIASES as RAW
    return {old: yf.split(".")[0] for old, yf in RAW.items()}


def main():
    with open(PIT_JSON) as f:
        pit = json.load(f)
    print(f"PIT reconstitution dates: {len(pit)}  ({min(pit)} → {max(pit)})\n")

    print("Fetching sector constituent CSVs from niftyindices.com:")
    sec_to_syms = {}
    failed = []
    for sec, fname in NIFTYINDICES_URLS.items():
        syms = fetch_constituents(sec, fname)
        if len(syms) == 0 and sec in HARDCODED_FALLBACKS:
            syms = set(HARDCODED_FALLBACKS[sec])
            print(f"  HRD  {sec:20s} {len(syms):3d} stocks (fallback)")
        elif len(syms) == 0:
            failed.append(sec)
            print(f"  FAIL {sec:20s}   0 stocks")
        else:
            print(f"  OK   {sec:20s} {len(syms):3d} stocks")
        sec_to_syms[sec] = syms
        time.sleep(0.5)
    if failed:
        print(f"\n  ABORT: {len(failed)} sector(s) returned empty: {failed}")
        sys.exit(1)

    sec_to_syms["NIFTY PVT BANK"]    = sec_to_syms["NIFTY BANK"] - sec_to_syms["NIFTY PSU BANK"]
    sec_to_syms["NIFTY FIN SERVICE"] = fin_service_set()
    sec_to_syms["NIFTY OIL & GAS"]   = OIL_GAS_HARDCODED
    print(f"  DRV  NIFTY PVT BANK        {len(sec_to_syms['NIFTY PVT BANK'])} stocks")
    print(f"  FBK  NIFTY FIN SERVICE     {len(sec_to_syms['NIFTY FIN SERVICE'])} stocks")
    print(f"  HRD  NIFTY OIL & GAS       {len(sec_to_syms['NIFTY OIL & GAS'])} stocks")

    # Build comprehensive {ticker: (primary, listed)} for ALL stocks across all
    # sector indices — NOT restricted to current N200.
    print(f"\nBuilding comprehensive sector classifier across all "
          f"{sum(len(v) for v in sec_to_syms.values())} index-membership rows...")
    all_tickers = set().union(*sec_to_syms.values())
    print(f"  {len(all_tickers)} unique stocks across all sector indices")

    classifier = {}
    for ticker in all_tickers:
        listed = [sec for sec in SECTOR_PRECEDENCE if ticker in sec_to_syms.get(sec, set())]
        primary = listed[0] if listed else "OTHER"
        if ticker in MANUAL_OVERRIDES:
            primary = MANUAL_OVERRIDES[ticker]
        classifier[ticker] = {
            "primary_sector":    primary,
            "listed_in_indices": ";".join(listed) if listed else "",
        }
    # Manual overrides may include tickers not in any sector index — add them
    for ticker, sec in MANUAL_OVERRIDES.items():
        if ticker not in classifier:
            classifier[ticker] = {"primary_sector": sec, "listed_in_indices": ""}
    print(f"  classifier built: {len(classifier)} stocks total")

    aliases = load_aliases()

    pit_map = {}
    unmapped_total = {}
    for date in sorted(pit.keys()):
        per_date = {}
        for ticker in sorted(pit[date]):
            entry = classifier.get(ticker)
            via_alias = None
            if entry is None and ticker in aliases:
                aliased = aliases[ticker]
                entry = classifier.get(aliased)
                if entry is not None:
                    via_alias = aliased
            if entry is None:
                entry = {"primary_sector": "OTHER", "listed_in_indices": ""}
                unmapped_total.setdefault(ticker, set()).add(date)
            row = {
                "primary_sector":    entry["primary_sector"],
                "listed_in_indices": entry["listed_in_indices"],
            }
            if via_alias:
                row["resolved_via_alias"] = via_alias
            per_date[ticker] = row
        pit_map[date] = per_date

    with open(OUTPUT_JSON, "w") as f:
        json.dump(pit_map, f, indent=2, sort_keys=True)

    n_total = sum(len(v) for v in pit_map.values())
    n_other = sum(1 for d in pit_map.values() for r in d.values()
                  if r["primary_sector"] == "OTHER")
    print(f"\nWrote {OUTPUT_JSON}")
    print(f"  {n_total} ticker-date rows; {n_other} unmapped (OTHER) "
          f"= {n_other/n_total*100:.1f}%")

    if unmapped_total:
        print(f"\n  {len(unmapped_total)} historical-only tickers still need manual review:")
        for ticker, dates in sorted(unmapped_total.items()):
            print(f"    {ticker:18s} {len(dates):2d} PIT dates  "
                  f"({min(dates)} → {max(dates)})")


if __name__ == "__main__":
    main()
