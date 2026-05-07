#!/usr/bin/env python3
"""
build_sector_map.py — Build NIFTY 200 stock → sectoral index mapping.

Reads:
  - nse_const/nifty200_pit.json   (current NIFTY 200 list; uses latest date)
  - 12 niftyindices.com sector CSVs (live download)
  - sector_mapping.STOCK_SECTOR_MAP (fallback for NIFTY FIN SERVICE — its
    CSV isn't exposed publicly at niftyindices.com under any URL we probed)

Writes:
  - nse_const/nifty200_sector_map.csv
    Columns: ticker, primary_sector, listed_in_indices

Primary-sector precedence (most specific → most general; first match wins):
  PVT BANK > PSU BANK > IT > AUTO > METAL > ENERGY > PHARMA > FMCG > REALTY
  > MEDIA > FIN SERVICE > BANK > MNC > PSE
Stocks in none of these → "OTHER" (logged at end).

Re-run when the NIFTY 200 PIT updates (e.g. after a reconstitution snapshot
in nifty200_pit.json) or when index constituents change.

Usage:
    python3 nse_const/build_sector_map.py
"""

import os, sys, csv, json, time
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

# 16 sector CSVs that resolve cleanly at niftyindices.com
NIFTYINDICES_URLS = {
    "NIFTY BANK":              "ind_niftybanklist.csv",
    "NIFTY PSU BANK":          "ind_niftypsubanklist.csv",
    "NIFTY IT":                "ind_niftyitlist.csv",
    "NIFTY AUTO":              "ind_niftyautolist.csv",
    "NIFTY METAL":             "ind_niftymetallist.csv",
    "NIFTY ENERGY":            "ind_niftyenergylist.csv",
    # NIFTY PHARMA dropped — strict subset of HEALTHCARE within N200 (all 13
    # pharma names also in HEALTHCARE; HEALTHCARE adds 3 hospitals).
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

# OIL & GAS: niftyindices.com doesn't expose its CSV under any URL probed;
# hardcoded from the published index composition.
OIL_GAS_HARDCODED = {"RELIANCE", "ONGC", "IOC", "BPCL", "GAIL", "HINDPETRO",
                     "PETRONET", "OIL", "ATGL", "MGL", "IGL", "GUJGASLTD"}

# Hardcoded fallbacks used ONLY when niftyindices.com fetch fails (rate-limit
# or timeout). These constituent lists are very stable (change once every few
# years on reconstitution). Update if NSE adds/removes a bank.
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

# Manual primary-sector overrides — apply AFTER auto-classification.
# Use for N200 stocks with no clean home in the niftyindices universe but
# where we want to assign a sensible primary anyway (typically retail /
# new-age tech / cement / paint / telecom / etc.).
# Format: { ticker: ("PRIMARY SECTOR LABEL", "rationale") }
MANUAL_OVERRIDES = {
    "DMART":   ("NIFTY FMCG",        "Avenue Supermarts — consumer-staples retailer; treated as FMCG-adjacent"),
    "GROWW":   ("NIFTY FIN SERVICE", "Groww (Billionbrains Garage Ventures) — discount brokerage / fin-services platform"),
    "TATACAP":  ("NIFTY FIN SERVICE", "Tata Capital — NBFC; classified as non-bank financial services"),
    "ICICIAMC":  ("NIFTY FIN SERVICE", "ICICI Prudential AMC — asset management"),
    "KPITTECH":  ("NIFTY IT",          "KPIT Technologies — software/IT services"),
    "TATAELXSI": ("NIFTY IT",          "Tata Elxsi — design/embedded software"),
    "NAUKRI":    ("NIFTY IT",          "Info Edge — internet/IT services platform (Naukri.com, 99acres)"),
    "JUBLFOOD":  ("NIFTY CONSUMPTION", "Jubilant FoodWorks — Domino's franchisee; QSR/consumption"),
    "IDEA":      ("NIFTY CONSUMPTION", "Vodafone Idea — telecom; alongside BHARTIARTL/INDUSTOWER"),
    "TATACOMM":  ("NIFTY CONSUMPTION", "Tata Communications — telecom; alongside BHARTIARTL/INDUSTOWER"),
    "GODFRYPHLP": ("NIFTY FMCG",       "Godfrey Phillips — tobacco/FMCG (alongside ITC)"),
}

# Precedence: first match wins → encodes "most specific first"
SECTOR_PRECEDENCE = [
    "NIFTY PVT BANK",          # derived BANK \ PSU BANK
    "NIFTY PSU BANK",
    "NIFTY INDIA DEFENCE",     # very narrow (~19 stocks)
    "NIFTY OIL & GAS",         # narrow oil/gas — beats broader ENERGY
    "NIFTY IT",
    "NIFTY AUTO",
    "NIFTY METAL",
    "NIFTY ENERGY",            # broader than O&G (includes power utilities)
    "NIFTY HEALTHCARE",        # superset of PHARMA + hospitals (PHARMA dropped)
    "NIFTY CONSUMER DURABLES", # specific consumer sub-sector
    "NIFTY FMCG",              # consumer staples
    "NIFTY REALTY",
    "NIFTY MEDIA",
    "NIFTY CONSUMPTION",       # broad consumer-facing — retail, hotels, media, etc. (~30 stocks)
    "NIFTY INFRA",             # infra/airports/ports/telecom/utilities (~30 stocks)
    "NIFTY INDIA MFG",         # broad catch-all manufacturing (~80 stocks)
    "NIFTY FIN SERVICE",       # from sector_mapping.STOCK_SECTOR_MAP fallback
    "NIFTY BANK",              # broader; rarely primary (PVT/PSU cover all)
    "NIFTY MNC",               # factor blend (lowest priority)
    "NIFTY PSE",               # factor blend
]

OUTPUT_CSV = os.path.join(HERE, "nifty200_sector_map.csv")
PIT_JSON   = os.path.join(HERE, "nifty200_pit.json")


def fetch_constituents(sector: str, fname: str, retries: int = 3) -> set:
    """Fetch one sector's constituent symbol set from niftyindices.com.
    Retries on timeout — niftyindices' endpoint rate-limits aggressively."""
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
        time.sleep(2 * (attempt + 1))   # 2s, 4s, 6s
    print(f"  ERR  {sector:18s} {last_err} (after {retries} retries)")
    return set()


def latest_n200():
    with open(PIT_JSON) as f:
        pit = json.load(f)
    latest_date = max(pit.keys())
    return set(pit[latest_date]), latest_date


def fin_service_set() -> set:
    """Fallback: read FIN SERVICE membership from the curated dict in
    sector_mapping.STOCK_SECTOR_MAP."""
    from sector_mapping import STOCK_SECTOR_MAP
    return {t for t, s in STOCK_SECTOR_MAP.items() if s == "NIFTY FIN SERVICE"}


def main():
    n200, latest_date = latest_n200()
    print(f"NIFTY 200 PIT latest: {latest_date}  ({len(n200)} stocks)")

    print("\nFetching sector constituent CSVs from niftyindices.com:")
    sec_to_syms = {}
    failed = []
    for sec, fname in NIFTYINDICES_URLS.items():
        syms = fetch_constituents(sec, fname)
        if len(syms) == 0 and sec in HARDCODED_FALLBACKS:
            syms = set(HARDCODED_FALLBACKS[sec])
            print(f"  HRD  {sec:18s} {len(syms):3d} stocks (hardcoded fallback — niftyindices fetch failed)")
        elif len(syms) == 0:
            failed.append(sec)
            print(f"  FAIL {sec:18s}   0 stocks")
        else:
            print(f"  OK   {sec:18s} {len(syms):3d} stocks")
        sec_to_syms[sec] = syms
        time.sleep(0.5)   # be polite

    if failed:
        print(f"\n  ABORT: {len(failed)} sector(s) returned empty after retries: {failed}")
        print(f"  Existing CSV left intact to avoid overwriting with bad data.")
        sys.exit(1)

    sec_to_syms["NIFTY PVT BANK"] = sec_to_syms["NIFTY BANK"] - sec_to_syms["NIFTY PSU BANK"]
    print(f"  DRV  NIFTY PVT BANK         {len(sec_to_syms['NIFTY PVT BANK']):3d} stocks  (BANK − PSU BANK)")

    sec_to_syms["NIFTY FIN SERVICE"] = fin_service_set()
    print(f"  FBK  NIFTY FIN SERVICE      {len(sec_to_syms['NIFTY FIN SERVICE']):3d} stocks  (sector_mapping.STOCK_SECTOR_MAP)")

    sec_to_syms["NIFTY OIL & GAS"] = OIL_GAS_HARDCODED
    print(f"  HRD  NIFTY OIL & GAS        {len(sec_to_syms['NIFTY OIL & GAS']):3d} stocks  (hardcoded — no public CSV)")

    rows = []
    overrides_applied = []
    for ticker in sorted(n200):
        listed = [sec for sec in SECTOR_PRECEDENCE if ticker in sec_to_syms.get(sec, set())]
        primary = listed[0] if listed else "OTHER"
        # Manual override beats auto-detection — but listed_in_indices keeps the
        # actual niftyindices membership (empty if stock isn't in any index)
        if ticker in MANUAL_OVERRIDES:
            new_primary, _rationale = MANUAL_OVERRIDES[ticker]
            overrides_applied.append((ticker, primary, new_primary))
            primary = new_primary
        rows.append({
            "ticker":            ticker,
            "primary_sector":    primary,
            "listed_in_indices": ";".join(listed) if listed else "",
        })

    if overrides_applied:
        print(f"\n  Manual overrides applied ({len(overrides_applied)}):")
        for t, was, now in overrides_applied:
            print(f"    {t:14s}  {was:22s} → {now}")

    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ticker", "primary_sector", "listed_in_indices"])
        w.writeheader()
        w.writerows(rows)

    by_primary = {}
    for r in rows:
        by_primary[r["primary_sector"]] = by_primary.get(r["primary_sector"], 0) + 1

    print(f"\n  Wrote {OUTPUT_CSV}  ({len(rows)} N200 stocks)")
    print(f"\n  Primary-sector distribution:")
    for sec, n in sorted(by_primary.items(), key=lambda kv: -kv[1]):
        print(f"    {sec:25s} {n:3d}")

    other = [r["ticker"] for r in rows if r["primary_sector"] == "OTHER"]
    if other:
        print(f"\n  WARNING: {len(other)} N200 stocks not in any of the 14 sectoral indices:")
        for t in other[:30]:
            print(f"    {t}")
        if len(other) > 30:
            print(f"    ... and {len(other)-30} more")
        print("\n  These are likely capital goods / construction / real estate / chemicals "
              "/ etc. Add them to a separate sector index or accept as 'OTHER' for the "
              "Phase 2 sector-mapping use case.")


if __name__ == "__main__":
    main()
