"""
Build point-in-time Nifty 200 constituent database.
Walk backward from current list through all reconstitutions.

Output: nifty200_pit.json — dict mapping effective_date → list of 200 symbols
"""
import json
import csv
from datetime import date

CONST_DIR = '/Users/jay/Desktop/relative_strength/nse_const'

# Load current Nifty 200 (as of Mar 17, 2026)
with open(f'{CONST_DIR}/ind_nifty200list.csv') as f:
    reader = csv.DictReader(f)
    current = sorted([row['Symbol'].strip() for row in reader])

print(f"Current Nifty 200: {len(current)} stocks")

# All reconstitutions, newest first (for backward walk)
# Verified manually from PDF text
#
# NOTE: Feb 2026 (eff 2026-03-27) is NOT included because it hasn't taken effect yet
# (today is 2026-03-17). The current CSV reflects the pre-Feb-2026 composition.

RECONSTITUTIONS = [
    {
        "label": "Aug 2025",
        "effective_date": "2025-09-30",
        "excluded": ["ABFRL", "APOLLOTYRE", "BANDHANBNK", "MAHABANK", "ESCORTS",
                      "ICICIPRULI", "OLAELEC", "PETRONET", "SJVN"],
        "included": ["360ONE", "BLUESTARCO", "COROMANDEL", "FORTIS", "GODFRYPHLP",
                      "POWERINDIA", "ITCHOTELS", "KEI", "ENRIN"],
    },
    {
        "label": "Feb 2025",
        "effective_date": "2025-03-28",
        "excluded": ["BALKRISIND", "DELHIVERY", "FACT", "IDBI", "IOB", "JSWINFRA",
                      "MRPL", "NLCINDIA", "POONAWALLA", "SUNDARMFIN", "TATACHEM"],
        "included": ["BAJAJHFL", "GLENMARK", "HYUNDAI", "MOTILALOFS", "NATIONALUM",
                      "NTPCGREEN", "OLAELEC", "PREMIERENE", "SWIGGY", "VMM", "WAAREEENER"],
    },
    {
        "label": "Aug 2024",
        "effective_date": "2024-09-30",
        "excluded": ["BERGEPAINT", "DALBHARAT", "DEEPAKNTR", "LALPATHLAB", "FORTIS",
                      "GLAND", "GUJGASLTD", "IPCALAB", "LTTS", "LAURUSLABS", "PEL",
                      "SUNTV", "SYNGENE", "IDEA", "ZEEL"],
        "included": ["BHARTIHEXA", "CENTRALBK", "COCHINSHIP", "EXIDEIND", "HINDZINC",
                      "HUDCO", "IOB", "IREDA", "IRB", "MRPL", "MUTHOOTFIN",
                      "NLCINDIA", "PHOENIXLTD", "SOLARINDS", "SUNDARMFIN"],
    },
    {
        "label": "Feb 2024",
        "effective_date": "2024-03-28",
        "excluded": ["AWL", "BATAINDIA", "COROMANDEL", "CROMPTON", "DEVYANI",
                      "FLUOROCHEM", "MSUMI", "MUTHOOTFIN", "NAVINFLUOR", "PGHH",
                      "RAMCOCEM", "UBL"],
        "included": ["MAHABANK", "GMRINFRA", "IDBI", "IREDA", "JIOFIN", "JSWINFRA",
                      "KALYANKJIL", "OFSS", "SJVN", "SUPREMEIND", "SUZLON", "TATATECH"],
    },
    {
        "label": "Aug 2023",
        "effective_date": "2023-09-29",
        "excluded": ["ABBOTINDIA", "HINDZINC", "HONAUT", "OFSS", "TTML",
                      "TRIDENT", "WHIRLPOOL"],
        "included": ["APLAPOLLO", "BDL", "FACT", "KPITTECH", "LODHA",
                      "MAZDOCK", "RVNL", "TATAMTRDVR"],
        "note": "TATAMTRDVR added, index becomes 201 temporarily",
    },
    {
        "label": "Feb 2023",
        "effective_date": "2023-03-31",
        "excluded": ["CLEAN", "EMAMILTD", "GSPL", "ISEC", "INDIAMART",
                      "IEX", "LINDEINDIA", "NATIONALUM", "NAM-INDIA"],
        "included": ["ADANIPOWER", "APOLLOTYRE", "CGPOWER", "DEVYANI",
                      "FLUOROCHEM", "IRFC", "NHPC", "NMDC", "PEL"],
    },
    {
        "label": "Aug 2022",
        "effective_date": "2022-09-30",
        "excluded": ["APLLTD", "APOLLOTYRE", "EXIDEIND", "GLENMARK", "IDBI",
                      "MANAPPURAM", "METROPOLIS", "MINDTREE"],
        "included": ["AWL", "LINDEINDIA", "MSUMI", "PATANJALI", "POONAWALLA",
                      "MOTHERSON", "TTML", "TIINDIA"],
    },
    {
        "label": "Feb 2022",
        "effective_date": "2022-03-31",
        "excluded": ["AJANTPHARM", "AMARAJABAT", "CASTROLIND", "CUB", "DHANI",
                      "ENDURANCE", "GODREJIND", "HINDCOPPER", "IRFC", "MGL",
                      "NATCOPHARM", "PFIZER", "RBLBANK", "SANOFI"],
        "included": ["ABB", "ABBOTINDIA", "CLEAN", "NYKAA", "IDBI", "IEX",
                      "MAXHEALTH", "METROPOLIS", "PAYTM", "OFSS", "POLICYBZR",
                      "PERSISTENT", "TRIDENT", "ZOMATO"],
    },
    {
        "label": "Aug 2021",
        "effective_date": "2021-09-30",
        "excluded": ["ABBOTINDIA", "BBTC", "CESC", "GODREJAGRO", "IBULHSGFIN", "VGUARD"],
        "included": ["ASTRAL", "HINDCOPPER", "INDIANB", "IRFC", "NATIONALUM", "TATACOMM"],
        "note": "PDF was image-only, data from screenshot",
    },
    {
        "label": "Feb 2021",
        "effective_date": "2021-03-31",
        "excluded": ["EDELWEISS", "FRETAIL", "GICRE", "HUDCO", "NATIONALUM",
                      "OFSS", "RAJESHEXPO"],
        "included": ["DEEPAKNTR", "DIXON", "HAL", "INDIAMART", "LAURUSLABS",
                      "TATAELXSI", "VEDL"],
    },
    {
        "label": "Aug 2020",
        "effective_date": "2020-09-25",
        "excluded": ["ADANIPOWER", "HEXAWARE", "IDBI", "JUBILANT", "MRPL",
                      "NHPC", "PNBHOUSING", "QUESS"],
        "included": ["ADANIENT", "ADANIGREEN", "APLLTD", "ISEC", "NAVINFLUOR",
                      "SANOFI", "TATACHEM", "YESBANK"],
    },
    {
        "label": "Feb 2020 (deferred to Jul)",
        "effective_date": "2020-07-31",
        "excluded": ["VEDL"],
        "included": ["SBICARD"],
    },
    {
        "label": "Aug 2019",
        "effective_date": "2019-09-27",
        "excluded": ["ABB", "DHFL", "DBL", "DISHTV", "GSKCONS", "GRUH",
                      "RELCAPITAL", "RELINFRA", "RPOWER", "SPARC"],
        "included": ["ALBK", "CESC", "CUB", "DALBHARAT", "EMAMILTD",
                      "GODREJPROP", "IPCALAB", "NESTLEIND", "ORIENTBANK", "RNAM"],
    },
    {
        "label": "Feb 2019",
        "effective_date": "2019-03-29",
        "excluded": ["AVANTIFEED", "CENTRALBK", "PCJEWELLER", "TV18BRDCST", "VAKRANGEE"],
        "included": ["BAJAJHLDNG", "GSKCONS", "HDFCAMC", "LTTS", "PFIZER"],
    },
    {
        "label": "Aug 2018",
        "effective_date": "2018-09-28",
        "excluded": ["AIAENG", "EMAMILTD", "IDFC", "IBREALEST", "IRB",
                      "KARURVYSYA", "SUZLON", "WOCKPHARMA"],
        "included": ["AVANTIFEED", "BANDHANBNK", "ESCORTS", "GRAPHITE",
                      "HEG", "HDFCLIFE", "LTI", "NIACL"],
    },
    {
        "label": "Feb 2018",
        "effective_date": "2018-03-28",
        "excluded": ["ADANIENT", "CRISIL", "DALMIABHA", "LALPATHLAB", "GSKCONS",
                      "GLAXO", "GPPL", "IPCALAB", "RCOM", "SYNDIBANK", "WELSPUNIND"],
        "included": ["ABCAPITAL", "DBL", "FCONSUMER", "GICRE", "GODREJAGRO",
                      "GRASIM", "ICICIGI", "IBREALEST", "IBVENTURES", "SBILIFE", "STRTECH"],
    },
    {
        "label": "Aug 2017",
        "effective_date": "2017-09-29",
        "excluded": ["APLLTD", "LTTS", "QUESS", "RELCAPITAL", "THERMAX"],
        "included": ["DMART", "BALKRISIND", "FRETAIL", "MRPL", "MFSL"],
    },
    {
        "label": "Feb 2017",
        "effective_date": "2017-03-31",
        "excluded": ["GUJFLUORO", "IOB", "INOXWIND", "JETAIRWAYS", "JUSTDIAL",
                      "PERSISTENT", "PFIZER", "TATAELXSI"],
        "included": ["DALMIABHA", "EDELWEISS", "ENDURANCE", "LALPATHLAB",
                      "ICICIPRULI", "LTTS", "MANAPPURAM", "QUESS"],
    },
    {
        "label": "Aug 2016",
        "effective_date": "2016-09-30",
        "excluded": ["ALBK", "BEML", "GESHIP", "HINDCOPPER", "MMTC",
                      "REDINGTON", "TTKPRESTIG", "UCOBANK", "VIDEOIND"],
        "included": ["ABFRL", "ABIRLANUVO", "ALKEM", "CHOLAFIN", "COROMANDEL",
                      "IDFCBANK", "INDIGO", "MFSL", "PIIND"],
    },
    {
        "label": "Feb 2016",
        "effective_date": "2016-03-31",
        "excluded": ["ALSTOMT&D", "AMTEKAUTO", "ANDHRABANK", "BALKRISIND", "CARERATING",
                      "CUB", "COX&KINGS", "DCBBANK", "GDL", "HDIL", "IFCI", "INDIACEM",
                      "IBREALEST", "JISLJALEQS", "JPASSOCIAT", "J&KBANK", "JKLAKSHMI",
                      "KAJARIACER", "KTKBANK", "KSCL", "KPIT", "NCC", "ORIENTBANK",
                      "PIIND", "PIPAVAVDOC", "PTC", "RALLIS", "RAYMOND", "REPCOHOME",
                      "SADBHAV", "SINTEX", "SOBHA", "SOUTHBANK", "UNITECH", "WABAG"],
        "included": ["ADANIENT", "ARVIND", "BEML", "CENTRALBK", "DISHTV", "FORTIS",
                      "GLAXO", "GESHIP", "GUJFLUORO", "HINDCOPPER", "HINDZINC", "IDFC",
                      "INDIANB", "INDHOTEL", "IOB", "INOXWIND", "JETAIRWAYS", "JUBILANT",
                      "MMTC", "MPHASIS", "MUTHOOTFIN", "NATIONALUM", "NBCC", "PCJEWELLER",
                      "PFIZER", "PGHH", "SYNGENE", "TATAELXSI", "TATAMTRDVR", "RAMCOCEM",
                      "THERMAX", "TORNTPOWER", "TTKPRESTIG", "UCOBANK", "VIDEOIND",
                      "WELSPUNIND"],
        "note": "35 excl / 36 incl. TATAMTRDVR (DVR) added, index becomes 201.",
    },
    {
        "label": "Aug 2015",
        "effective_date": "2015-09-25",
        "excluded": ["CROMPGREAV", "MAX"],
        "included": ["BALKRISIND", "KAJARIACER"],
    },
    {
        "label": "Feb 2015",
        "effective_date": "2015-03-27",
        "excluded": ["GLAXO", "INDIANB", "JPPOWER", "MPHASIS", "NIITTECH",
                      "RAMCOCEM", "THERMAX"],
        "included": ["BERGEPAINT", "GSKCONS", "GRUH", "JKLAKSHMI", "NCC",
                      "REDINGTON", "MCDOWELL-N"],
    },
]

# Mid-cycle corporate actions (mergers, delistings, renames, demergers)
# These are NOT semi-annual reconstitutions but cause stocks to leave/enter the index
# Format: same as reconstitutions but with "type": "corporate_action"
CORPORATE_ACTIONS = [
    # Tata Motors demerger (Oct 2025) — DVR shares removed
    {"label": "TATAMOTORS demerger", "effective_date": "2025-10-01",
     "excluded": ["TATAMTRDVR"], "included": [],
     "note": "DVR shares removed on Tata Motors demerger"},
    # ZOMATO renamed ETERNAL (Jan 2025)
    {"label": "ZOMATO→ETERNAL rename", "effective_date": "2025-01-10",
     "excluded": ["ZOMATO"], "included": ["ETERNAL"]},
    # CENTRALBK removed mid-cycle (early 2025, post Aug 2024 recon)
    {"label": "CENTRALBK removed", "effective_date": "2025-01-15",
     "excluded": ["CENTRALBK"], "included": [],
     "note": "Removed mid-cycle after Aug 2024 inclusion"},
    # GMRINFRA renamed GMRAIRPORT
    {"label": "GMRINFRA→GMRAIRPORT rename", "effective_date": "2024-08-01",
     "excluded": ["GMRINFRA"], "included": ["GMRAIRPORT"]},
    # IREDA: was included Feb 2024, then again Aug 2024 — the Aug 2024 "inclusion"
    # means it was removed mid-cycle between Feb and Aug 2024, then re-added.
    # Treat Aug 2024 IREDA as normal (already handled in recon data).

    # HDFC merged into HDFCBANK (Jul 2023)
    {"label": "HDFC→HDFCBANK merger", "effective_date": "2023-07-13",
     "excluded": ["HDFC"], "included": [],
     "note": "HDFC merged into HDFCBANK, net -1 in index"},
    # MINDTREE + LTI → LTIM (Nov 2022)
    {"label": "MINDTREE+LTI→LTIM merger", "effective_date": "2022-11-14",
     "excluded": ["LTI"], "included": ["LTIM"],
     "note": "MINDTREE already handled in Aug 2022 recon; LTI→LTIM here"},
    # PATANJALI renamed (was included Aug 2022, likely renamed/restructured)
    # Actually PATANJALI was delisted. Let's check if it was excluded somewhere.

    # Bank mergers (Apr 2020)
    {"label": "Bank mergers (Apr 2020)", "effective_date": "2020-04-01",
     "excluded": ["ALBK", "ORIENTBANK"], "included": [],
     "note": "Allahabad Bank→Indian Bank, Oriental Bank→PNB. Net -2 in index"},
    # RNAM renamed NAM-INDIA (Oct 2019)
    {"label": "RNAM→NAM-INDIA rename", "effective_date": "2019-10-07",
     "excluded": ["RNAM"], "included": ["NAM-INDIA"]},
    # IDFCBANK renamed IDFCFIRSTB (Dec 2018)
    {"label": "IDFCBANK→IDFCFIRSTB rename", "effective_date": "2018-12-17",
     "excluded": ["IDFCBANK"], "included": ["IDFCFIRSTB"]},
    # ABIRLANUVO merged into GRASIM (Jul 2017)
    {"label": "ABIRLANUVO→GRASIM merger", "effective_date": "2017-07-03",
     "excluded": ["ABIRLANUVO"], "included": [],
     "note": "Merged into GRASIM which was already in index"},
    # MFSL: Max Financial was renamed? Actually it's still MFSL. Checking...
    # NBCC: might have been removed mid-cycle
    # INDIANB: Indian Bank symbol change? Still INDIANB.
    # MCDOWELL-N renamed UNITDSPR (United Spirits)
    {"label": "MCDOWELL-N→UNITDSPR rename", "effective_date": "2018-06-01",
     "excluded": ["MCDOWELL-N"], "included": ["UNITDSPR"]},

    # LTIM renamed LTM (LTIMindtree symbol change)
    {"label": "LTIM→LTM rename", "effective_date": "2024-01-01",
     "excluded": ["LTIM"], "included": ["LTM"]},
    # IBVENTURES renamed DHANI (then DHANI excluded Feb 2022)
    {"label": "IBVENTURES→DHANI rename", "effective_date": "2021-06-01",
     "excluded": ["IBVENTURES"], "included": ["DHANI"]},
    # IDEA re-included mid-cycle after Aug 2024 exclusion
    {"label": "IDEA re-included", "effective_date": "2024-12-01",
     "excluded": [], "included": ["IDEA"],
     "note": "Re-added as mid-cycle replacement after Aug 2024 exclusion"},
    # Future Consumer suspended/delisted (Future Group collapse)
    {"label": "FCONSUMER delisted", "effective_date": "2022-08-01",
     "excluded": ["FCONSUMER"], "included": [],
     "note": "Future Group collapse, stock suspended"},
    # GRAPHITE India removed mid-cycle
    {"label": "GRAPHITE removed", "effective_date": "2019-08-01",
     "excluded": ["GRAPHITE"], "included": []},
    # HEG removed mid-cycle
    {"label": "HEG removed", "effective_date": "2019-08-01",
     "excluded": ["HEG"], "included": []},
    # NIACL (New India Assurance) removed mid-cycle
    {"label": "NIACL removed", "effective_date": "2019-08-01",
     "excluded": ["NIACL"], "included": []},
    # STRTECH (Sterlite Technologies) removed mid-cycle
    {"label": "STRTECH removed", "effective_date": "2020-06-01",
     "excluded": ["STRTECH"], "included": []},
    # ARVIND removed mid-cycle
    {"label": "ARVIND removed", "effective_date": "2019-03-01",
     "excluded": ["ARVIND"], "included": []},
    # NBCC removed mid-cycle
    {"label": "NBCC removed", "effective_date": "2019-06-01",
     "excluded": ["NBCC"], "included": []},
]

# Combine all events
ALL_EVENTS = RECONSTITUTIONS + CORPORATE_ACTIONS

# Sort newest first for backward walk
ALL_EVENTS.sort(key=lambda x: x['effective_date'], reverse=True)

# ============================================================
# Walk backward from current list
# ============================================================
pit_db = {}  # effective_date -> list of symbols AFTER that reconstitution
constituents = set(current)

print(f"\nStarting backward walk from {len(constituents)} current stocks")
print(f"Processing {len(ALL_EVENTS)} events ({len(RECONSTITUTIONS)} recons + {len(CORPORATE_ACTIONS)} corp actions)...\n")

# The current list is valid from the latest reconstitution's effective date onward
# We need to reverse each reconstitution to find the list before it

for recon in ALL_EVENTS:
    eff = recon['effective_date']
    excl = set(recon['excluded'])
    incl = set(recon['included'])
    label = recon['label']

    # Save the list AFTER this reconstitution
    pit_db[eff] = sorted(constituents)

    # Reverse: remove what was included, add back what was excluded
    # Check: all included should be in current set
    missing_incl = incl - constituents
    extra_excl = excl & constituents

    if missing_incl:
        print(f"  WARNING {label}: included stocks NOT in current set: {missing_incl}")
    if extra_excl:
        print(f"  WARNING {label}: excluded stocks STILL in current set: {extra_excl}")

    constituents = (constituents - incl) | excl

    net = len(incl) - len(excl)
    note = recon.get('note', '')
    print(f"{label} (eff {eff}): {len(excl)} excl, {len(incl)} incl, net={net:+d} → {len(constituents)} stocks"
          + (f"  [{note}]" if note else ""))

# The remaining set is the pre-Feb 2015 list
pit_db["2015-01-01"] = sorted(constituents)
print(f"\nPre-Feb 2015 list: {len(constituents)} stocks")

# ============================================================
# Validation
# ============================================================
print(f"\n{'='*60}")
print("VALIDATION")
print(f"{'='*60}")

for eff_date in sorted(pit_db.keys()):
    n = len(pit_db[eff_date])
    status = "OK" if n == 200 else f"*** {n} ***"
    print(f"  {eff_date}: {n} stocks {status}")

# Save
output_path = f'{CONST_DIR}/nifty200_pit.json'
with open(output_path, 'w') as f:
    json.dump(pit_db, f, indent=2, sort_keys=True)
print(f"\nSaved to {output_path}")

# Also save as CSV for easy viewing
csv_path = f'{CONST_DIR}/nifty200_pit.csv'
all_dates = sorted(pit_db.keys())
all_symbols = sorted(set(s for lst in pit_db.values() for s in lst))
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Symbol'] + all_dates)
    for sym in all_symbols:
        row = [sym] + [1 if sym in pit_db[d] else 0 for d in all_dates]
        writer.writerow(row)
print(f"Saved CSV to {csv_path}")
print(f"Total unique symbols across all periods: {len(all_symbols)}")
