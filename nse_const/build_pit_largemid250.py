"""
Build point-in-time Nifty LargeMidcap 250 constituent database.
Walk backward from current list through all reconstitutions.

Index launched: November 30, 2017
Output: nse_const/largemid250_pit.json

Method: identical to build_pit_nifty200.py — start from current CSV,
reverse each semi-annual reconstitution to reconstruct historical snapshots.
DUMMY Vedanta placeholder stocks (DUMMYVEDL1-4) excluded from all snapshots.
"""
import json
import csv
from datetime import date

CONST_DIR = '/Users/jay/Desktop/relative_strength/nse_const'

# Load current LargeMidcap 250 — strip DUMMY placeholder stocks (Vedanta demerger)
DUMMY_STOCKS = {"DUMMYVEDL1", "DUMMYVEDL2", "DUMMYVEDL3", "DUMMYVEDL4"}
with open(f'{CONST_DIR}/ind_niftylargemidcap250list.csv') as f:
    reader = csv.DictReader(f)
    current = sorted([
        row['Symbol'].strip()
        for row in reader
        if row['Symbol'].strip() not in DUMMY_STOCKS
    ])

print(f"Current LargeMidcap 250 (real stocks): {len(current)}")
assert len(current) == 250, f"Expected 250, got {len(current)}"

# ─── Semi-annual reconstitutions (newest → oldest) ─────────────────────────
# Extracted from NSE press release PDFs in nse_const/.
# Each entry = the change that took effect on 'effective_date'.
# Backward walk: we REMOVE what was included, ADD BACK what was excluded.

RECONSTITUTIONS = [
    {
        "label": "Feb 2026",
        "effective_date": "2026-03-30",
        "excluded": ["DEEPAKNTR", "FACT", "GUJGASLTD", "IDBI", "IOB",
                     "IGL", "IRB", "PGHH", "SONACOMS", "SYNGENE",
                     "TATATECH", "UCOBANK"],
        "included": ["ANTHEM", "AIIL", "GROWW", "HDBFS", "ICICIAMC",
                     "LAURUSLABS", "LENSKART", "LGEINDIA", "MCX",
                     "RADICO", "TATACAP", "TMCV"],
    },
    {
        "label": "Aug 2025",
        "effective_date": "2025-09-30",
        "excluded": ["ABFRL", "BANDHANBNK", "EMAMILTD", "GLAND", "MRPL",
                     "MSUMI", "OLAELEC", "STARHEALTH", "SUNTV"],
        "included": ["FACT", "GODFRYPHLP", "HEXT", "IDBI", "IOB",
                     "ITCHOTELS", "PGHH", "ENRIN", "UCOBANK"],
    },
    {
        "label": "Feb 2025",
        "effective_date": "2025-03-28",
        "excluded": ["BAYERCROP", "CARBORUNIV", "DELHIVERY", "FACT", "GRINDWELL",
                     "IDBI", "IOB", "METROBRAND", "POONAWALLA", "PGHL",
                     "SKFINDIA", "SUNDRMFAST", "TATACHEM", "TIMKEN", "ZFCVINDIA"],
        "included": ["360ONE", "APARINDS", "BAJAJHFL", "BLUESTARCO",
                     "GVT&D", "GLENMARK", "HYUNDAI", "MOTILALOFS",
                     "NATIONALUM", "NTPCGREEN", "OLAELEC", "PREMIERENE",
                     "SWIGGY", "VMM", "WAAREEENER"],
    },
    {
        "label": "Aug 2024",
        "effective_date": "2024-09-30",
        "excluded": ["ATUL", "BATAINDIA", "DEVYANI", "LALPATHLAB", "ISEC",
                     "KAJARIACER", "KANSAINER", "LAURUSLABS",
                     "PEL", "SUMICHEM", "RAMCOCEM", "MANYAVAR", "IDEA", "ZEEL"],
        "included": ["BHARTIHEXA", "CENTRALBK", "COCHINSHIP", "EXIDEIND",
                     "MEDANTA", "POWERINDIA", "HUDCO", "IOB", "IREDA", "IRB",
                     "MRPL", "NAM-INDIA", "NLCINDIA", "TATAINVEST"],
    },
    {
        "label": "Feb 2024",
        "effective_date": "2024-03-28",
        "excluded": ["AARTIIND", "BLUEDART", "CROMPTON", "NAVINFLUOR",
                     "PFIZER", "RAJESHEXPO", "RELAXO", "TRIDENT",
                     "VINATIORGA", "WHIRLPOOL"],
        "included": ["IDBI", "IREDA", "JIOFIN", "JSWINFRA", "KALYANKJIL",
                     "KEI", "LLOYDSME", "POLYCAB", "SJVN", "SUZLON", "TATATECH"],
    },
    {
        "label": "Aug 2023",
        "effective_date": "2023-09-29",
        # TATAMTRDVR (Tata Motors DVR) added as extra; count becomes 251 temporarily
        "excluded": ["AAVAS", "AFFLE", "ALKYLAMINE", "CLEAN", "FINEORG",
                     "HAPPSTMNDS", "NAM-INDIA", "TTML"],
        "included": ["MAHABANK", "BDL", "CARBORUNIV", "FACT", "JSL",
                     "KPITTECH", "MAZDOCK", "RVNL", "TATAMTRDVR"],
    },
    {
        "label": "Feb 2023",
        "effective_date": "2023-03-31",
        "excluded": ["APLLTD", "GSPL", "HATSUN", "INDIAMART", "IEX",
                     "NATCOPHARM", "NATIONALUM", "SANOFI"],
        "included": ["AARTIIND", "ADANIPOWER", "APOLLOTYRE", "FINEORG",
                     "METROBRAND", "NMDC", "PEL", "TIMKEN"],
    },
    {
        "label": "Sep 2022",
        "effective_date": "2022-09-30",
        "excluded": ["APOLLOTYRE", "EXIDEIND", "GLENMARK", "IDBI",
                     "MANAPPURAM", "METROPOLIS", "MINDTREE", "NUVOCO"],
        "included": ["AWL", "DEVYANI", "KPRMILL", "MSUMI", "PATANJALI",
                     "POONAWALLA", "MOTHERSON", "MANYAVAR"],
    },
    {
        "label": "Feb 2022",
        "effective_date": "2022-03-31",
        "excluded": ["AMARAJABAT", "CASTROLIND", "CHOLAHLDNG", "CUB", "DHANI",
                     "GILLETTE", "HINDCOPPER", "ITI", "MGL", "MOTILALOFS",
                     "RBLBANK", "SUVENPHAR", "TTKPRESTIG", "VAIBHAVGBL",
                     "ZYDUSWELL"],
        "included": ["CGPOWER", "CLEAN", "NYKAA", "GRINDWELL", "FLUOROCHEM",
                     "HAPPSTMNDS", "IDBI", "IEX", "NUVOCO", "PAYTM",
                     "POLICYBZR", "STARHEALTH", "TTML", "TRIDENT", "ZOMATO"],
    },
    {
        "label": "Aug 2021",
        "effective_date": "2021-09-30",
        # PDF is image-only; extracted via OCR
        "excluded": ["AKZOINDIA", "ASTRAZEN", "BBTC", "CESC", "CREDITACC",
                     "GILLETTE", "GODREJAGRO", "IDBI", "IIFLWAM",
                     "IBULHSGFIN", "JMFINANCIL", "MOTILALOFS", "PGHL",
                     "SJVN", "VGUARD"],
        "included": ["AFFLE", "ALKYLAMINE", "APLAPOLLO", "BLUEDART", "EMBASSY",
                     "HINDCOPPER", "INDIANB", "IRFC", "KAJARIACER", "LINDEINDIA",
                     "LODHA", "MINDSPACE", "NATIONALUM", "VAIBHAVGBL", "WABCOINDIA"],
    },
    {
        "label": "Feb 2021",
        "effective_date": "2021-03-31",
        "excluded": ["EDELWEISS", "ERIS", "FRETAIL", "HUDCO", "MRPL",
                     "NATIONALUM", "NLCINDIA", "SFL", "SHRIRAMCIT",
                     "SYMPHONY", "WABCOINDIA"],
        "included": ["DEEPAKNTR", "DIXON", "IDBI", "INDIAMART", "LAURUSLABS",
                     "MAXHEALTH", "PERSISTENT", "PGHL", "SUVENPHAR",
                     "TATAELXSI", "VEDL"],
    },
    {
        "label": "Aug 2020",
        "effective_date": "2020-09-25",
        "excluded": ["ADANIPOWER", "BLUEDART", "CENTRALBK", "EIHOTEL",
                     "HEXAWARE", "IDBI", "IOB", "JUBILANT", "PNBHOUSING",
                     "QUESS", "UCOBANK"],
        "included": ["ADANIENT", "ASTRAZEN", "ITI", "JMFINANCIL", "METROPOLIS",
                     "NAVINFLUOR", "SFL", "TATACHEM", "TIINDIA", "YESBANK",
                     "ZYDUSWELL"],
    },
    {
        "label": "Jul 2020 (special)",
        "effective_date": "2020-07-31",
        "excluded": ["VEDL"],
        "included": ["SBICARD"],
    },
    {
        # Feb 2020 recon deferred due to COVID-19 volatility (NSE press release 23 Mar 2020)
        # Effective June 26, 2020 (ind_prs10062020.pdf); 11 in / 11 out → net 0
        "label": "Feb 2020 (deferred, eff Jun 2020)",
        "effective_date": "2020-06-26",
        "excluded": ["ENGINERSIN", "FCONSUMER", "GRAPHITE", "HEG", "INDIANB",
                     "KRBL", "MAHINDCIE", "NBCC", "STRTECH", "TATAMTRDVR", "VARROC"],
        "included": ["ABB", "ADANIGREEN", "AKZOINDIA", "CREDITACC", "IIFLWAM", "IRCTC",
                     "JKCEMENT", "MINDAIND", "NIITTECH", "SUMICHEM", "TATACOMM"],
    },
    {
        "label": "Feb 2019",
        "effective_date": "2019-03-29",
        "excluded": ["AARTIIND", "AVANTIFEED", "GET&D", "PCJEWELLER",
                     "TV18BRDCST", "VAKRANGEE"],
        "included": ["ATUL", "ERIS", "GUJFLUORO", "HDFCAMC", "IPCALAB", "RELAXO"],
    },
    {
        "label": "Aug 2019",
        "effective_date": "2019-09-27",
        "excluded": ["ABB", "DHFL", "DBL", "DISHTV", "FINCABLES", "GSKCONS",
                     "GRUH", "JMFINANCIL", "KIOCL", "RELCAPITAL", "RELINFRA",
                     "RPOWER", "SPARC"],
        "included": ["AAVAS", "ABBOTINDIA", "ADANIGAS", "ALBK", "BAYERCROP",
                     "CESC", "CORPBANK", "FORTIS", "IOB", "NESTLEIND",
                     "ORIENTBANK", "UCOBANK", "VINATIORGA"],
    },
    {
        "label": "Aug 2018",
        "effective_date": "2018-09-28",
        "excluded": ["AKZOINDIA", "GUJFLUORO", "IDFC", "IBREALEST", "IRB",
                     "KAJARIACER", "KARURVYSYA", "SCHAEFFLER", "SUZLON",
                     "TTKPRESTIG", "WOCKPHARMA"],
        "included": ["ASTRAL", "BANDHANBNK", "ESCORTS", "GRAPHITE", "HEG",
                     "HDFCLIFE", "HAL", "ISEC", "RNAM", "SUNDARMFIN", "NIACL"],
    },
    {
        "label": "Feb 2018",
        # PDF: "effective from April 02, 2018 (close of March 28, 2018)"
        "effective_date": "2018-04-02",
        "excluded": ["ADANIENT", "DBCORP", "DALMIABHA", "LALPATHLAB", "GPPL",
                     "IOB", "IPCALAB", "JYOTHYLAB", "NH", "RCOM",
                     "SYNDIBANK", "VTL", "VIJAYABANK", "WELSPUNIND"],
        "included": ["ABCAPITAL", "AVANTIFEED", "DBL", "FCONSUMER", "GICRE",
                     "GODREJAGRO", "GRASIM", "HONAUT", "ICICIGI", "IIFL",
                     "IBREALEST", "IBVENTURES", "SBILIFE", "STRTECH"],
    },
]

# ─── Mid-cycle corporate actions ───────────────────────────────────────────
# Mergers, delistings, renames that changed composition between reconstitutions.

CORPORATE_ACTIONS = [
    # LTIM renamed LTM (LTIMindtree → LTM Limited, Feb 27, 2026)
    {"label": "LTIM→LTM rename", "effective_date": "2026-02-27",
     "excluded": ["LTIM"], "included": ["LTM"]},

    # TATAMOTORS renamed TMPV when Tata Motors demerged (TMCV listed Nov 10, 2025;
    # original Tata Motors entity continued as TMPV)
    {"label": "TATAMOTORS→TMPV rename", "effective_date": "2025-11-10",
     "excluded": ["TATAMOTORS"], "included": ["TMPV"],
     "note": "Original Tata Motors continued as TMPV; TMCV listed as new entity Nov 10 2025"},

    # Tata Motors DVR removed on demerger (Oct 2025)
    {"label": "TATAMTRDVR demerger removal", "effective_date": "2025-10-01",
     "excluded": ["TATAMTRDVR"], "included": [],
     "note": "DVR shares removed; index returns to 250"},

    # ZOMATO renamed ETERNAL (Jan 2025)
    {"label": "ZOMATO→ETERNAL rename", "effective_date": "2025-01-10",
     "excluded": ["ZOMATO"], "included": ["ETERNAL"]},

    # CENTRALBK removed mid-cycle → MANKIND added as replacement (Jan 2025)
    {"label": "CENTRALBK removed → MANKIND", "effective_date": "2025-01-15",
     "excluded": ["CENTRALBK"], "included": ["MANKIND"],
     "note": "CENTRALBK removed mid-cycle; MANKIND (IPO Apr 2023) added as replacement"},

    # GMRINFRA renamed GMRAIRPORT (Dec 11, 2024)
    {"label": "GMRINFRA→GMRAIRPORT rename", "effective_date": "2024-12-11",
     "excluded": ["GMRINFRA"], "included": ["GMRAIRPORT"]},

    # HDFC merged into HDFCBANK (Jul 2023)
    {"label": "HDFC→HDFCBANK merger", "effective_date": "2023-07-13",
     "excluded": ["HDFC"], "included": [],
     "note": "HDFC merged into HDFCBANK; net -1"},

    # LTI renamed LTIM (LTIMindtree, Nov 2022)
    {"label": "LTI→LTIM rename", "effective_date": "2022-11-14",
     "excluded": ["LTI"], "included": ["LTIM"],
     "note": "LTI merged with Mindtree; Mindtree already excluded Sep 2022 recon"},

    # Future Consumer suspended/delisted → LICI added as mid-cycle replacement (Aug 2022)
    {"label": "FCONSUMER delisted → LICI", "effective_date": "2022-08-01",
     "excluded": ["FCONSUMER"], "included": ["LICI"],
     "note": "Future Group collapse; LICI (IPO May 2022) added as mid-cycle replacement"},

    # IBVENTURES renamed DHANI (Jun 2021)
    {"label": "IBVENTURES→DHANI rename", "effective_date": "2021-06-01",
     "excluded": ["IBVENTURES"], "included": ["DHANI"]},

    # MINDAIND renamed UNOMINDA (Minda Industries → Uno Minda, Oct 2022)
    {"label": "MINDAIND→UNOMINDA rename", "effective_date": "2022-10-01",
     "excluded": ["MINDAIND"], "included": ["UNOMINDA"]},

    # NIITTECH renamed COFORGE (NIIT Technologies → Coforge, Sep 2020)
    {"label": "NIITTECH→COFORGE rename", "effective_date": "2020-09-01",
     "excluded": ["NIITTECH"], "included": ["COFORGE"]},

    # ADANIGAS renamed ATGL (Adani Total Gas Limited, Jan 1, 2021)
    {"label": "ADANIGAS→ATGL rename", "effective_date": "2021-01-01",
     "excluded": ["ADANIGAS"], "included": ["ATGL"]},

    # INFRATEL renamed INDUSTOWER after merger with Indus Towers (Dec 10, 2020)
    {"label": "INFRATEL→INDUSTOWER rename", "effective_date": "2020-12-10",
     "excluded": ["INFRATEL"], "included": ["INDUSTOWER"]},

    # Bank mergers (Apr 2020):
    #   Allahabad Bank → Indian Bank (ALBK→INDIANB)
    #   Oriental Bank → PNB (ORIENTBANK→PNB; PNB already in index)
    #   Corporation Bank → Union Bank (CORPBANK→UNIONBANK; UNIONBANK already in index)
    {"label": "Bank mergers Apr 2020", "effective_date": "2020-04-01",
     "excluded": ["ALBK", "ORIENTBANK", "CORPBANK"], "included": [],
     "note": "Merged into INDIANB/PNB/UNIONBANK which were already in index"},

    # VARROC mid-cycle entry into Nifty Midcap 150 (IPO Jun 2018; not in Aug 2018/Feb 2019/Aug 2019 recons per PDFs)
    # Exact date unknown — placed Oct 2019 as best estimate (post-Aug-2019 effective date)
    {"label": "VARROC mid-cycle IPO entry", "effective_date": "2019-10-15",
     "excluded": [], "included": ["VARROC"],
     "note": "Exact entry date unknown; confirmed NOT in Aug2018/Feb2019/Aug2019 recons per PDFs"},

    # RNAM renamed NAM-INDIA (Nippon AMC, Oct 2019)
    {"label": "RNAM→NAM-INDIA rename", "effective_date": "2019-10-07",
     "excluded": ["RNAM"], "included": ["NAM-INDIA"]},

    # IDFCBANK renamed IDFCFIRSTB (Dec 2018)
    {"label": "IDFCBANK→IDFCFIRSTB rename", "effective_date": "2018-12-17",
     "excluded": ["IDFCBANK"], "included": ["IDFCFIRSTB"]},
]

# ─── Combine and sort newest → oldest ──────────────────────────────────────
ALL_EVENTS = RECONSTITUTIONS + CORPORATE_ACTIONS
ALL_EVENTS.sort(key=lambda x: x['effective_date'], reverse=True)

# ─── Backward walk ─────────────────────────────────────────────────────────
pit_db = {}
constituents = set(current)

print(f"\nStarting backward walk from {len(constituents)} current stocks")
print(f"Processing {len(ALL_EVENTS)} events "
      f"({len(RECONSTITUTIONS)} recons + {len(CORPORATE_ACTIONS)} corp actions)...\n")

for event in ALL_EVENTS:
    eff   = event['effective_date']
    excl  = set(event['excluded'])
    incl  = set(event['included'])
    label = event['label']

    # Save state AFTER this event (valid from eff → next event)
    pit_db[eff] = sorted(constituents)

    # Validate
    missing_incl = incl - constituents
    extra_excl   = excl & constituents
    if missing_incl:
        print(f"  WARNING {label}: included NOT in current set: {missing_incl}")
    if extra_excl:
        print(f"  WARNING {label}: excluded STILL in current set: {extra_excl}")

    # Reverse the event
    constituents = (constituents - incl) | excl

    net  = len(incl) - len(excl)
    note = event.get('note', '')
    print(f"{label} ({eff}): -{len(excl)} +{len(incl)} net={net:+d} → {len(constituents)} stocks"
          + (f"  [{note}]" if note else ""))

# State before the first reconstitution = initial composition at index launch
pit_db["2017-11-30"] = sorted(constituents)
print(f"\nLaunch composition (2017-11-30): {len(constituents)} stocks")

# ─── Validation ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("VALIDATION — stock count at each snapshot date")
print(f"{'='*60}")
for d in sorted(pit_db.keys()):
    n = len(pit_db[d])
    flag = "" if n == 250 else f"  *** {n} (expected 250) ***"
    print(f"  {d}: {n}{flag}")

# ─── Save outputs ──────────────────────────────────────────────────────────
out_json = f'{CONST_DIR}/largemid250_pit.json'
with open(out_json, 'w') as f:
    json.dump(pit_db, f, indent=2, sort_keys=True)
print(f"\nSaved JSON → {out_json}")

out_csv = f'{CONST_DIR}/largemid250_pit.csv'
all_dates   = sorted(pit_db.keys())
all_symbols = sorted(set(s for lst in pit_db.values() for s in lst))
with open(out_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Symbol'] + all_dates)
    for sym in all_symbols:
        row = [sym] + [1 if sym in pit_db[d] else 0 for d in all_dates]
        writer.writerow(row)
print(f"Saved CSV  → {out_csv}")
print(f"Total unique symbols across all periods: {len(all_symbols)}")
