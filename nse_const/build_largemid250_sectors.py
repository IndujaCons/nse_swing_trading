"""
Build sector mapping for Nifty LargeMidcap 250 historical universe (417 symbols).
Fetches yfinance sector/industry, normalises to ~15 NSE-style sectors.
Run once → saves nse_const/largemid250_sectors.json

Sector list (designed for a 4-stock-per-sector cap):
  BANKING, FINANCIAL_SERVICES, IT, PHARMA, HEALTHCARE,
  FMCG, AUTO, AUTO_ANCILLARY, OIL_GAS, ENERGY,
  METALS, CHEMICALS, CAPITAL_GOODS, INFRASTRUCTURE,
  CONSUMER_DURABLES, CONSUMER_DISCRETIONARY, TEXTILES,
  REAL_ESTATE, TELECOM, MEDIA, LOGISTICS, OTHER
"""
import json, time
import yfinance as yf

CONST_DIR = '/Users/jay/Desktop/relative_strength/nse_const'

# ─── Manual overrides ────────────────────────────────────────────────────────
# Anything here skips yfinance lookup entirely.
OVERRIDES = {
    # BANKING
    "ALBK": "BANKING", "AXISBANK": "BANKING", "BANDHANBNK": "BANKING",
    "CANARABANK": "BANKING", "CORPBANK": "BANKING", "CUB": "BANKING",
    "DCBBANK": "BANKING", "FEDERALBNK": "BANKING", "HDFCBANK": "BANKING",
    "ICICIBANK": "BANKING", "IDBI": "BANKING", "IDFCBANK": "BANKING",
    "IDFCFIRSTB": "BANKING", "INDIANB": "BANKING", "INDUSINDBK": "BANKING",
    "IOB": "BANKING", "KARURVYSYA": "BANKING", "KOTAKBANK": "BANKING",
    "LAKSHVILAS": "BANKING", "MAHABANK": "BANKING", "ORIENTBANK": "BANKING",
    "PNB": "BANKING", "RBLBANK": "BANKING", "SBIN": "BANKING",
    "SOUTHBANK": "BANKING", "SYNDIBANK": "BANKING", "UCOBANK": "BANKING",
    "UNIONBANK": "BANKING", "VIJAYABANK": "BANKING", "YESBANK": "BANKING",

    # FINANCIAL SERVICES (NBFC, insurance, AMC, broker, exchange)
    "360ONE": "FINANCIAL_SERVICES", "ABCAPITAL": "FINANCIAL_SERVICES",
    "AWL": "FMCG",   # Adani Wilmar = food
    "BAJAJFINSV": "FINANCIAL_SERVICES", "BAJFINANCE": "FINANCIAL_SERVICES",
    "CHOLAFIN": "FINANCIAL_SERVICES", "CHOLAHLDNG": "FINANCIAL_SERVICES",
    "CREDITACC": "FINANCIAL_SERVICES", "EDELWEISS": "FINANCIAL_SERVICES",
    "GICRE": "FINANCIAL_SERVICES", "GROWW": "FINANCIAL_SERVICES",
    "HDBFS": "FINANCIAL_SERVICES", "HDFC": "FINANCIAL_SERVICES",
    "HDFCAMC": "FINANCIAL_SERVICES", "HDFCLIFE": "FINANCIAL_SERVICES",
    "HUDCO": "FINANCIAL_SERVICES", "IBULHSGFIN": "FINANCIAL_SERVICES",
    "IBVENTURES": "FINANCIAL_SERVICES", "DHANI": "FINANCIAL_SERVICES",
    "ICICIAMC": "FINANCIAL_SERVICES", "ICICIGI": "FINANCIAL_SERVICES",
    "ICICIPRULI": "FINANCIAL_SERVICES", "IIFL": "FINANCIAL_SERVICES",
    "IIFLWAM": "FINANCIAL_SERVICES", "IREDA": "FINANCIAL_SERVICES",
    "IRFC": "FINANCIAL_SERVICES", "ISEC": "FINANCIAL_SERVICES",
    "JIOFIN": "FINANCIAL_SERVICES", "JMFINANCIL": "FINANCIAL_SERVICES",
    "LICHSGFIN": "FINANCIAL_SERVICES", "LICI": "FINANCIAL_SERVICES",
    "MANAPPURAM": "FINANCIAL_SERVICES", "MCX": "FINANCIAL_SERVICES",
    "MOTILALOFS": "FINANCIAL_SERVICES", "MUTHOOTFIN": "FINANCIAL_SERVICES",
    "NAM-INDIA": "FINANCIAL_SERVICES", "NIACL": "FINANCIAL_SERVICES",
    "PNBHOUSING": "FINANCIAL_SERVICES", "POONAWALLA": "FINANCIAL_SERVICES",
    "RECLTD": "FINANCIAL_SERVICES", "RELCAPITAL": "FINANCIAL_SERVICES",
    "RNAM": "FINANCIAL_SERVICES", "SBICARD": "FINANCIAL_SERVICES",
    "SBILIFE": "FINANCIAL_SERVICES", "SHRIRAMCIT": "FINANCIAL_SERVICES",
    "STARHEALTH": "FINANCIAL_SERVICES", "TATACAP": "FINANCIAL_SERVICES",
    "TATAINVEST": "FINANCIAL_SERVICES",

    # IT / TECHNOLOGY
    "AFFLE": "IT", "BIRLASOFT": "IT", "BSOFT": "IT",
    "COFORGE": "IT", "HCLTECH": "IT", "HEXAWARE": "IT",
    "KPITTECH": "IT", "LTI": "IT", "LTIM": "IT", "LTM": "IT",
    "MINDTREE": "IT", "MPHASIS": "IT", "NIITTECH": "IT",
    "OFSS": "IT", "PERSISTENT": "IT", "TATAELXSI": "IT",
    "TATATECH": "IT", "TECHM": "IT", "ZENSARTECH": "IT",
    "QUESS": "IT",   # staffing/tech services

    # PHARMA
    "ABBOTINDIA": "PHARMA", "AJANTPHARM": "PHARMA", "APLLTD": "PHARMA",
    "ASTRAZEN": "PHARMA", "AUROBINDO": "PHARMA", "BIOCON": "PHARMA",
    "CIPLA": "PHARMA", "DIVISLAB": "PHARMA", "ERIS": "PHARMA",
    "GLAND": "PHARMA", "GLENMARK": "PHARMA", "GRANULES": "PHARMA",
    "GSKCONS": "PHARMA", "IPCALAB": "PHARMA", "LAURUSLABS": "PHARMA",
    "LUPIN": "PHARMA", "NATCOPHARM": "PHARMA", "NAVINFLUOR": "PHARMA",
    "PFIZER": "PHARMA", "SANOFI": "PHARMA", "SPARC": "PHARMA",
    "SUNPHARMA": "PHARMA", "SUVENPHAR": "PHARMA", "TORNTPHARM": "PHARMA",
    "VINATIORGA": "PHARMA", "WOCKPHARMA": "PHARMA",

    # HEALTHCARE (hospitals, diagnostics)
    "FORTIS": "HEALTHCARE", "LALPATHLAB": "HEALTHCARE",
    "MAXHEALTH": "HEALTHCARE", "MEDANTA": "HEALTHCARE",
    "METROPOLIS": "HEALTHCARE", "NH": "HEALTHCARE",
    "THYROCARE": "HEALTHCARE",

    # FMCG
    "AVANTIFEED": "FMCG", "BRITANNIA": "FMCG", "COLPAL": "FMCG",
    "DABUR": "FMCG", "EMAMILTD": "FMCG", "GILLETTE": "FMCG",
    "GODREJAGRO": "FMCG", "GODREJCP": "FMCG", "GODFRYPHLP": "FMCG",
    "HATSUN": "FMCG", "ITC": "FMCG", "JYOTHYLAB": "FMCG",
    "KRBL": "FMCG", "MARICO": "FMCG", "NESTLEIND": "FMCG",
    "PATANJALI": "FMCG", "PGHL": "FMCG", "PGHH": "FMCG",
    "RADICO": "FMCG", "TATACONSUM": "FMCG", "UBL": "FMCG",
    "UNITDSPR": "FMCG", "VBL": "FMCG", "ZYDUSWELL": "FMCG",
    "HONAUT": "FMCG",    # Honeywell automation → capital goods actually
    "GSKPHARMA": "PHARMA",

    # AUTO (OEMs)
    "ASHOKLEY": "AUTO", "BAJAJ-AUTO": "AUTO", "EICHERMOT": "AUTO",
    "ESCORTS": "AUTO", "HEROMOTOCO": "AUTO", "M&M": "AUTO",
    "MARUTI": "AUTO", "TATAMOTORS": "AUTO", "TMPV": "AUTO",
    "TMCV": "AUTO", "TATAMTRDVR": "AUTO", "TVSMOTOR": "AUTO",

    # AUTO ANCILLARY
    "AMARAJABAT": "AUTO_ANCILLARY",
    "APOLLOTYRE": "AUTO_ANCILLARY", "BALKRISIND": "AUTO_ANCILLARY",
    "BHARATFORG": "AUTO_ANCILLARY",   # forgings for auto (yfinance → Consumer Cyclical)
    "BOSCHLTD": "AUTO_ANCILLARY", "ENDURANCE": "AUTO_ANCILLARY",
    "EXIDEIND": "AUTO_ANCILLARY",
    "MAHINDCIE": "AUTO_ANCILLARY", "MINDAIND": "AUTO_ANCILLARY",
    "MOTHERSON": "AUTO_ANCILLARY", "MRF": "AUTO_ANCILLARY",
    "MSUMI": "AUTO_ANCILLARY", "SCHAEFFLER": "AUTO_ANCILLARY",
    "SONACOMS": "AUTO_ANCILLARY",
    "SUNDRMFAST": "AUTO_ANCILLARY", "TIINDIA": "AUTO_ANCILLARY",
    "UNOMINDA": "AUTO_ANCILLARY", "VARROC": "AUTO_ANCILLARY",
    "WABCOINDIA": "AUTO_ANCILLARY",

    # OIL & GAS
    "ADANIGAS": "OIL_GAS", "ATGL": "OIL_GAS", "BPCL": "OIL_GAS",
    "CASTROLIND": "OIL_GAS", "GAIL": "OIL_GAS", "GSPL": "OIL_GAS",
    "HINDPETRO": "OIL_GAS", "IGL": "OIL_GAS", "IOC": "OIL_GAS",
    "MGL": "OIL_GAS", "MRPL": "OIL_GAS", "OIL": "OIL_GAS",
    "ONGC": "OIL_GAS", "PETRONET": "OIL_GAS",

    # ENERGY (power, renewables, utilities)
    "ADANIENT": "ENERGY", "ADANIGREEN": "ENERGY", "ADANIPOWER": "ENERGY",
    "CESC": "ENERGY", "NTPC": "ENERGY", "NTPCGREEN": "ENERGY",
    "NHPC": "ENERGY", "NLCINDIA": "ENERGY", "POWERGRID": "ENERGY",
    "PREMIERENE": "ENERGY", "RPOWER": "ENERGY", "SJVN": "ENERGY",
    "SUZLON": "ENERGY", "TATAPOWER": "ENERGY", "TORNTPOWER": "ENERGY",
    "WAAREEENER": "ENERGY", "ENRIN": "ENERGY",

    # METALS & MINING
    "AIAENG": "METALS", "APLAPOLLO": "METALS",
    "COALINDIA": "METALS", "GRAPHITE": "METALS",
    "HEG": "METALS", "HINDALCO": "METALS",
    "HINDCOPPER": "METALS", "JSL": "METALS",
    "JSWSTEEL": "METALS", "KIOCL": "METALS",
    "NATIONALUM": "METALS", "NMDC": "METALS",
    "RATNAMANI": "METALS", "SAIL": "METALS",
    "TATASTEEL": "METALS", "VEDL": "METALS",
    "WELCORP": "METALS",

    # CHEMICALS (incl. paints, specialty, agrochem, fertilizers)
    "AARTIIND": "CHEMICALS", "AKZOINDIA": "CHEMICALS",
    "ALKYLAMINE": "CHEMICALS", "ATUL": "CHEMICALS",
    "BAYERCROP": "CHEMICALS", "CLEAN": "CHEMICALS",
    "COROMANDEL": "CHEMICALS", "DEEPAKNTR": "CHEMICALS",
    "FACT": "CHEMICALS", "FINEORG": "CHEMICALS",
    "FLUOROCHEM": "CHEMICALS", "GUJFLUORO": "CHEMICALS",
    "KANSAINER": "CHEMICALS", "LINDEINDIA": "CHEMICALS",
    "PHILIPCARB": "CHEMICALS", "PIDILITIND": "CHEMICALS",
    "SUMICHEM": "CHEMICALS",

    # CAPITAL GOODS / ENGINEERING / DEFENSE
    "ABB": "CAPITAL_GOODS", "BDL": "CAPITAL_GOODS",
    "BHEL": "CAPITAL_GOODS", "CGPOWER": "CAPITAL_GOODS",
    "COCHINSHIP": "CAPITAL_GOODS", "CUMMINSIND": "CAPITAL_GOODS",
    "ELGIEQUIP": "CAPITAL_GOODS", "ENGINERSIN": "CAPITAL_GOODS",
    "GVT&D": "CAPITAL_GOODS", "GRINDWELL": "CAPITAL_GOODS",
    "HAL": "CAPITAL_GOODS", "HONAUT": "CAPITAL_GOODS",
    "KEC": "CAPITAL_GOODS", "KEI": "CAPITAL_GOODS",
    "LLOYDSME": "CAPITAL_GOODS", "MAZDOCK": "CAPITAL_GOODS",
    "PEL": "CAPITAL_GOODS", "POLYCAB": "CAPITAL_GOODS",
    "POWERINDIA": "CAPITAL_GOODS", "SIEMENS": "CAPITAL_GOODS",
    "SKFINDIA": "CAPITAL_GOODS", "THERMAX": "CAPITAL_GOODS",
    "TIMKEN": "CAPITAL_GOODS", "TRITURBINE": "CAPITAL_GOODS",
    "ZFCVINDIA": "CAPITAL_GOODS", "APARINDS": "CAPITAL_GOODS",

    # CEMENT & BUILDING MATERIALS
    "ACC": "CEMENT", "AMBUJACEMENT": "CEMENT",
    "DALMIABHA": "CEMENT", "GRASIM": "CEMENT",
    "JKCEMENT": "CEMENT", "NUVOCO": "CEMENT",
    "RAMCOCEM": "CEMENT", "SHREECEM": "CEMENT",
    "ULTRACEMCO": "CEMENT",

    # INFRASTRUCTURE / CONSTRUCTION / PORTS
    "ADANIPORTS": "INFRASTRUCTURE", "ASTRAL": "INFRASTRUCTURE",
    "DBL": "INFRASTRUCTURE", "GMRAIRPORT": "INFRASTRUCTURE",
    "GMRINFRA": "INFRASTRUCTURE", "GPPL": "INFRASTRUCTURE",
    "IRB": "INFRASTRUCTURE", "JSWINFRA": "INFRASTRUCTURE",
    "NBCC": "INFRASTRUCTURE", "RVNL": "INFRASTRUCTURE",
    "WABAG": "INFRASTRUCTURE",

    # REAL ESTATE
    "DBREALTY": "REAL_ESTATE", "EMBASSY": "REAL_ESTATE",
    "IBREALEST": "REAL_ESTATE", "LODHA": "REAL_ESTATE",
    "MINDSPACE": "REAL_ESTATE", "OBEROIRLTY": "REAL_ESTATE",
    "PHENIXMILLS": "REAL_ESTATE",

    # CONSUMER DURABLES / ELECTRONICS
    "AMBER": "CONSUMER_DURABLES", "BLUESTARCO": "CONSUMER_DURABLES",
    "CROMPTON": "CONSUMER_DURABLES", "DIXON": "CONSUMER_DURABLES",
    "KAJARIACER": "CONSUMER_DURABLES", "LGEINDIA": "CONSUMER_DURABLES",
    "SYMPHONY": "CONSUMER_DURABLES", "VGUARD": "CONSUMER_DURABLES",
    "VOLTAS": "CONSUMER_DURABLES", "WHIRLPOOL": "CONSUMER_DURABLES",

    # CONSUMER DISCRETIONARY (retail, jewellery, fashion, QSR, footwear)
    "BATAINDIA": "CONSUMER_DISCRETIONARY",
    "DEVYANI": "CONSUMER_DISCRETIONARY",
    "KALYANKJIL": "CONSUMER_DISCRETIONARY",
    "MANYAVAR": "CONSUMER_DISCRETIONARY",
    "RAJESHEXPO": "CONSUMER_DISCRETIONARY",
    "RELAXO": "CONSUMER_DISCRETIONARY",
    "TITAN": "CONSUMER_DISCRETIONARY",
    "TRENT": "CONSUMER_DISCRETIONARY",
    "VAIBHAVGBL": "CONSUMER_DISCRETIONARY",
    "DMART": "CONSUMER_DISCRETIONARY",
    "SWIGGY": "CONSUMER_DISCRETIONARY",
    "LENSKART": "CONSUMER_DISCRETIONARY",
    "MANKIND": "PHARMA",  # Mankind Pharma

    # TEXTILES / APPAREL
    "ABFRL": "TEXTILES", "ARVINDFASN": "TEXTILES",
    "KPRMILL": "TEXTILES", "PAGEIND": "TEXTILES",
    "TRIDENT": "TEXTILES", "WELSPUNIND": "TEXTILES",

    # TELECOM
    "BHARTIHEXA": "TELECOM", "IDEA": "TELECOM",
    "INDUSTOWER": "TELECOM", "INFRATEL": "TELECOM",
    "RCOM": "TELECOM", "TATACOMM": "TELECOM", "TTML": "TELECOM",

    # MEDIA / ENTERTAINMENT
    "DBCORP": "MEDIA", "DISHTV": "MEDIA",
    "SUNTV": "MEDIA", "TV18BRDCST": "MEDIA",
    "ZEEL": "MEDIA",

    # LOGISTICS
    "BLUEDART": "LOGISTICS", "DELHIVERY": "LOGISTICS",
    "MAHLOG": "LOGISTICS", "TITAGARH": "CAPITAL_GOODS",

    # FMCG / SPECIALTY (corrections)
    "INDIAMART": "IT",       # B2B marketplace = tech/IT
    "IRCTC": "CONSUMER_DISCRETIONARY",
    "JUBILANT": "CONSUMER_DISCRETIONARY",
    "ITCHOTELS": "CONSUMER_DISCRETIONARY",
    "HYUNDAI": "AUTO",
    "OLAELEC": "AUTO",

    # New-listing misc
    "AIIL": "FINANCIAL_SERVICES",
    "ANTHEM": "PHARMA",
    "HEXT": "IT",
    "JSWJIEN": "INFRASTRUCTURE",

    # Delisted / merged stubs
    "FCONSUMER": "FMCG", "FRETAIL": "CONSUMER_DISCRETIONARY",
    "GRUH": "FINANCIAL_SERVICES",
    "GSKCONS": "PHARMA",   # GSK Consumer = Horlicks → FMCG but classified under pharma parent
    "STRTECH": "IT",       # Strides Tech = pharma actually
    "RELINFRA": "INFRASTRUCTURE",
    "DHFL": "FINANCIAL_SERVICES",
    "JPASSOCIAT": "INFRASTRUCTURE",
    "JPPOWER": "ENERGY",
    "JKIL": "INFRASTRUCTURE",
    "IBULISL": "FINANCIAL_SERVICES",
    "IBULSGFIN": "FINANCIAL_SERVICES",
    "HDIL": "REAL_ESTATE",
    "VMM": "CAPITAL_GOODS",

    # Additional manually resolved
    "GET&D": "CAPITAL_GOODS",      # GE T&D India — power transmission equipment
    "IDFC": "FINANCIAL_SERVICES",  # IDFC Limited — merged into IDFC FIRST Bank
    "ZOMATO": "CONSUMER_DISCRETIONARY",  # old ticker before rename to ETERNAL
}

# ─── yfinance sector → standardised sector ───────────────────────────────────
# Used only for symbols NOT in OVERRIDES.
# industry string used to sub-divide Financial Services into Banking.
BANK_INDUSTRIES = {"Banks—Regional", "Banks—Diversified", "Bank", "Savings & Cooperative Banks"}

YF_SECTOR_TO_STD = {
    "Financial Services":       "FINANCIAL_SERVICES",   # refined below via industry
    "Technology":               "IT",
    "Healthcare":               "PHARMA",               # refined below via industry
    "Consumer Defensive":       "FMCG",
    "Consumer Cyclical":        "CONSUMER_DISCRETIONARY",
    "Basic Materials":          "CHEMICALS",
    "Energy":                   "OIL_GAS",
    "Industrials":              "CAPITAL_GOODS",
    "Communication Services":   "TELECOM",
    "Real Estate":              "REAL_ESTATE",
    "Utilities":                "ENERGY",
}

HOSPITAL_INDUSTRIES = {"Medical Care Facilities", "Diagnostics & Research",
                       "Medical Devices", "Health Information Services"}


def classify(sym, info):
    sector   = info.get("sector", "") or ""
    industry = info.get("industry", "") or ""
    std = YF_SECTOR_TO_STD.get(sector, "OTHER")
    # Refine Financial Services → Banking
    if std == "FINANCIAL_SERVICES" and industry in BANK_INDUSTRIES:
        std = "BANKING"
    # Refine Healthcare → hospitals vs pharma
    if std == "PHARMA" and industry in HOSPITAL_INDUSTRIES:
        std = "HEALTHCARE"
    return std


# ─── Main ────────────────────────────────────────────────────────────────────
with open(f"{CONST_DIR}/largemid250_pit.json") as f:
    pit_db = json.load(f)

all_symbols = sorted(set(s for lst in pit_db.values() for s in lst))
print(f"Total symbols: {len(all_symbols)}")

sectors = {}
missing = []

for sym in all_symbols:
    if sym in OVERRIDES:
        sectors[sym] = OVERRIDES[sym]
        continue

    nse_sym = sym.replace("&", "%26")   # URL-encode for yfinance
    try:
        info = yf.Ticker(f"{nse_sym}.NS").info
        s = classify(sym, info)
        if s == "OTHER":
            missing.append(sym)
        sectors[sym] = s
        print(f"  {sym:20s} {info.get('sector','?'):25s} {info.get('industry','?'):35s} → {s}")
    except Exception as e:
        sectors[sym] = "OTHER"
        missing.append(sym)
        print(f"  {sym:20s} ERROR: {e}")
    time.sleep(0.15)

with open(f"{CONST_DIR}/largemid250_sectors.json", "w") as f:
    json.dump(sectors, f, indent=2, sort_keys=True)

print(f"\nSaved → {CONST_DIR}/largemid250_sectors.json")
print(f"Symbols with manual override : {sum(1 for s in all_symbols if s in OVERRIDES)}")
print(f"Symbols resolved via yfinance: {len(all_symbols) - sum(1 for s in all_symbols if s in OVERRIDES) - len(missing)}")
print(f"Unresolved (OTHER)           : {len(missing)}")
if missing:
    print("  ", missing)
