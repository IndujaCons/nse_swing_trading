#!/usr/bin/env python3
"""
build_sector_map_pit.py — Build a PIT (point-in-time) sector mapping.

For each reconstitution date in nifty200_pit.json (default) or
nifty500_pit.json (--n500), emit the sector classification of every
member stock at that date.

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
 - MANUAL_OVERRIDES (this file) — for stocks no NSE sector index currently
   lists (delisted, merged, bankrupt, niche) where we still want a
   classification

Usage:
  python3 build_sector_map_pit.py           # N200 → nifty200_sector_map_pit.json
  python3 build_sector_map_pit.py --n500    # N500 → nifty500_sector_map_pit.json

Schema: { "<reconstitution_date>": { "<ticker>": {"primary_sector",
                                                   "listed_in_indices",
                                                   "resolved_via_alias?"} }, ... }
"""

import argparse
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
# historical-only tickers — delisted/merged/bankrupt/niche — that we still
# want classified. ALL of these apply at every PIT date.
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
    "ABIRLANUVO":  "NIFTY INFRA",              # Aditya Birla Nuvo (demerged into Grasim)
    "ALSTOMT&D":   "NIFTY ENERGY",             # Alstom T&D (now GE Power India)
    "AMTEKAUTO":   "NIFTY AUTO",               # Amtek Auto (bankrupt)
    "BHARTIHEXA":  "NIFTY IT",                 # Bharti Hexacom
    "COX&KINGS":   "NIFTY CONSUMPTION",        # travel/hospitality (fraud, delisted)
    "DHANI":       "NIFTY FIN SERVICE",        # ex-Indiabulls Ventures (delisted)
    "DHFL":        "NIFTY FIN SERVICE",        # Dewan Housing (bankrupt)
    "FCONSUMER":   "NIFTY CONSUMPTION",        # Future Consumer (bankrupt)
    "FRETAIL":     "NIFTY CONSUMPTION",        # Future Retail (bankrupt)
    "GDL":         "NIFTY INFRA",              # Gateway Distriparks (privatized)
    "GSKCONS":     "NIFTY FMCG",               # GSK Consumer (merged HUL)
    "GUJFLUORO":   "NIFTY METAL",              # Gujarat Fluorochem (chemicals)
    "HDIL":        "NIFTY REALTY",             # Housing Development & Infrastructure
    "HEXAWARE":    "NIFTY IT",                 # Hexaware (delisted by Carlyle)
    "IBREALEST":   "NIFTY REALTY",             # Indiabulls Real Estate
    "IBULHSGFIN":  "NIFTY FIN SERVICE",        # Indiabulls Housing Finance
    "IBVENTURES":  "NIFTY FIN SERVICE",        # Indiabulls Ventures (→DHANI)
    "INOXWIND":    "NIFTY ENERGY",             # Inox Wind
    "ISEC":        "NIFTY FIN SERVICE",        # ICICI Securities (delisted)
    "JETAIRWAYS":  "NIFTY CONSUMPTION",        # Jet Airways (bankrupt)
    "JISLJALEQS":  "NIFTY METAL",              # Jain Irrigation (chemicals)
    "JPASSOCIAT":  "NIFTY INFRA",              # Jaiprakash Associates
    "JPPOWER":     "NIFTY ENERGY",             # Jaiprakash Power
    "JUSTDIAL":    "NIFTY IT",                 # local search
    "KSCL":        "NIFTY FMCG",               # Kaveri Seed
    "MMTC":        "NIFTY METAL",              # MMTC (commodity trader)
    "PCJEWELLER":  "NIFTY CONSUMER DURABLES",  # PC Jeweller
    "PIPAVAVDOC":  "NIFTY INFRA",              # Pipavav Defence (acquired)
    "RAYMOND":     "NIFTY CONSUMPTION",        # Raymond (apparel)
    "RCOM":        "NIFTY CONSUMPTION",        # Reliance Communications (bankrupt)
    "RELCAPITAL":  "NIFTY FIN SERVICE",        # Reliance Capital (bankrupt)
    "RELINFRA":    "NIFTY INFRA",              # Reliance Infrastructure
    "RPOWER":      "NIFTY ENERGY",             # Reliance Power
    "SADBHAV":     "NIFTY INFRA",              # Sadbhav Engineering
    "SINTEX":      "NIFTY METAL",              # Sintex Industries (bankrupt)
    "SUNTV":       "NIFTY MEDIA",              # Sun TV
    "TATAMTRDVR":  "NIFTY AUTO",               # Tata Motors DVR (delisted)
    "TV18BRDCST":  "NIFTY MEDIA",              # TV18 Broadcast (merged Network18)
    "UNITECH":     "NIFTY REALTY",             # Unitech (bankrupt)
    "VAKRANGEE":   "NIFTY IT",                 # Vakrangee
    "VIDEOIND":    "NIFTY CONSUMER DURABLES",  # Videocon
    "ZEEL":        "NIFTY MEDIA",              # Zee Entertainment

    # Previously "OTHER" in N200 — banking:
    "BANDHANBNK":  "NIFTY PVT BANK",           # Bandhan Bank
    "CUB":         "NIFTY PVT BANK",           # City Union Bank
    "DCBBANK":     "NIFTY PVT BANK",           # DCB Bank
    "IDBI":        "NIFTY PSU BANK",           # IDBI Bank (LIC-controlled)
    "IFCI":        "NIFTY FIN SERVICE",        # IFCI Ltd (DFI)
    "J&KBANK":     "NIFTY PVT BANK",           # Jammu & Kashmir Bank
    "KARURVYSYA":  "NIFTY PVT BANK",           # Karur Vysya Bank
    "KTKBANK":     "NIFTY PVT BANK",           # Karnataka Bank
    "RBLBANK":     "NIFTY PVT BANK",           # RBL Bank
    "SOUTHBANK":   "NIFTY PVT BANK",           # South Indian Bank

    # Previously "OTHER" in N200 — pharma / healthcare:
    "AJANTPHARM":  "NIFTY HEALTHCARE",         # Ajanta Pharma
    "APLLTD":      "NIFTY HEALTHCARE",         # Alembic Pharma
    "GLAXO":       "NIFTY HEALTHCARE",         # GSK Pharma India
    "LALPATHLAB":  "NIFTY HEALTHCARE",         # Dr Lal PathLabs
    "METROPOLIS":  "NIFTY HEALTHCARE",         # Metropolis Healthcare
    "NATCOPHARM":  "NIFTY HEALTHCARE",         # Natco Pharma
    "PFIZER":      "NIFTY HEALTHCARE",         # Pfizer India
    "SANOFI":      "NIFTY HEALTHCARE",         # Sanofi India
    "SPARC":       "NIFTY HEALTHCARE",         # Sun Pharma Advanced Research
    "WOCKPHARMA":  "NIFTY HEALTHCARE",         # Wockhardt

    # Previously "OTHER" in N200 — IT / tech:
    "DELHIVERY":   "NIFTY INFRA",              # Delhivery (logistics tech)
    "INDIAMART":   "NIFTY IT",                 # IndiaMART InterMESH
    "KPIT":        "NIFTY IT",                 # KPIT Technologies (old ticker)
    "LTI":         "NIFTY IT",                 # L&T Infotech (→LTIM)
    "LTIM":        "NIFTY IT",                 # LTIMindtree
    "LTTS":        "NIFTY IT",                 # L&T Technology Services
    "MINDTREE":    "NIFTY IT",                 # Mindtree (→LTI→LTIM)
    "REDINGTON":   "NIFTY IT",                 # Redington (IT distribution)
    "STRTECH":     "NIFTY IT",                 # Strides Tech
    "TATATECH":    "NIFTY IT",                 # Tata Technologies

    # Previously "OTHER" in N200 — cement / construction materials:
    "ACC":         "NIFTY INDIA MFG",          # ACC Ltd (cement)
    "DALBHARAT":   "NIFTY INDIA MFG",          # Dalmia Bharat (cement)
    "DALMIABHA":   "NIFTY INDIA MFG",          # Dalmia Bharat Holdings
    "INDIACEM":    "NIFTY INDIA MFG",          # India Cements
    "JKLAKSHMI":   "NIFTY INDIA MFG",          # JK Lakshmi Cement
    "RAMCOCEM":    "NIFTY INDIA MFG",          # Ramco Cements

    # Previously "OTHER" in N200 — chemicals / materials:
    "CLEAN":       "NIFTY INDIA MFG",          # Clean Science & Technology
    "DEEPAKNTR":   "NIFTY INDIA MFG",          # Deepak Nitrite (chemicals)
    "FLUOROCHEM":  "NIFTY METAL",              # Gujarat Fluorochem (renamed GUJFLUORO)
    "GRAPHITE":    "NIFTY METAL",              # Graphite India
    "HEG":         "NIFTY METAL",              # HEG Ltd (graphite electrodes)
    "TATACHEM":    "NIFTY INDIA MFG",          # Tata Chemicals

    # Previously "OTHER" in N200 — auto / components:
    "AMARAJABAT":  "NIFTY AUTO",               # Amara Raja Batteries
    "APOLLOTYRE":  "NIFTY AUTO",               # Apollo Tyres
    "ENDURANCE":   "NIFTY AUTO",               # Endurance Technologies
    "MSUMI":       "NIFTY AUTO",               # Motherson Sumi Systems
    "OLAELEC":     "NIFTY AUTO",               # Ola Electric

    # Previously "OTHER" in N200 — FMCG / consumer:
    "AVANTIFEED":  "NIFTY FMCG",               # Avanti Feeds (aquaculture/shrimp)
    "BBTC":        "NIFTY FMCG",               # Bombay Burmah Trading (tea/agri)
    "GODREJAGRO":  "NIFTY FMCG",               # Godrej Agrovet
    "PGHH":        "NIFTY FMCG",               # P&G Hygiene & Health Care
    "RALLIS":      "NIFTY FMCG",               # Rallis India (agrochemicals)

    # Previously "OTHER" in N200 — consumer durables / lifestyle:
    "BERGEPAINT":  "NIFTY CONSUMER DURABLES",  # Berger Paints
    "RAJESHEXPO":  "NIFTY CONSUMER DURABLES",  # Rajesh Exports (gold jewellery mfg)
    "TTKPRESTIG":  "NIFTY CONSUMER DURABLES",  # TTK Prestige (kitchen appliances)
    "VGUARD":      "NIFTY CONSUMER DURABLES",  # V-Guard Industries (electrical)

    # Previously "OTHER" in N200 — consumption / lifestyle:
    "ABFRL":       "NIFTY CONSUMPTION",        # Aditya Birla Fashion & Retail
    "ARVIND":      "NIFTY CONSUMPTION",        # Arvind Ltd (textiles)
    "DEVYANI":     "NIFTY CONSUMPTION",        # Devyani International (QSR)
    "GODREJIND":   "NIFTY CONSUMPTION",        # Godrej Industries (diversified)
    "ITCHOTELS":   "NIFTY CONSUMPTION",        # ITC Hotels
    "JUBILANT":    "NIFTY CONSUMPTION",        # Jubilant FoodWorks / Pharmova
    "TRIDENT":     "NIFTY CONSUMPTION",        # Trident Ltd (home textiles)
    "TTML":        "NIFTY CONSUMPTION",        # Tata Teleservices Maharashtra
    "WELSPUNIND":  "NIFTY CONSUMPTION",        # Welspun India (home textiles)

    # Previously "OTHER" in N200 — financial services:
    "CARERATING":  "NIFTY FIN SERVICE",        # CARE Ratings
    "EDELWEISS":   "NIFTY FIN SERVICE",        # Edelweiss Financial Services
    "GRUH":        "NIFTY FIN SERVICE",        # GRUH Finance (merged into Bandhan)
    "PEL":         "NIFTY FIN SERVICE",        # Piramal Enterprises
    "QUESS":       "NIFTY FIN SERVICE",        # Quess Corp (staffing / BPO)
    "REPCOHOME":   "NIFTY FIN SERVICE",        # Repco Home Finance

    # Previously "OTHER" in N200 — infra / logistics / ports:
    "DBL":         "NIFTY INFRA",              # Dilip Buildcon (road infra)
    "GESHIP":      "NIFTY INFRA",              # Great Eastern Shipping
    "GPPL":        "NIFTY INFRA",              # Gujarat Pipavav Port
    "IRB":         "NIFTY INFRA",              # IRB Infrastructure Developers
    "JSWINFRA":    "NIFTY INFRA",              # JSW Infrastructure
    "NBCC":        "NIFTY INFRA",              # NBCC India (govt construction)
    "NCC":         "NIFTY INFRA",              # NCC Ltd (construction)
    "WABAG":       "NIFTY INFRA",              # VA Tech WABAG (water solutions)

    # Previously "OTHER" in N200 — energy / oil & gas:
    "FACT":        "NIFTY ENERGY",             # Fertilizers & Chemicals Travancore
    "GSPL":        "NIFTY OIL & GAS",          # Gujarat State Petronet (gas pipeline)
    "MRPL":        "NIFTY OIL & GAS",          # Mangalore Refinery & Petrochemicals
    "PTC":         "NIFTY ENERGY",             # PTC India (power trading)

    # Previously "OTHER" in N200 — media / telecom:
    "DISHTV":      "NIFTY MEDIA",              # Dish TV India

    # N500-only additions — banking / finance:
    "AADHARHFC":   "NIFTY FIN SERVICE",        # Aadhaar Housing Finance
    "AAVAS":       "NIFTY FIN SERVICE",        # AAVAS Financiers (housing finance)
    "ANANDRATHI":  "NIFTY FIN SERVICE",        # Anand Rathi Wealth
    "ANGELONE":    "NIFTY FIN SERVICE",        # Angel One (stockbroker)
    "APTUS":       "NIFTY FIN SERVICE",        # Aptus Value Housing Finance
    "ABSLAMC":     "NIFTY FIN SERVICE",        # Aditya Birla Sun Life AMC
    "CREDITACC":   "NIFTY FIN SERVICE",        # Credit Access Grameen (MFI)
    "EPIGRAL":     "NIFTY FIN SERVICE",        # Epigral (ex-Olin) — chemicals actually
    "IIFL":        "NIFTY FIN SERVICE",        # IIFL Finance
    "IIFLTRD":     "NIFTY FIN SERVICE",        # IIFL Securities (trading)
    "IIFLSEC":     "NIFTY FIN SERVICE",        # IIFL Securities
    "JINDALSAW":   "NIFTY METAL",              # Jindal SAW (steel pipes)
    "MFSL":        "NIFTY FIN SERVICE",        # Max Financial Services
    "MOTILALOFS":  "NIFTY FIN SERVICE",        # Motilal Oswal Financial Services
    "NUVAMA":      "NIFTY FIN SERVICE",        # Nuvama Wealth Management
    "PAISALO":     "NIFTY FIN SERVICE",        # Paisalo Digital (NBFC)
    "POONAWALLA":  "NIFTY FIN SERVICE",        # Poonawalla Fincorp
    "RATEGAIN":    "NIFTY IT",                 # RateGain Travel Technologies
    "SAHYADRI":    "NIFTY HEALTHCARE",         # Sahyadri Hospitals (private)
    "SAMMAANCAP":  "NIFTY FIN SERVICE",        # Sammaan Capital (ex-Indiabulls HFL)
    "SBICARDS":    "NIFTY FIN SERVICE",        # SBI Cards & Payment Services
    "SPANDANA":    "NIFTY FIN SERVICE",        # Spandana Sphoorty (MFI)
    "UTIAMC":      "NIFTY FIN SERVICE",        # UTI AMC

    # N500-only additions — IT / tech:
    "AFFLE":       "NIFTY IT",                 # Affle India (digital advertising tech)
    "ALKYLAMINE":  "NIFTY INDIA MFG",          # Alkyl Amines Chemicals
    "BSOFT":       "NIFTY IT",                 # BIRLASOFT
    "CAMS":        "NIFTY FIN SERVICE",        # CAMS (mutual fund registrar)
    "CYIENT":      "NIFTY IT",                 # Cyient Ltd (engineering services)
    "DATAMATICS":  "NIFTY IT",                 # Datamatics Global Services
    "INTELLECT":   "NIFTY IT",                 # Intellect Design Arena
    "MASTEK":      "NIFTY IT",                 # Mastek Ltd
    "NIITMTS":     "NIFTY IT",                 # NIIT Technologies (→COFORGE)
    "NIITTECH":    "NIFTY IT",                 # NIIT Technologies (old)
    "ONMOBILE":    "NIFTY IT",                 # OnMobile Global
    "RBLBANK":     "NIFTY PVT BANK",           # RBL Bank (duplicate from N200)
    "SEQUENT":     "NIFTY HEALTHCARE",         # SeQuent Scientific (animal health)
    "SIEVERT":     "NIFTY IT",                 # Siemens subsidiary
    "SUBROS":      "NIFTY AUTO",               # Subros Ltd (auto AC systems)
    "TANLA":       "NIFTY IT",                 # Tanla Platforms

    # N500-only additions — auto / EV:
    "OLACABS":     "NIFTY CONSUMPTION",        # Ola Cabs (ride-hailing)
    "STARHEALTH":  "NIFTY HEALTHCARE",         # Star Health Insurance
    "WENDT":       "NIFTY INDIA MFG",          # Wendt India (precision tools)

    # N500-only additions — infra / logistics / ports:
    "ABGSHIP":     "NIFTY INFRA",              # ABG Shipyard (bankrupt)
    "AEGISLOG":    "NIFTY INFRA",              # Aegis Logistics
    "AEGISVOPAK":  "NIFTY INFRA",              # Aegis VOPAK Terminals
    "AFCONS":      "NIFTY INFRA",              # Afcons Infrastructure
    "ALLCARGO":    "NIFTY INFRA",              # Allcargo Logistics
    "ANANTRAJ":    "NIFTY REALTY",             # Anant Raj (real estate)
    "APARINDS":    "NIFTY ENERGY",             # Apar Industries (power cables/conductors)
    "ASHOKA":      "NIFTY INFRA",              # Ashoka Buildcon
    "GPPL":        "NIFTY INFRA",              # Gujarat Pipavav Port (duplicate)

    # N500-only additions — pharma / healthcare:
    "AARTIDRUGS":  "NIFTY HEALTHCARE",         # Aarti Drugs
    "ADVENZYMES":  "NIFTY HEALTHCARE",         # Advanced Enzyme Technologies
    "AGARWALEYE":  "NIFTY HEALTHCARE",         # Dr Agarwal's Eye Hospital
    "AKUMS":       "NIFTY HEALTHCARE",         # Akums Drugs & Pharmaceuticals
    "ALEMBICLTD":  "NIFTY HEALTHCARE",         # Alembic Ltd (pharma)
    "ALIVUS":      "NIFTY HEALTHCARE",         # Alivus Life Sciences
    "ASTERDM":     "NIFTY HEALTHCARE",         # Aster DM Healthcare
    "ASTRAZEN":    "NIFTY HEALTHCARE",         # AstraZeneca India
    "WINDLAS":     "NIFTY HEALTHCARE",         # Windlas Biotech

    # N500-only additions — energy / chemicals:
    "ABAN":        "NIFTY ENERGY",             # Aban Offshore (oil drilling)
    "ABREL":       "NIFTY ENERGY",             # Aditya Birla Renewables
    "ACMESOLAR":   "NIFTY ENERGY",             # Acme Solar Holdings
    "AETHER":      "NIFTY INDIA MFG",          # Aether Industries (specialty chemicals)
    "AARTIIND":    "NIFTY INDIA MFG",          # Aarti Industries (chemicals)
    "ACE":         "NIFTY INDIA MFG",          # Action Construction Equipment
    "ADLABS":      "NIFTY MEDIA",              # Adlabs Entertainment

    # N500-only additions — consumption / textiles / FMCG:
    "ALOKINDS":    "NIFTY CONSUMPTION",        # Alok Industries (textiles)
    "ALOKTEXT":    "NIFTY CONSUMPTION",        # Alok Textiles (old)
    "AMBER":       "NIFTY CONSUMER DURABLES",  # Amber Enterprises (HVAC/cooling)
    "ATFL":        "NIFTY FMCG",               # Agro Tech Foods (ConAgra India)

    # N500-only — banking / financial services:
    "ANURAS":       "NIFTY FIN SERVICE",        # Anuras (financial services)
    "CORPBANK":     "NIFTY PSU BANK",           # Corporation Bank (merged → Union Bank)
    "CSBBANK":      "NIFTY PVT BANK",           # CSB Bank (Catholic Syrian Bank)
    "DHANBANK":     "NIFTY PVT BANK",           # Dhanalakshmi Bank
    "EQUITASBNK":   "NIFTY PVT BANK",           # Equitas Small Finance Bank
    "FINANTECH":    "NIFTY FIN SERVICE",        # Finan Tech (NBFC)
    "GEOJITBNPP":   "NIFTY FIN SERVICE",        # Geojit BNP Paribas (stockbroker)
    "GLS":          "NIFTY FIN SERVICE",        # GLS (NBFC / housing finance)
    "ICRA":         "NIFTY FIN SERVICE",        # ICRA Ltd (credit rating)
    "JBFIND":       "NIFTY FIN SERVICE",        # JB Financial
    "JSWHL":        "NIFTY FIN SERVICE",        # JSW Holdings (investment holding)
    "MYSOREBANK":   "NIFTY PSU BANK",           # Mysore Bank (merged → Canara Bank)
    "PFS":          "NIFTY FIN SERVICE",        # PTC India Financial Services
    "RELIGARE":     "NIFTY FIN SERVICE",        # Religare Enterprises
    "RHFL":         "NIFTY FIN SERVICE",        # Reliance Home Finance (bankrupt)
    "RBA":          "NIFTY FIN SERVICE",        # RBA Financial Services
    "SBBJ":         "NIFTY PSU BANK",           # SB of Bikaner & Jaipur (merged → SBI)
    "SBT":          "NIFTY PSU BANK",           # State Bank of Travancore (merged → SBI)
    "SFL":          "NIFTY CONSUMER DURABLES",  # Sheela Foam Ltd (mattresses/foam)
    "SIS":          "NIFTY FIN SERVICE",        # SIS Ltd (security/staffing services)
    "TMB":          "NIFTY PVT BANK",           # Tamilnad Mercantile Bank
    "UJJIVAN":      "NIFTY FIN SERVICE",        # Ujjivan Financial Services
    "UJJIVANSFB":   "NIFTY PVT BANK",           # Ujjivan Small Finance Bank
    "UNITEDBNK":    "NIFTY PSU BANK",           # United Bank of India (merged → PNB)
    "VIJAYA":       "NIFTY HEALTHCARE",         # Vijaya Diagnostic Centre (listed 2021)

    # N500-only — IT / tech:
    "8KMILES":      "NIFTY IT",                 # 8K Miles Software Services
    "BCG":          "NIFTY IT",                 # Brightcom Group (digital advertising)
    "CARTRADE":     "NIFTY IT",                 # CarTrade Tech (used car marketplace)
    "ECLERX":       "NIFTY IT",                 # eClerx Services (KPO)
    "FSL":          "NIFTY IT",                 # Firstsource Solutions Ltd
    "GEOMETRIC":    "NIFTY IT",                 # Geometric Ltd (CAD/PLM, merged → HCL)
    "HAPPSTMNDS":   "NIFTY IT",                 # Happiest Minds Technologies
    "HCL-INSYS":    "NIFTY IT",                 # HCL Infosystems
    "HFCL":         "NIFTY IT",                 # HFCL Ltd (fiber optic/telecom)
    "INFIBEAM":     "NIFTY IT",                 # Infibeam Avenues (e-commerce/payments)
    "ITI":          "NIFTY IT",                 # ITI Ltd (telecom equipment PSU)
    "LATENTVIEW":   "NIFTY IT",                 # LatentView Analytics
    "MAPMYINDIA":   "NIFTY IT",                 # MapMyIndia (C.E. Info Systems)
    "NETWEB":       "NIFTY IT",                 # Netweb Technologies
    "NEWGEN":       "NIFTY IT",                 # Newgen Software
    "NIITLTD":      "NIFTY IT",                 # NIIT Ltd (IT training/education)
    "RAILTEL":      "NIFTY IT",                 # RailTel Corporation (telecom infra)
    "RAMCOSYS":     "NIFTY IT",                 # Ramco Systems (ERP software)
    "ROLTA":        "NIFTY IT",                 # Rolta India (bankrupt)
    "ROUTE":        "NIFTY IT",                 # Route Mobile (CPaaS/messaging)
    "SITINET":      "NIFTY MEDIA",              # SITI Networks (cable TV)
    "SONATSOFTW":   "NIFTY IT",                 # Sonata Software
    "STLTECH":      "NIFTY IT",                 # STL Tech (optical fiber/cables)
    "TBOTEK":       "NIFTY IT",                 # TBO Tek (B2B travel tech)
    "TEJASNET":     "NIFTY IT",                 # Tejas Networks (optical networking)
    "ZENSARTECH":   "NIFTY IT",                 # Zensar Technologies

    # N500-only — pharma / healthcare:
    "BLUEJET":      "NIFTY HEALTHCARE",         # Blue Jet Healthcare (pharma intermediates)
    "CONCORDBIO":   "NIFTY HEALTHCARE",         # Concord Biotech
    "EMCURE":       "NIFTY HEALTHCARE",         # Emcure Pharmaceuticals
    "ERIS":         "NIFTY HEALTHCARE",         # Eris Lifesciences
    "FDC":          "NIFTY HEALTHCARE",         # FDC Ltd
    "GRANULES":     "NIFTY HEALTHCARE",         # Granules India
    "HIKAL":        "NIFTY HEALTHCARE",         # Hikal Ltd (pharma/crop chemicals)
    "INDOCO":       "NIFTY HEALTHCARE",         # Indoco Remedies
    "IOLCP":        "NIFTY HEALTHCARE",         # IOL Chemicals & Pharmaceuticals
    "JBCHEPHARM":   "NIFTY HEALTHCARE",         # JB Chemicals & Pharmaceuticals
    "JUBLPHARMA":   "NIFTY HEALTHCARE",         # Jubilant Pharmova
    "KIMS":         "NIFTY HEALTHCARE",         # Krishna Institute of Medical Sciences
    "MARKSANS":     "NIFTY HEALTHCARE",         # Marksans Pharma
    "MEDANTA":      "NIFTY HEALTHCARE",         # Global Health (Medanta hospitals)
    "MEDPLUS":      "NIFTY HEALTHCARE",         # Medplus Health Services
    "NH":           "NIFTY HEALTHCARE",         # Narayana Hrudayalaya
    "PGHL":         "NIFTY HEALTHCARE",         # Procter & Gamble Health Ltd
    "POLYMED":      "NIFTY HEALTHCARE",         # Poly Medicure
    "RAINBOW":      "NIFTY HEALTHCARE",         # Rainbow Children's Medicare
    "SAGILITY":     "NIFTY HEALTHCARE",         # Sagility India (US healthcare BPO)
    "SAILIFE":      "NIFTY HEALTHCARE",         # SAI Life Sciences
    "SHILPAMED":    "NIFTY HEALTHCARE",         # Shilpa Medicare
    "SOLARA":       "NIFTY HEALTHCARE",         # Solara Active Pharma Sciences
    "STAR":         "NIFTY HEALTHCARE",         # Strides Pharma Science (old ticker)
    "THYROCARE":    "NIFTY HEALTHCARE",         # Thyrocare Technologies
    "UNICHEMLAB":   "NIFTY HEALTHCARE",         # Unichem Laboratories

    # N500-only — auto / components:
    "ARE&M":        "NIFTY AUTO",               # Amara Raja Energy & Mobility (→ AMARAJABAT)
    "ASAHIINDIA":   "NIFTY AUTO",               # Asahi India Glass (automotive glass)
    "AUTOAXLES":    "NIFTY AUTO",               # Auto Axles India
    "CEATLTD":      "NIFTY AUTO",               # CEAT (tyres)
    "CIEINDIA":     "NIFTY AUTO",               # CIE Automotive India
    "CRAFTSMAN":    "NIFTY AUTO",               # Craftsman Automation
    "FMGOETZE":     "NIFTY AUTO",               # Federal-Mogul Goetze India (pistons)
    "GABRIEL":      "NIFTY AUTO",               # Gabriel India (shocks/suspension)
    "GREAVESCOT":   "NIFTY AUTO",               # Greaves Cotton (engines/EVs)
    "IGARASHI":     "NIFTY AUTO",               # Igarashi Motors India
    "JAMNAAUTO":    "NIFTY AUTO",               # Jamna Auto Industries (leaf springs)
    "JBMA":         "NIFTY AUTO",               # JBM Auto
    "JCHAC":        "NIFTY CONSUMER DURABLES",  # Johnson Controls-Hitachi AC
    "JKTYRE":       "NIFTY AUTO",               # JK Tyre & Industries
    "JMTAUTOLTD":   "NIFTY AUTO",               # JMT Auto
    "MAHSCOOTER":   "NIFTY AUTO",               # Mahindra Scooters
    "MINDACORP":    "NIFTY AUTO",               # Minda Corporation (auto components)
    "MUNJALSHOW":   "NIFTY AUTO",               # Munjal Showa (auto shocks)
    "OLECTRA":      "NIFTY AUTO",               # Olectra Greentech (EV buses)
    "RICOAUTO":     "NIFTY AUTO",               # Rico Auto Industries
    "RKFORGE":      "NIFTY AUTO",               # Ramkrishna Forgings
    "SMLISUZU":     "NIFTY AUTO",               # SML Isuzu (LCVs)
    "SUNDRMFAST":   "NIFTY AUTO",               # Sundram Fasteners (auto components)
    "SUPRAJIT":     "NIFTY AUTO",               # Suprajit Engineering
    "SWARAJENG":    "NIFTY AUTO",               # Swaraj Engines (tractors)
    "TVSSCS":       "NIFTY AUTO",               # TVS Supply Chain Solutions
    "TVSSRICHAK":   "NIFTY AUTO",               # TVS Srichakra
    "VARROC":       "NIFTY AUTO",               # Varroc Engineering
    "ZFCVINDIA":    "NIFTY AUTO",               # ZF Commercial Vehicle Control Systems

    # N500-only — metals / steel / materials:
    "BHUSANSTL":    "NIFTY METAL",              # Bhushan Steel (bankrupt → Tata Steel)
    "ELECTCAST":    "NIFTY METAL",              # Electrosteel Castings
    "ESSDEE":       "NIFTY METAL",              # Essdee Aluminium
    "GALLANTT":     "NIFTY METAL",              # Gallantt Ispat
    "GMDCLTD":      "NIFTY METAL",              # GMDC (Gujarat Mineral Development Corp)
    "GPIL":         "NIFTY METAL",              # Godawari Power & Ispat
    "JAIBALAJI":    "NIFTY METAL",              # Jai Balaji Industries
    "JINDALPOLY":   "NIFTY METAL",              # Jindal Poly Films (polyester/steel)
    "KIOCL":        "NIFTY METAL",              # KIOCL (iron ore)
    "MAHSEAMLES":   "NIFTY METAL",              # Mahindra Seamless (steel pipes)
    "MOIL":         "NIFTY METAL",              # MOIL (manganese ore)
    "NSLNISP":      "NIFTY METAL",              # NSL Neyveli Ispat / NINL
    "ORISSAMINE":   "NIFTY METAL",              # Odisha Minerals
    "RAIN":         "NIFTY METAL",              # Rain Industries (carbon/chemicals)
    "RATNAMANI":    "NIFTY METAL",              # Ratnamani Metals & Tubes
    "RHIM":         "NIFTY METAL",              # RHI Magnesita India (refractory)
    "SHYAMMETL":    "NIFTY METAL",              # Shyam Metalics
    "TATASPONGE":   "NIFTY METAL",              # Tata Sponge Iron (→ Tata Steel LP)
    "TATASTLLP":    "NIFTY METAL",              # Tata Steel Long Products
    "USHAMART":     "NIFTY METAL",              # Usha Martin (wire rope/steel)

    # N500-only — cement / construction materials:
    "BIRLACORPN":   "NIFTY INDIA MFG",          # Birla Corporation (cement)
    "HEIDELBERG":   "NIFTY INDIA MFG",          # HeidelbergCement India
    "JKCEMENT":     "NIFTY INDIA MFG",          # JK Cement
    "JSWCEMENT":    "NIFTY INDIA MFG",          # JSW Cement
    "NUVOCO":       "NIFTY INDIA MFG",          # Nuvoco Vistas (cement)
    "ORIENTCEM":    "NIFTY INDIA MFG",          # Orient Cement
    "STARCEMENT":   "NIFTY INDIA MFG",          # Star Cement

    # N500-only — chemicals / specialty materials:
    "ATUL":         "NIFTY INDIA MFG",          # Atul Ltd (specialty chemicals)
    "BALAMINES":    "NIFTY INDIA MFG",          # Balaji Amines
    "BASF":         "NIFTY INDIA MFG",          # BASF India (specialty chemicals)
    "CAMLINFINE":   "NIFTY INDIA MFG",          # Camlin Fine Sciences
    "CARBORUNIV":   "NIFTY INDIA MFG",          # Carborundum Universal (abrasives)
    "CHEMPLASTS":   "NIFTY INDIA MFG",          # Chemplast Sanmar
    "DCMSHRIRAM":   "NIFTY INDIA MFG",          # DCM Shriram (chemicals/fertilizers)
    "DEEPAKFERT":   "NIFTY INDIA MFG",          # Deepak Fertilisers
    "FINEORG":      "NIFTY INDIA MFG",          # Fine Organics
    "GALAXYSURF":   "NIFTY INDIA MFG",          # Galaxy Surfactants
    "GNFC":         "NIFTY INDIA MFG",          # Gujarat Narmada Valley Fertilizers
    "GSFC":         "NIFTY INDIA MFG",          # Gujarat State Fertilizers & Chemicals
    "GUJALKALI":    "NIFTY INDIA MFG",          # Gujarat Alkalies & Chemicals
    "INEOSSTYRO":   "NIFTY INDIA MFG",          # INEOS Styrolution India (plastics)
    "JUBLINGREA":   "NIFTY INDIA MFG",          # Jubilant Ingrevia (specialty chemicals)
    "LXCHEM":       "NIFTY INDIA MFG",          # Laxmi Organic Industries
    "NOCIL":        "NIFTY INDIA MFG",          # NOCIL (rubber chemicals)
    "PCBL":         "NIFTY INDIA MFG",          # PCBL (carbon black)
    "ROSSARI":      "NIFTY INDIA MFG",          # Rossari Biotech
    "SUDARSCHEM":   "NIFTY INDIA MFG",          # Sudarshan Chemical Industries
    "SUMICHEM":     "NIFTY INDIA MFG",          # Sumitomo Chemical India
    "VINATIORGA":   "NIFTY INDIA MFG",          # Vinati Organics

    # N500-only — FMCG / agri / consumer staples:
    "BAJAJHIND":    "NIFTY FMCG",               # Bajaj Hindusthan Sugar
    "BAJAJCON":     "NIFTY FMCG",               # Bajaj Consumer Care (hair oil)
    "BALRAMCHIN":   "NIFTY FMCG",               # Balrampur Chini Mills (sugar)
    "BAYERCROP":    "NIFTY FMCG",               # Bayer CropScience (agro)
    "BIKAJI":       "NIFTY FMCG",               # Bikaji Foods (snacks)
    "CCL":          "NIFTY FMCG",               # CCL Products (coffee)
    "CHAMBLFERT":   "NIFTY INDIA MFG",          # Chambal Fertilisers & Chemicals
    "DHANUKA":      "NIFTY FMCG",               # Dhanuka Agritech (agro chemicals)
    "EIDPARRY":     "NIFTY FMCG",               # EID Parry (sugar/agri)
    "GAEL":         "NIFTY FMCG",               # Gujarat Ambuja Exports (agri processing)
    "GILLETTE":     "NIFTY FMCG",               # Gillette India
    "HATSUN":       "NIFTY FMCG",               # Hatsun Agro Products (dairy)
    "HERITGFOOD":   "NIFTY FMCG",               # Heritage Foods (dairy)
    "KRBL":         "NIFTY FMCG",               # KRBL (basmati rice)
    "KWALITY":      "NIFTY FMCG",               # Kwality Dairy
    "MANPASAND":    "NIFTY FMCG",               # Manpasand Beverages (fraud/delisted)
    "MCLEODRUSS":   "NIFTY FMCG",               # McLeod Russel (tea)
    "MONSANTO":     "NIFTY FMCG",               # Monsanto India (seeds, now Bayer)
    "NFL":          "NIFTY INDIA MFG",          # National Fertilizers Ltd
    "RCF":          "NIFTY INDIA MFG",          # Rashtriya Chemicals & Fertilizers
    "RENUKA":       "NIFTY FMCG",               # Shree Renuka Sugars
    "RUCHISOYA":    "NIFTY FMCG",               # Ruchi Soya (edible oils, → Patanjali)
    "SHARDACROP":   "NIFTY FMCG",               # Sharda Cropchem (agrochemicals)
    "TASTYBITE":    "NIFTY FMCG",               # Tasty Bite Eatables
    "TATACOFFEE":   "NIFTY FMCG",               # Tata Coffee (→ Tata Consumer)
    "TRIVENI":      "NIFTY FMCG",               # Triveni Engineering (sugar + turbines)
    "ZYDUSWELL":    "NIFTY FMCG",               # Zydus Wellness

    # N500-only — consumer durables / home products:
    "CAMPUS":       "NIFTY CONSUMER DURABLES",  # Campus Activewear (sports shoes)
    "CELLO":        "NIFTY CONSUMER DURABLES",  # Cello World (plastic/glass products)
    "CERA":         "NIFTY CONSUMER DURABLES",  # Cera Sanitaryware
    "CENTURYPLY":   "NIFTY CONSUMER DURABLES",  # Century Plyboards
    "DOMS":         "NIFTY CONSUMER DURABLES",  # DOMS Industries (stationery)
    "EVEREADY":     "NIFTY CONSUMER DURABLES",  # Eveready Industries (batteries)
    "GREENPANEL":   "NIFTY CONSUMER DURABLES",  # Greenpanel Industries (MDF boards)
    "GREENPLY":     "NIFTY CONSUMER DURABLES",  # Greenply Industries (plywood)
    "IFBIND":       "NIFTY CONSUMER DURABLES",  # IFB Industries (home appliances)
    "JSWDULUX":     "NIFTY CONSUMER DURABLES",  # JSW Paints / Dulux JV
    "KANSAINER":    "NIFTY CONSUMER DURABLES",  # Kansai Nerolac Paints
    "LAOPALA":      "NIFTY CONSUMER DURABLES",  # La Opala RG (glass tableware)
    "LUXIND":       "NIFTY CONSUMER DURABLES",  # Lux Industries (innerwear)
    "NILKAMAL":     "NIFTY CONSUMER DURABLES",  # Nilkamal (plastic furniture)
    "ORIENTELEC":   "NIFTY CONSUMER DURABLES",  # Orient Electric (fans/appliances)
    "RELAXO":       "NIFTY CONSUMER DURABLES",  # Relaxo Footwears
    "SAFARI":       "NIFTY CONSUMER DURABLES",  # Safari Industries (luggage)
    "SOMANYCERA":   "NIFTY CONSUMER DURABLES",  # Somany Ceramics
    "SYMPHONY":     "NIFTY CONSUMER DURABLES",  # Symphony Ltd (evaporative coolers)
    "VAIBHAVGBL":   "NIFTY CONSUMER DURABLES",  # Vaibhav Global (e-commerce jewellery)
    "VIPIND":       "NIFTY CONSUMER DURABLES",  # VIP Industries (luggage)

    # N500-only — consumption / retail / hospitality / fashion:
    "BOMDYEING":    "NIFTY CONSUMPTION",        # Bombay Dyeing (textiles)
    "CHALET":       "NIFTY CONSUMPTION",        # Chalet Hotels
    "DELTACORP":    "NIFTY CONSUMPTION",        # Delta Corp (gaming/casinos)
    "EASEMYTRIP":   "NIFTY CONSUMPTION",        # Easy Trip Planners (travel tech)
    "EIHOTEL":      "NIFTY CONSUMPTION",        # EIH Ltd (Oberoi Hotels)
    "FIRSTCRY":     "NIFTY CONSUMPTION",        # FirstCry (baby products retail)
    "FLFL":         "NIFTY CONSUMPTION",        # Future Lifestyle Fashions
    "GOCOLORS":     "NIFTY CONSUMPTION",        # Go Colors (fashion retail)
    "HONASA":       "NIFTY CONSUMPTION",        # Honasa Consumer (Mamaearth)
    "KITEX":        "NIFTY CONSUMPTION",        # Kitex Garments (children's clothing)
    "LEMONTREE":    "NIFTY CONSUMPTION",        # Lemon Tree Hotels
    "MANYAVAR":     "NIFTY CONSUMPTION",        # Vedant Fashions (Manyavar)
    "MEESHO":       "NIFTY CONSUMPTION",        # Meesho (social commerce)
    "METROBRAND":   "NIFTY CONSUMPTION",        # Metro Brands (footwear retail)
    "MHRIL":        "NIFTY CONSUMPTION",        # Mahindra Holidays & Resorts
    "NESCO":        "NIFTY CONSUMPTION",        # Nesco Ltd (exhibition centre)
    "RUPA":         "NIFTY CONSUMPTION",        # Rupa & Company (innerwear)
    "SHOPERSTOP":   "NIFTY CONSUMPTION",        # Shoppers Stop (retail)
    "SPICEJET":     "NIFTY CONSUMPTION",        # SpiceJet (aviation)
    "SWANCORP":     "NIFTY ENERGY",             # Swan Energy (LNG infrastructure)
    "TCNSBRANDS":   "NIFTY CONSUMPTION",        # TCNS Clothing (W, Aurelia brands)
    "VENKEYS":      "NIFTY FMCG",               # Venky's India (poultry/chicken)
    "VMART":        "NIFTY CONSUMPTION",        # V-Mart Retail
    "VTL":          "NIFTY CONSUMPTION",        # Vardhman Textiles Ltd
    "WELSPUNLIV":   "NIFTY CONSUMPTION",        # Welspun Living (home textiles)
    "WESTLIFE":     "NIFTY CONSUMPTION",        # Westlife Foodworld (McDonald's India)
    "WONDERLA":     "NIFTY CONSUMPTION",        # Wonderla Holidays (amusement parks)

    # N500-only — realty:
    "ASHIANA":      "NIFTY REALTY",             # Ashiana Housing
    "DBREALTY":     "NIFTY REALTY",             # D.B. Realty
    "HEMIPROP":     "NIFTY REALTY",             # Hemi Properties
    "KOLTEPATIL":   "NIFTY REALTY",             # Kolte-Patil Developers
    "MAHLIFE":      "NIFTY REALTY",             # Mahindra Lifespace Developers
    "OMAXE":        "NIFTY REALTY",             # Omaxe (real estate)
    "PARSVNATH":    "NIFTY REALTY",             # Parsvnath Developers
    "PURVA":        "NIFTY REALTY",             # Puravankara (real estate)
    "RUSTOMJEE":    "NIFTY REALTY",             # Rustomjee / Keystone Realtors
    "SUNTECK":      "NIFTY REALTY",             # Sunteck Realty

    # N500-only — infra / logistics / shipping / engineering:
    "AHLUCONT":     "NIFTY INFRA",              # Ahluwalia Contracts India
    "BLUEDART":     "NIFTY INFRA",              # Blue Dart Express (courier/logistics)
    "DREDGECORP":   "NIFTY INFRA",              # Dredging Corporation of India
    "ENGINERSIN":   "NIFTY INFRA",              # Engineers India Ltd (petroleum EPC)
    "FINCABLES":    "NIFTY ENERGY",             # Finolex Cables (electrical cables)
    "FINPIPE":      "NIFTY INFRA",              # Finolex Industries (PVC pipes)
    "GAMMNINFRA":   "NIFTY INFRA",              # Gammon Infrastructure
    "GATI":         "NIFTY INFRA",              # Gati Ltd (logistics)
    "GRINFRA":      "NIFTY INFRA",              # GMR Infrastructure
    "GVKPIL":       "NIFTY INFRA",              # GVK Power & Infrastructure
    "HCC":          "NIFTY INFRA",              # Hindustan Construction Company
    "IL&FSENGG":    "NIFTY INFRA",              # IL&FS Engineering (bankrupt)
    "IL&FSTRANS":   "NIFTY INFRA",              # IL&FS Transportation (bankrupt)
    "IRCON":        "NIFTY INFRA",              # IRCON International (rail EPC)
    "JKIL":         "NIFTY INFRA",              # JK Infrastructure
    "JPINFRATEC":   "NIFTY INFRA",              # Jaypee Infratech
    "KEC":          "NIFTY INFRA",              # KEC International (power T&D)
    "KNRCON":       "NIFTY INFRA",              # KNR Constructions
    "KPIL":         "NIFTY INFRA",              # Kalpataru Power Transmission
    "MAHLOG":       "NIFTY INFRA",              # Mahindra Logistics
    "NAVKARCORP":   "NIFTY INFRA",              # Navkar Corporation (container logistics)
    "PNCINFRA":     "NIFTY INFRA",              # PNC Infratech
    "PSPPROJECT":   "NIFTY INFRA",              # PSP Projects (civil construction)
    "PUNJLLOYD":    "NIFTY INFRA",              # Punj Lloyd (EPC contractor)
    "RITES":        "NIFTY INFRA",              # RITES Ltd (rail/transport consulting)
    "SADBHIN":      "NIFTY INFRA",              # Sadbhav Infrastructure Project
    "SCI":          "NIFTY INFRA",              # Shipping Corporation of India
    "SREINFRA":     "NIFTY INFRA",              # Sree Infra (construction)
    "TCI":          "NIFTY INFRA",              # Transport Corp of India
    "TCIEXP":       "NIFTY INFRA",              # TCI Express (courier)
    "TITAGARH":     "NIFTY INFRA",              # Titagarh Rail Systems (railcars)
    "VRLLOG":       "NIFTY INFRA",              # VRL Logistics

    # N500-only — energy / power / renewables:
    "BGRENERGY":    "NIFTY ENERGY",             # BGR Energy Systems (power plant EPC)
    "BFUTILITIE":   "NIFTY ENERGY",             # BF Utilities
    "BORORENEW":    "NIFTY ENERGY",             # Borosil Renewables (solar glass)
    "EMMVEE":       "NIFTY ENERGY",             # Emmvee Photovoltaic
    "KSK":          "NIFTY ENERGY",             # KSK Energy Ventures
    "PRAJIND":      "NIFTY ENERGY",             # Praj Industries (biofuels/ethanol)
    "RTNPOWER":     "NIFTY ENERGY",             # RattanIndia Power
    "SCHNEIDER":    "NIFTY ENERGY",             # Schneider Electric Infrastructure
    "SWSOLAR":      "NIFTY ENERGY",             # Sterling & Wilson Renewable Energy
    "TRITURBINE":   "NIFTY ENERGY",             # Triveni Turbine

    # N500-only — media / publishing:
    "DEN":          "NIFTY MEDIA",              # DEN Networks (cable TV)
    "EROSMEDIA":    "NIFTY MEDIA",              # Eros Now / Eros STX
    "JAGRAN":       "NIFTY MEDIA",              # Jagran Prakashan (newspaper)
    "NDTV":         "NIFTY MEDIA",              # NDTV
    "NAVNETEDUL":   "NIFTY MEDIA",              # Navneet Education (publishing)
    "TVTODAY":      "NIFTY MEDIA",              # TV Today Network
    "ZEELEARN":     "NIFTY MEDIA",              # Zee Learn (education media)

    # N500-only — manufacturing / industrial machinery:
    "BALLARPUR":    "NIFTY INDIA MFG",          # Ballarpur Industries (paper)
    "ESABINDIA":    "NIFTY INDIA MFG",          # ESAB India (welding equipment)
    "ELGIEQUIP":    "NIFTY INDIA MFG",          # Elgi Equipments (air compressors)
    "EPL":          "NIFTY INDIA MFG",          # EPL Ltd (packaging tubes)
    "ESSELPACK":    "NIFTY INDIA MFG",          # Essel Propack (packaging)
    "GARFIBRES":    "NIFTY INDIA MFG",          # Garware Technical Fibres
    "GHCL":         "NIFTY INDIA MFG",          # GHCL Ltd (textiles/soda ash)
    "GMMPFAUDLR":   "NIFTY INDIA MFG",          # GMM Pfaudler (chemical reactors)
    "GRINDWELL":    "NIFTY INDIA MFG",          # Grindwell Norton (abrasives)
    "HBLENGINE":    "NIFTY INDIA MFG",          # HBL Engineering (industrial batteries)
    "INGERRAND":    "NIFTY INDIA MFG",          # Ingersoll Rand India
    "INOXINDIA":    "NIFTY INDIA MFG",          # Inox India (cryogenic equipment)
    "JKPAPER":      "NIFTY INDIA MFG",          # JK Paper
    "KSB":          "NIFTY INDIA MFG",          # KSB Ltd (pumps/valves)
    "KSBPUMPS":     "NIFTY INDIA MFG",          # KSB Pumps (old ticker)
    "LAXMIMACH":    "NIFTY INDIA MFG",          # Laxmi Machine Works (textile machinery)
    "POLYPLEX":     "NIFTY INDIA MFG",          # Polyplex Corp (polyester film)
    "PRINCEPIPE":   "NIFTY INDIA MFG",          # Prince Pipes & Fittings
    "RATNAMANI":    "NIFTY METAL",              # Ratnamani Metals & Tubes (duplicate fix)
    "TNPL":         "NIFTY INDIA MFG",          # Tamil Nadu Newsprint & Papers
    "UFLEX":        "NIFTY INDIA MFG",          # UFlex (flexible packaging)
    "VESUVIUS":     "NIFTY INDIA MFG",          # Vesuvius India (refractory materials)

    # N500-only — misc / small appearance counts:
    "ADLABS":       "NIFTY MEDIA",              # Adlabs Entertainment (cinema)
    "BHARAT":       "NIFTY FMCG",               # Bharat (agri/diversified)
    "ENIL":         "NIFTY MEDIA",              # Entertainment Network India (Radio Mirchi)
    "GEOJITBNPP":   "NIFTY FIN SERVICE",        # Geojit BNP Paribas
    "HGS":          "NIFTY IT",                 # Hinduja Global Solutions (BPO)
    "MAHLIFE":      "NIFTY REALTY",             # (duplicate fix)
    "PRVSF":        "NIFTY FIN SERVICE",        # Priviscl / PRVS Finance
    "RRTGLOBAL":    "NIFTY INFRA",              # RRT Global (logistics)
    "TEAMLEASE":    "NIFTY FIN SERVICE",        # TeamLease Services (staffing)

    # N500-only — 88 remaining OTHER (all previously unclassified):
    # Numeric tickers — likely PDF parsing artifacts from reconstitution notices
    "100":          "NIFTY INDIA MFG",          # likely data artifact / page number
    "110":          "NIFTY INDIA MFG",          # likely data artifact / page number

    # Auto / components:
    "BELRISE":      "NIFTY AUTO",               # Belrise Industries (auto body parts)
    "CASTEXTECH":   "NIFTY AUTO",               # Castex Technologies (auto castings)
    "FORCEMOT":     "NIFTY AUTO",               # Force Motors (commercial vehicles/SUVs)
    "HAPPYFORGE":   "NIFTY AUTO",               # Happy Forgings (auto forgings)
    "LGBBROSLTD":   "NIFTY AUTO",               # LG Balakrishnan & Bros (auto chains)
    "SONASTEER":    "NIFTY AUTO",               # Sona Koyo Steering Systems
    "WHEELS":       "NIFTY AUTO",               # Wheels India (auto wheel rims)

    # Banking / financial services:
    "CANHLIFE":     "NIFTY FIN SERVICE",        # Canara HSBC Life Insurance
    "HDBFS":        "NIFTY FIN SERVICE",        # HDB Financial Services (HDFC NBFC)
    "IGIL":         "NIFTY FIN SERVICE",        # IIFL related / financial services
    "NBVENTURES":   "NIFTY FIN SERVICE",        # NB Ventures (NBFC)
    "PINELABS":     "NIFTY FIN SERVICE",        # Pine Labs (fintech / POS payments)
    "PIRAMALFIN":   "NIFTY FIN SERVICE",        # Piramal Finance (NBFC)
    "SEINV":        "NIFTY FIN SERVICE",        # SE Investments (NBFC/MFI)

    # Pharma / healthcare:
    "IKS":          "NIFTY HEALTHCARE",         # IKS Health (US healthcare BPO/RCM)
    "INDGN":        "NIFTY HEALTHCARE",         # Indegene (healthcare tech/CDMO)  — duplicate fix
    "NEULANDLAB":   "NIFTY HEALTHCARE",         # Neuland Laboratories (pharma API)
    "OPTOCIRCUI":   "NIFTY HEALTHCARE",         # Opto Circuits India (medical devices)

    # IT / tech:
    "CPPLUS":       "NIFTY IT",                 # CP Plus (CCTV / surveillance tech)
    "GTLINFRA":     "NIFTY IT",                 # GTL Infrastructure (telecom towers)
    "HEXT":         "NIFTY IT",                 # Hexaware Technologies (re-listed)

    # Energy / oil & gas / renewables:
    "ATHERENERG":   "NIFTY ENERGY",             # Ather Energy (EV / e-scooters)
    "GIPCL":        "NIFTY ENERGY",             # Gujarat Industries Power Company
    "JAINREC":      "NIFTY ENERGY",             # Jain Irrigation Solar (agri/solar)
    "LITL":         "NIFTY ENERGY",             # Lanco Infratech (power, bankrupt)
    "SPLPETRO":     "NIFTY OIL & GAS",          # SPL Petroleum
    "SUPPETRO":     "NIFTY INDIA MFG",          # Supreme Petrochem (petrochemicals)
    "TDPOWERSYS":   "NIFTY ENERGY",             # TD Power Systems (generators)
    "TECHNOE":      "NIFTY ENERGY",             # Techno Electric & Engineering  — duplicate fix

    # Infra / logistics / construction:
    "GLOBOFFS":     "NIFTY INFRA",              # Global Offshore Services
    "IVRCLINFRA":   "NIFTY INFRA",              # IVRCL Infrastructure (bankrupt)
    "ITDCEM":       "NIFTY INFRA",              # ITD Cementation (construction)
    "JWL":          "NIFTY INFRA",              # Jupiter Wagons (railway wagons) — duplicate fix
    "JYOTISTRUC":   "NIFTY INFRA",              # Jyoti Structures (T&D EPC, bankrupt)
    "MBLINFRA":     "NIFTY INFRA",              # MBL Infrastructure (road EPC)
    "MERCATOR":     "NIFTY INFRA",              # Mercator Lines (shipping / coal)
    "NOIDATOLL":    "NIFTY INFRA",              # Noida Toll Bridge
    "PATELENG":     "NIFTY INFRA",              # Patel Engineering (hydro/tunnels)
    "SIMPLEXINF":   "NIFTY INFRA",              # Simplex Infrastructure
    "SUPREMEINF":   "NIFTY INFRA",              # Supreme Infrastructure (road EPC)

    # Metals / mining:
    "GRAVITA":      "NIFTY METAL",              # Gravita India (lead recycling)
    "PENIND":       "NIFTY METAL",              # Pennar Industries (steel products)
    "PRAKASH":      "NIFTY METAL",              # Prakash Industries (steel/energy)
    "SARDAEN":      "NIFTY METAL",              # Sarda Energy & Minerals (ferro alloys)
    "STCINDIA":     "NIFTY METAL",              # State Trading Corp (commodity trader)
    "UTTAMSTL":     "NIFTY METAL",              # Uttam Galva Steels (bankrupt)

    # Manufacturing / industrial / chemicals:
    "AIIL":         "NIFTY INDIA MFG",          # Allied / industrial manufacturing
    "CENTENKA":     "NIFTY INDIA MFG",          # Century Enka (nylon yarn)
    "CLNINDIA":     "NIFTY INDIA MFG",          # CLN India (industrial)
    "FLEXITUFF":    "NIFTY INDIA MFG",          # Flexituff Ventures (FIBC flexible packaging)
    "HMT":          "NIFTY INDIA MFG",          # HMT Ltd (machine tools, PSU)
    "IPAPPM":       "NIFTY INDIA MFG",          # IPA Paper Mills
    "JAICORPLTD":   "NIFTY INDIA MFG",          # Jai Corp Ltd (plastics/textiles) — duplicate fix
    "KCP":          "NIFTY INDIA MFG",          # KCP Ltd (cement/sugar/engineering)
    "KENNAMET":     "NIFTY INDIA MFG",          # Kennametal India (cutting tools)
    "KESORAMIND":   "NIFTY INDIA MFG",          # Kesoram Industries — duplicate fix
    "MAYURUNIQ":    "NIFTY INDIA MFG",          # Mayur Uniquoters (synthetic leather)
    "METALFORGE":   "NIFTY INDIA MFG",          # Metal Forgings (industrial forgings)
    "PARADEEP":     "NIFTY INDIA MFG",          # Paradeep Phosphates (fertilizers)
    "PRIVISCL":     "NIFTY INDIA MFG",          # Privi Speciality Chemicals
    "SHANTIGEAR":   "NIFTY INDIA MFG",          # Shanti Gears (industrial gears)
    "SPTL":         "NIFTY INDIA MFG",          # Sintex Plastics Technology
    "SRIPIPES":     "NIFTY INDIA MFG",          # Sri Pipes (pipe manufacturing)
    "SYRMA":        "NIFTY INDIA MFG",          # Syrma SGS Technology (EMS) — duplicate fix
    "TARIL":        "NIFTY INDIA MFG",          # Taril (engineering/industrial)
    "TEGA":         "NIFTY INDIA MFG",          # Tega Industries (mineral processing equip)
    "VALIANTORG":   "NIFTY INDIA MFG",          # Valiant Organics (specialty chemicals)
    "VESUVIUS":     "NIFTY INDIA MFG",          # Vesuvius India — duplicate fix
    "VIVIDHA":      "NIFTY INDIA MFG",          # Vividh Chemicals

    # FMCG / agri:
    "BINDALAGRO":   "NIFTY FMCG",               # Bindal Agro Chem
    "GULFOILLUB":   "NIFTY OIL & GAS",          # Gulf Oil Lubricants India
    "TIDEWATER":    "NIFTY OIL & GAS",          # Tide Water Oil (lubricants)

    # Consumer durables:
    "ACI":          "NIFTY CONSUMER DURABLES",  # Acrysil India (kitchen sinks/quartz)
    "GITANJALI":    "NIFTY CONSUMER DURABLES",  # Gitanjali Gems (jewellery, bankrupt/fraud)
    "SHRENUJ":      "NIFTY CONSUMER DURABLES",  # Shrenuj & Company (diamonds/jewellery)
    "TBZ":          "NIFTY CONSUMER DURABLES",  # Tribhovandas Bhimji Zaveri (jewellery)

    # Consumption / lifestyle / hotels:
    "ABLBL":        "NIFTY CONSUMPTION",        # ABL BioLogicals / recent listing
    "ABDL":         "NIFTY CONSUMPTION",        # ABD Ltd (recent listing)
    "ACUTAAS":      "NIFTY FIN SERVICE",        # Acuitas Capital (fintech/NBFC)
    "CEMPRO":       "NIFTY INDIA MFG",          # Cempro / cement products
    "HMVL":         "NIFTY MEDIA",              # Hindustan Media Ventures (HT Media)
    "HOTELEELA":    "NIFTY CONSUMPTION",        # Hotel Leelaventure (luxury hotels)
    "JINDWORLD":    "NIFTY CONSUMPTION",        # Jindal Worldwide (textiles)
    "KKCL":         "NIFTY CONSUMPTION",        # Kewal Kiran Clothing (KILLER jeans)
    "MTEDUCARE":    "NIFTY CONSUMPTION",        # MT Educare (tutoring/education)
    "NESCO":        "NIFTY CONSUMPTION",        # Nesco Ltd (exhibition centre) — duplicate fix
    "PWL":          "NIFTY CONSUMPTION",        # PW Ltd (PhysicsWallah EdTech)
    "TENNIND":      "NIFTY INDIA MFG",          # Tennil Industries (manufacturing)
    "THELEELA":     "NIFTY CONSUMPTION",        # The Leela Palaces (luxury hotels)
    "TRAVELFOOD":   "NIFTY CONSUMPTION",        # Travel Food Services (airport F&B)
    "TREEHOUSE":    "NIFTY CONSUMPTION",        # Treehouse Education (preschool chain)
    "URBANCO":      "NIFTY CONSUMPTION",        # Urban Company (home services)
    "VENTIVE":      "NIFTY CONSUMPTION",        # Ventive Hospitality (luxury hotels)

    # Realty:
    "PENINLAND":    "NIFTY REALTY",             # Peninsular Land (real estate)

    # N500-only — remaining recurring tickers (5+ PIT dates):
    "BHARATRAS":    "NIFTY FMCG",               # Bharat Rasayan (agrochemicals)
    "BRFL":         "NIFTY CONSUMPTION",        # Bombay Rayon Fashions (textiles)
    "CAPLIPOINT":   "NIFTY HEALTHCARE",         # Caplin Point Laboratories
    "CHENNPETRO":   "NIFTY OIL & GAS",          # Chennai Petroleum Corporation
    "ELECON":       "NIFTY INDIA MFG",          # Elecon Engineering (gearboxes)
    "HLEGLAS":      "NIFTY INDIA MFG",          # HLE Glascoat (glass-lined equipment)
    "ICIL":         "NIFTY CONSUMPTION",        # Indo Count Industries (bed linen)
    "INDGN":        "NIFTY HEALTHCARE",         # Indegene (healthcare tech/CDMO)
    "INDIGOPNTS":   "NIFTY CONSUMER DURABLES",  # Indigo Paints
    "JAICORPLTD":   "NIFTY INDIA MFG",          # Jai Corp Ltd (plastics/textiles)
    "JWL":          "NIFTY INFRA",              # Jupiter Wagons (railway wagons)
    "JYOTICNC":     "NIFTY INDIA MFG",          # Jyoti CNC Automation
    "JYOTHYLAB":    "NIFTY FMCG",               # Jyothy Laboratories (household products)
    "KESORAMIND":   "NIFTY INDIA MFG",          # Kesoram Industries (cement/tyres)
    "KIRLOSBROS":   "NIFTY INDIA MFG",          # Kirloskar Brothers (pumps)
    "KIRLOSENG":    "NIFTY INDIA MFG",          # Kirloskar Electric Company
    "LTFOODS":      "NIFTY FMCG",               # LT Foods (Daawat rice brands)
    "MTNL":         "NIFTY CONSUMPTION",        # MTNL (telecom PSU)
    "NITINFIRE":    "NIFTY INFRA",              # Nitin Fire Protection
    "PTCIL":        "NIFTY INDIA MFG",          # PTC Industries (precision castings)
    "RAYMONDLSL":   "NIFTY CONSUMPTION",        # Raymond Lifestyle (textiles/fashion)
    "RIIL":         "NIFTY INFRA",              # Reliance Industrial Infrastructure
    "RNAVAL":       "NIFTY INFRA",              # Reliance Naval & Engineering
    "RRKABEL":      "NIFTY ENERGY",             # RR Kabel (electrical cables/wires)
    "RTNINDIA":     "NIFTY CONSUMPTION",        # RattanIndia Enterprises (drones/fashion)
    "SAPPHIRE":     "NIFTY CONSUMPTION",        # Sapphire Foods (KFC/Pizza Hut QSR)
    "SHANKARA":     "NIFTY INFRA",              # Shankara Building Products (retail)
    "SIGNATURE":    "NIFTY REALTY",             # Signature Global India (real estate)
    "SNOWMAN":      "NIFTY INFRA",              # Snowman Logistics (cold chain)
    "SYRMA":        "NIFTY INDIA MFG",          # Syrma SGS Technology (EMS)
    "TECHNOE":      "NIFTY ENERGY",             # Techno Electric & Engineering
    "TEXRAIL":      "NIFTY INDIA MFG",          # Texrail Industries
    "VSTIND":       "NIFTY FMCG",               # VST Industries (tobacco/cigarettes)
}

def _parse_args():
    p = argparse.ArgumentParser(description="Build PIT sector map for N200 or N500")
    p.add_argument("--n500", action="store_true", help="Build N500 map instead of N200")
    return p.parse_args()

_ARGS       = _parse_args()
PIT_JSON    = os.path.join(HERE, "nifty500_pit.json"            if _ARGS.n500 else "nifty200_pit.json")
OUTPUT_JSON = os.path.join(HERE, "nifty500_sector_map_pit.json" if _ARGS.n500 else "nifty200_sector_map_pit.json")


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
