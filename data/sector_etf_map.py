"""
SECTOR_TO_ETF — maps a Phase 1 sector index name to the NSE-listed ETF
that tracks it (used by Mom20 basket flow to top up under-covered top-5
sectors with <4 stocks).

Format: {sector_name: (etf_symbol, etf_human_name)} or sector_name: None
when no NSE ETF exists for that index. Existing yfinance lookups expect
.NS suffix; the symbols here are the bare NSE tradingsymbols.
"""

SECTOR_TO_ETF = {
    "NIFTY INDIA DEFENCE":     ("MODEFENCE",  "Motilal Oswal Defence ETF"),
    "NIFTY METAL":             ("METALIETF",  "Nippon Nifty Metal ETF"),
    "NIFTY HEALTHCARE":        ("HEALTHIETF", "Nippon Nifty Healthcare ETF"),
    "NIFTY IT":                ("ITBEES",     "Nippon Nifty IT ETF"),
    "NIFTY AUTO":              ("AUTOBEES",   "Nippon Nifty Auto ETF"),
    "NIFTY FMCG":              ("FMCGIETF",   "ICICI Pru Nifty FMCG ETF"),
    "NIFTY PVT BANK":          ("HDFCPVTBAN", "HDFC Nifty Private Bank ETF"),
    "NIFTY PSU BANK":          ("PSUBNKBEES", "Nippon Nifty PSU Bank ETF"),
    "NIFTY BANK":              ("BANKBEES",   "Nippon Nifty Bank ETF"),
    "NIFTY INFRA":             ("INFRABEES",  "Nippon Nifty Infra ETF"),
    "NIFTY OIL & GAS":         ("OILIETF",    "ICICI Pru Nifty Oil & Gas ETF"),
    "NIFTY CONSUMPTION":       ("CONSUMBEES", "Nippon Nifty Consumption ETF"),
    "NIFTY REALTY":            ("MOREALTY",   "Motilal Oswal Realty ETF"),
    "NIFTY PSE":               ("CPSEETF",    "(proxy) CPSE ETF"),
    "NIFTY INDIA MFG":         ("MAKEINDIA",  "Mirae Asset Make-in-India ETF"),
    "NIFTY CONSUMER DURABLES": ("CONSUMBEES", "(proxy) Nippon Consumption ETF"),
    # Sectors with no NSE-tradable single-sector ETF (verified missing on
    # yfinance under common ticker forms — basket flow simply skips top-up):
    "NIFTY ENERGY":            None,   # ENERGYIETF / NETFNIFENG / etc. all 404
    "NIFTY MEDIA":             None,
    "NIFTY MNC":               None,
    "NIFTY FIN SERVICE":       None,
}
