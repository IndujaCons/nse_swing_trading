"""
Build point-in-time Nifty 500 constituent database.
Parses all NSE press-release PDFs to extract Nifty 500 reconstitutions,
then walks backward from the current list.

Output: nifty500_pit.json — dict mapping effective_date → list of symbols
"""
import json, csv, re, os
from datetime import datetime
import pdfplumber

CONST_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load current Nifty 500 list ───────────────────────────────────────────────
with open(f'{CONST_DIR}/ind_nifty500list.csv') as f:
    reader = csv.DictReader(f)
    current = sorted([row['Symbol'].strip() for row in reader])
print(f"Current Nifty 500: {len(current)} stocks")

# ── PDF parser ────────────────────────────────────────────────────────────────
MONTH_MAP = {
    'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,
    'july':7,'august':8,'september':9,'october':10,'november':11,'december':12
}

def parse_effective_date(text):
    """Extract effective date from PDF text."""
    patterns = [
        r'effective\s+from\s+([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})',
        r'w\.e\.f\.?\s+([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            mon = m.group(1).lower().strip()
            if mon in MONTH_MAP:
                try:
                    return f"{m.group(3)}-{MONTH_MAP[mon]:02d}-{int(m.group(2)):02d}"
                except:
                    pass
    return None

def extract_symbols_from_block(block_text):
    """Extract NSE symbols from a table block.
    Lines look like: '1 Company Name Ltd. SYMBOL'
    Symbol is the last ALL-CAPS token on the line.
    """
    syms = []
    for line in block_text.split('\n'):
        line = line.strip()
        # Must start with a number (row number)
        if not re.match(r'^\d+\s', line):
            continue
        # Last token that is all uppercase letters/digits (NSE symbol pattern)
        tokens = line.split()
        if len(tokens) < 3:
            continue
        # Try last token first, then second-last (sometimes trailing spaces)
        for tok in reversed(tokens):
            if re.match(r'^[A-Z0-9&\-\.]{2,20}$', tok) and tok not in ('LTD', 'LTD.', 'LLP', 'PVT'):
                syms.append(tok)
                break
    return syms

def parse_nifty500_from_pdf(path):
    """Return (effective_date_str, excluded_syms, included_syms) for Nifty 500."""
    with pdfplumber.open(path) as pdf:
        text = '\n'.join(page.extract_text() or '' for page in pdf.pages)

    eff_date = parse_effective_date(text)

    # Find the Nifty 500 replacement section (must contain "excluded" or "included")
    # Handles both numbered (3) Nifty 500) and letter-prefixed (c) NIFTY 500) formats
    terminator = (
        r'(?=\n\s*\d+\)\s+(?:Nifty|NIFTY)|'
        r'\n\s*[a-z]\)\s+(?:Nifty|NIFTY)|'
        r'\n\s*[A-Z]\.\s+|'
        r'\n\s*Note:|$)'
    )
    n500_pat = re.compile(
        r'(?:Nifty 500|NIFTY 500).*?' + terminator,
        re.DOTALL | re.IGNORECASE
    )
    section = None
    for m in n500_pat.finditer(text):
        candidate = m.group(0)
        if re.search(r'\bexcluded\b|\bincluded\b', candidate, re.IGNORECASE):
            section = candidate
            break
    if not section:
        return eff_date, [], []

    # Split excluded / included blocks
    excl_m = re.search(
        r'(?:being\s+)?excluded[:\s]*(.*?)(?=(?:being\s+)?included[:\s]|$)',
        section, re.DOTALL | re.IGNORECASE
    )
    incl_m = re.search(
        r'(?:being\s+)?included[:\s]*(.*?)(?=\n\s*\d+\)\s+(?:Nifty|NIFTY)|'
        r'\n\s*[A-Z]\.\s+Replacements|'
        r'Note:|$)',
        section, re.DOTALL | re.IGNORECASE
    )

    excl = extract_symbols_from_block(excl_m.group(1)) if excl_m else []
    incl = extract_symbols_from_block(incl_m.group(1)) if incl_m else []
    return eff_date, excl, incl

# ── Parse all PDFs ────────────────────────────────────────────────────────────
pdfs = sorted([f for f in os.listdir(CONST_DIR) if f.endswith('.pdf')])
print(f"\nParsing {len(pdfs)} PDFs...")

recons = []
for fname in pdfs:
    path = os.path.join(CONST_DIR, fname)
    eff, excl, incl = parse_nifty500_from_pdf(path)
    status = "OK" if eff else "DATE?"
    print(f"  {fname}: eff={eff} | excl={len(excl)} | incl={len(incl)} [{status}]")
    if eff and (excl or incl):
        recons.append({
            "label":          fname.replace('.pdf',''),
            "effective_date": eff,
            "excluded":       excl,
            "included":       incl,
        })

# Sort newest first for backward walk
recons.sort(key=lambda x: x['effective_date'], reverse=True)

print(f"\n{len(recons)} reconstitutions parsed")
print("\nReconstitution summary (newest first):")
for r in recons:
    print(f"  {r['effective_date']}: -{len(r['excluded'])} +{len(r['included'])}"
          f"  (net {len(r['included'])-len(r['excluded']):+d})")

# ── Corporate actions (mergers, renames, delistings) ─────────────────────────
# These are NOT in the semi-annual PDFs — manually maintained
# (same events as Nifty 200 where applicable, plus 500-specific ones)
CORPORATE_ACTIONS = [
    {"label": "TATAMOTORS demerger",         "effective_date": "2025-10-01",
     "excluded": ["TATAMTRDVR"], "included": []},
    {"label": "ZOMATO→ETERNAL rename",        "effective_date": "2025-01-10",
     "excluded": ["ZOMATO"], "included": ["ETERNAL"]},
    {"label": "GMRINFRA→GMRAIRPORT rename",   "effective_date": "2024-08-01",
     "excluded": ["GMRINFRA"], "included": ["GMRAIRPORT"]},
    {"label": "HDFC→HDFCBANK merger",         "effective_date": "2023-07-13",
     "excluded": ["HDFC"], "included": []},
    {"label": "MINDTREE+LTI→LTIM merger",     "effective_date": "2022-11-14",
     "excluded": ["LTI"], "included": ["LTIM"]},
    {"label": "Bank mergers Apr 2020",        "effective_date": "2020-04-01",
     "excluded": ["ALBK", "ORIENTBANK", "SYNDIBANK", "ANDHRABANK"], "included": []},
    {"label": "RNAM→NAM-INDIA rename",        "effective_date": "2019-10-07",
     "excluded": ["RNAM"], "included": ["NAM-INDIA"]},
    {"label": "IDFCBANK→IDFCFIRSTB rename",   "effective_date": "2018-12-17",
     "excluded": ["IDFCBANK"], "included": ["IDFCFIRSTB"]},
    {"label": "MCDOWELL-N→UNITDSPR rename",   "effective_date": "2018-06-01",
     "excluded": ["MCDOWELL-N"], "included": ["UNITDSPR"]},
    {"label": "ABIRLANUVO→GRASIM merger",     "effective_date": "2017-07-03",
     "excluded": ["ABIRLANUVO"], "included": []},
    {"label": "IBVENTURES→DHANI rename",      "effective_date": "2021-06-01",
     "excluded": ["IBVENTURES"], "included": ["DHANI"]},
    {"label": "LTIM→LTM rename",              "effective_date": "2024-01-01",
     "excluded": ["LTIM"], "included": ["LTM"]},
    {"label": "FCONSUMER delisted",           "effective_date": "2022-08-01",
     "excluded": ["FCONSUMER"], "included": []},
]

ALL_EVENTS = recons + CORPORATE_ACTIONS
ALL_EVENTS.sort(key=lambda x: x['effective_date'], reverse=True)

# ── Backward walk ─────────────────────────────────────────────────────────────
pit_db = {}
constituents = set(current)

print(f"\nStarting backward walk from {len(constituents)} stocks")

for ev in ALL_EVENTS:
    eff  = ev['effective_date']
    excl = set(ev['excluded'])
    incl = set(ev['included'])
    label = ev['label']

    # Save list AFTER this event
    pit_db[eff] = sorted(constituents)

    # Warn on mismatches
    missing = incl - constituents
    if missing:
        print(f"  WARN {label}: included not in set: {missing}")

    # Reverse the event
    constituents = (constituents - incl) | excl

    print(f"  {eff} {label}: -{len(excl)} +{len(incl)} → {len(constituents)} stocks")

# Pre-first-event list
pit_db["2015-01-01"] = sorted(constituents)
print(f"\nPre-2015 list: {len(constituents)} stocks")

# ── Validation ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("VALIDATION — stock counts per period:")
print(f"{'='*60}")
for d in sorted(pit_db):
    n = len(pit_db[d])
    flag = "" if 490 <= n <= 510 else f"  *** CHECK: {n} ***"
    print(f"  {d}: {n} stocks{flag}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_json = os.path.join(CONST_DIR, 'nifty500_pit.json')
with open(out_json, 'w') as f:
    json.dump(pit_db, f, indent=2, sort_keys=True)
print(f"\nSaved → {out_json}")

out_csv = os.path.join(CONST_DIR, 'nifty500_pit.csv')
all_dates   = sorted(pit_db.keys())
all_symbols = sorted(set(s for lst in pit_db.values() for s in lst))
with open(out_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Symbol'] + all_dates)
    for sym in all_symbols:
        row = [sym] + [1 if sym in pit_db[d] else 0 for d in all_dates]
        writer.writerow(row)
print(f"Saved → {out_csv}")
print(f"Total unique symbols: {len(all_symbols)}")
