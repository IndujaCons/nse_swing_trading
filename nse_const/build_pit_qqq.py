"""
Build point-in-time Nasdaq-100 (QQQ) constituent database.
Parses Wikipedia change history and walks backward from current list.

Output: qqq_pit.json — dict mapping effective_date → list of tickers
"""
import json, re, os, requests
import pandas as pd
from io import StringIO
from datetime import datetime

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}

print("Fetching Wikipedia Nasdaq-100 page...")
r = requests.get('https://en.wikipedia.org/wiki/Nasdaq-100', headers=HEADERS)
r.raise_for_status()
tables = pd.read_html(StringIO(r.text))

# ── Current components (Table 5) ──────────────────────────────────────────────
comp_table = tables[5]
current = sorted(comp_table['Ticker'].dropna().str.strip().tolist())
print(f"Current Nasdaq-100: {len(current)} tickers")

# ── Historical changes (Table 6) ─────────────────────────────────────────────
chg = tables[6].copy()
# Multi-level columns — flatten
chg.columns = ['date', 'added_ticker', 'added_name', 'removed_ticker', 'removed_name', 'reason']

# Drop the repeated header row (Wikipedia quirk)
chg = chg[chg['date'] != 'Date'].reset_index(drop=True)

def parse_date(s):
    if pd.isna(s) or not str(s).strip():
        return None
    s = str(s).strip()
    for fmt in ['%B %d, %Y', '%B %d %Y', '%b %d, %Y']:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            pass
    # Try partial formats
    m = re.search(r'(\w+ \d+,?\s*\d{4})', s)
    if m:
        try:
            return datetime.strptime(m.group(1).replace(',', ''), '%B %d %Y').date().isoformat()
        except Exception:
            pass
    return None

def clean_ticker(s):
    if pd.isna(s):
        return None
    s = str(s).strip()
    if not s or s == 'nan':
        return None
    # Take first word (ticker) — ignore any trailing annotation
    return s.split()[0].upper()

events = []
for _, row in chg.iterrows():
    eff = parse_date(row['date'])
    added   = clean_ticker(row['added_ticker'])
    removed = clean_ticker(row['removed_ticker'])
    if eff and (added or removed):
        events.append({
            'effective_date': eff,
            'added':   [added]   if added   else [],
            'removed': [removed] if removed else [],
            'label':   f"{eff}: +{added or '—'} / -{removed or '—'}",
        })

# Sort newest first for backward walk
events.sort(key=lambda x: x['effective_date'], reverse=True)

print(f"Parsed {len(events)} change events ({events[-1]['effective_date']} → {events[0]['effective_date']})")

# ── Backward walk ─────────────────────────────────────────────────────────────
pit_db = {}
constituents = set(current)
print(f"\nStarting backward walk from {len(constituents)} tickers")

for ev in events:
    eff     = ev['effective_date']
    added   = set(ev['added'])
    removed = set(ev['removed'])

    # Save AFTER this event (current state before reversing)
    pit_db[eff] = sorted(constituents)

    # Warn on mismatches
    missing = added - constituents
    if missing:
        print(f"  WARN {ev['label']}: added not in set: {missing}")

    # Reverse: remove what was added, add back what was removed
    constituents = (constituents - added) | removed

    print(f"  {eff}: -{len(added)} +{len(removed)} → {len(constituents)} tickers")

# Earliest snapshot
pit_db['2007-01-01'] = sorted(constituents)
print(f"\nPre-2007 list: {len(constituents)} tickers")

# ── Validation ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("VALIDATION — ticker counts per period:")
print(f"{'='*60}")
for d in sorted(pit_db):
    n = len(pit_db[d])
    flag = "" if 95 <= n <= 105 else f"  *** CHECK: {n} ***"
    print(f"  {d}: {n} tickers{flag}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_json = os.path.join(OUT_DIR, 'qqq_pit.json')
with open(out_json, 'w') as f:
    json.dump(pit_db, f, indent=2, sort_keys=True)
print(f"\nSaved → {out_json}")

all_dates   = sorted(pit_db.keys())
all_tickers = sorted(set(t for lst in pit_db.values() for t in lst))
print(f"Total unique tickers: {len(all_tickers)}")
print(f"Date range: {all_dates[0]} → {all_dates[-1]}")
