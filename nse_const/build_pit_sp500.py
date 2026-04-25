"""
Build point-in-time S&P 500 constituent database.
Parses Wikipedia change history and walks backward from current list.

Output: sp500_pit.json — dict mapping effective_date → list of tickers
Note: Wikipedia tracks ~394 major changes (not all ~20/yr). Pre-2010
counts drift due to mergers/delistings not tracked — fine for backtests
as missing tickers have no price data and are skipped automatically.
"""
import json, re, os, requests
import pandas as pd
from io import StringIO
from datetime import datetime

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}

print("Fetching Wikipedia S&P 500 page...")
r = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=HEADERS)
r.raise_for_status()
tables = pd.read_html(StringIO(r.text))

# ── Current components (Table 0) ──────────────────────────────────────────────
comp_table = tables[0]
current = sorted(comp_table['Symbol'].dropna().str.strip().str.replace('.', '-', regex=False).tolist())
print(f"Current S&P 500: {len(current)} tickers")

# ── Historical changes (Table 1) ─────────────────────────────────────────────
chg = tables[1].copy()
chg.columns = ['date', 'added_ticker', 'added_name', 'removed_ticker', 'removed_name', 'reason']
chg = chg[chg['date'] != 'Effective Date'].reset_index(drop=True)

def parse_date(s):
    if pd.isna(s) or not str(s).strip():
        return None
    s = str(s).strip()
    for fmt in ['%B %d, %Y', '%B %d %Y', '%b %d, %Y', '%b. %d, %Y']:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            pass
    m = re.search(r'(\w+\.?\s+\d+,?\s*\d{4})', s)
    if m:
        try:
            cleaned = m.group(1).replace(',', '').replace('.', '')
            return datetime.strptime(cleaned, '%B %d %Y').date().isoformat()
        except Exception:
            pass
    return None

def clean_ticker(s):
    if pd.isna(s):
        return None
    s = str(s).strip().replace('.', '-')  # BRK.B → BRK-B
    if not s or s == 'nan':
        return None
    return s.split()[0].upper()

events = []
for _, row in chg.iterrows():
    eff     = parse_date(row['date'])
    added   = clean_ticker(row['added_ticker'])
    removed = clean_ticker(row['removed_ticker'])
    if eff and (added or removed):
        events.append({
            'effective_date': eff,
            'added':   [added]   if added   else [],
            'removed': [removed] if removed else [],
            'label':   f"+{added or '—'} / -{removed or '—'}",
        })

events.sort(key=lambda x: x['effective_date'], reverse=True)
print(f"Parsed {len(events)} total change events ({events[-1]['effective_date']} → {events[0]['effective_date']})")

# ── Filter to 2015+ only ──────────────────────────────────────────────────────
events = [e for e in events if e['effective_date'] >= '2015-01-01']
print(f"Using {len(events)} events from 2015+ ({events[-1]['effective_date']} → {events[0]['effective_date']})")

# ── Backward walk ─────────────────────────────────────────────────────────────
pit_db = {}
constituents = set(current)
print(f"\nStarting backward walk from {len(constituents)} tickers")

warn_count = 0
for ev in events:
    eff     = ev['effective_date']
    added   = set(ev['added'])
    removed = set(ev['removed'])

    pit_db[eff] = sorted(constituents)

    missing = added - constituents
    if missing:
        warn_count += 1

    constituents = (constituents - added) | removed

pit_db['2015-01-01'] = sorted(constituents)
print(f"2015-01-01 baseline: {len(constituents)} tickers")
print(f"Total WARNs (delisted/merged stocks): {warn_count}")

# ── Validation ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("VALIDATION — recent periods (2010+):")
print(f"{'='*60}")
for d in sorted(pit_db):
    if d < '2015':
        continue
    n = len(pit_db[d])
    flag = "" if 480 <= n <= 520 else f"  *** CHECK: {n} ***"
    print(f"  {d}: {n} tickers{flag}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_json = os.path.join(OUT_DIR, 'sp500_pit.json')
with open(out_json, 'w') as f:
    json.dump(pit_db, f, indent=2, sort_keys=True)
print(f"\nSaved → {out_json}")

all_tickers = sorted(set(t for lst in pit_db.values() for t in lst))
all_dates   = sorted(pit_db.keys())
print(f"Total unique tickers: {len(all_tickers)}")
print(f"Date range: {all_dates[0]} → {all_dates[-1]}")
