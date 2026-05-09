#!/usr/bin/env python3
"""
diag_sector_fetch.py — diagnose why score_live_sectors() is returning
fewer than 19 sectors on EC2. Prints each sector's source + bar count.

Usage:
    venv/bin/python3 scripts/diag_sector_fetch.py
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from data.score_live_sectors import (  # noqa: E402
    _fetch_live_sector_prices,
    SECTOR_UNIVERSE,
    _YF_SECTORS_PKL,
    _load_cached_yf_sector_closes,
)

print("=" * 70)
print("Fetching live sector prices...")
t0 = time.time()
closes = _fetch_live_sector_prices()
elapsed = time.time() - t0
print(f"Took {elapsed:.1f}s, got {len(closes)} sector entries\n")

ok_count = 0
fail_count = 0
for sec, src in SECTOR_UNIVERSE:
    s = closes.get(sec)
    n = len(s) if s is not None else 0
    src_short = src if len(src) <= 30 else src[:27] + "..."
    flag = "OK   " if n >= 262 else f"FAIL ({n:>3} bars)"
    if n >= 262:
        ok_count += 1
    else:
        fail_count += 1
    print(f"  {sec:<26}  {src_short:<31}  {flag}")

print()
print(f"Summary: {ok_count} OK / {fail_count} FAIL out of {len(SECTOR_UNIVERSE)}")

print()
print("=" * 70)
print("Disk fallback pickle:")
if os.path.exists(_YF_SECTORS_PKL):
    age_h = (time.time() - os.path.getmtime(_YF_SECTORS_PKL)) / 3600.0
    size_kb = os.path.getsize(_YF_SECTORS_PKL) / 1024.0
    print(f"  Path:  {_YF_SECTORS_PKL}")
    print(f"  Age:   {age_h:.1f}h")
    print(f"  Size:  {size_kb:.1f} KB")
    disk = _load_cached_yf_sector_closes()
    print(f"  Sectors in disk pkl: {len(disk)}")
    for sec, s in disk.items():
        print(f"    {sec:<26}  {len(s)} bars")
else:
    print(f"  NOT FOUND at {_YF_SECTORS_PKL}")
    print(f"  → No disk fallback available. Once a fresh fetch succeeds,")
    print(f"    this pkl will be created automatically.")

print()
print("=" * 70)
print("Diagnostic done.")
