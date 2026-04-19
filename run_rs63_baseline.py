#!/usr/bin/env python3
"""Run the frozen RS63 baseline: run_rs55_backtest()"""
import sys
sys.path.insert(0, 'data')

from momentum_backtest import MomentumBacktester
from datetime import datetime
import pandas as pd

bt = MomentumBacktester()
print("Running RS63 baseline backtest (data: 06-Apr-2026 cache)...")
r  = bt.run_rs55_backtest(end_date=datetime(2026, 4, 6))

trades = pd.DataFrame(r['trades'])
if trades.empty:
    print("No trades returned.")
    sys.exit(1)

trades['exit_date'] = pd.to_datetime(trades['exit_date'])
winners = trades[trades['pnl'] > 0]
losers  = trades[trades['pnl'] <= 0]
pf      = winners['pnl'].sum() / abs(losers['pnl'].sum()) if len(losers) > 0 else float('inf')

print()
print("=" * 60)
print("  RS63 BASELINE RESULTS")
print("=" * 60)
print(f"  Trades      : {len(trades)}")
print(f"  Win Rate    : {len(winners)/len(trades)*100:.1f}%")
print(f"  Profit Factor: {pf:.2f}")
print(f"  Net PnL     : ₹{trades['pnl'].sum():,.0f}")
print()
print("  YEAR-BY-YEAR (% of initial ₹10L capital):")
INITIAL = 10_00_000.0
for yr, grp in trades.groupby(trades['exit_date'].dt.year):
    pnl_y = grp['pnl'].sum()
    ret_y = pnl_y / INITIAL * 100
    bar   = ("█" * min(int(abs(ret_y)), 50)) if ret_y > 0 else ("░" * min(int(abs(ret_y)), 50))
    print(f"  {yr}  {'+' if ret_y>=0 else '-'}{abs(ret_y):5.1f}%  {bar}")
print()
