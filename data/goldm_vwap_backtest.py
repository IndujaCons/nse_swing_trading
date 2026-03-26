#!/usr/bin/env python3
"""
GOLDM VWAP Standard Deviation Backtest
Mean-reversion strategy on MCX Gold Mini 5-minute data.

Entry: Price touches ±2SD from VWAP + RSI confirmation + volume confirmation
Exit: T1 at VWAP (60%), T2 at opposite ±1SD (40%), SL at ±3SD, time stop 13:25

Usage: python3 data/goldm_vwap_backtest.py
"""

import os
import pandas as pd
import numpy as np
from datetime import time as dtime
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(__file__), "goldm_5min.csv")
LOT_SIZE = 100          # grams per lot
CHARGES_PER_LOT = 220   # ₹ round trip
RSI_PERIOD = 14
RSI_LONG_THRESHOLD = 35   # RSI < 35 for long
RSI_SHORT_THRESHOLD = 65  # RSI > 65 for short
VOL_MULTIPLIER = 1.2      # volume > 1.2x 10-bar avg
ENTRY_START = dtime(9, 30)
ENTRY_END = dtime(13, 0)
SQUAREOFF_TIME = dtime(13, 25)
ADX_THRESHOLD = 30        # skip if ADX > 30 on 15-min
GAP_THRESHOLD = 0.005     # skip if open gap > 0.5%
SD_SL = 3.0               # stop loss at 3 SD
T1_FRACTION = 0.6         # take 60% at T1 (VWAP)
CAPITAL = 50000            # ₹50K allocated
MAX_DAILY_LOSS = 1000      # ₹1K daily loss cap


def compute_rsi(closes, period=14):
    """RSI on Series."""
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def compute_adx(df_15m, period=14):
    """ADX on 15-min DataFrame."""
    high = df_15m['high']
    low = df_15m['low']
    close = df_15m['close']

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-10)) * 100
    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return adx


def run_backtest():
    """Run the VWAP SD backtest."""
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_convert('Asia/Kolkata')
    df = df.sort_values('date').reset_index(drop=True)
    df['time'] = df['date'].dt.time
    df['day'] = df['date'].dt.date

    # Filter to trading hours only (09:00 - 23:30 for MCX, but we trade 09:00-13:25)
    print(f"Total candles: {len(df)}")
    print(f"Trading days: {df['day'].nunique()}")
    print(f"Date range: {df['day'].min()} to {df['day'].max()}")

    # Pre-compute RSI on 5-min
    df['rsi'] = compute_rsi(df['close'], RSI_PERIOD)

    # Volume rolling avg (10 bars)
    df['vol_avg10'] = df['volume'].rolling(10, min_periods=1).mean()

    # Build 15-min bars for ADX filter
    df_15m = df.set_index('date').resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    df_15m['adx'] = compute_adx(df_15m)
    # Map ADX back to 5-min bars
    adx_map = {}
    for dt, row in df_15m.iterrows():
        adx_map[dt] = row['adx']

    # Previous day close for gap filter
    day_closes = df.groupby('day')['close'].last().to_dict()
    days_sorted = sorted(day_closes.keys())

    # ── VWAP computation + backtest loop (day by day) ──
    trades = []
    daily_pnl = defaultdict(float)

    for day_idx, day in enumerate(days_sorted):
        day_df = df[df['day'] == day].copy()
        if len(day_df) < 20:
            continue

        # Gap filter: skip if open > 0.5% from previous close
        if day_idx > 0:
            prev_close = day_closes[days_sorted[day_idx - 1]]
            day_open = day_df['open'].iloc[0]
            gap_pct = abs(day_open - prev_close) / prev_close
            if gap_pct > GAP_THRESHOLD:
                continue

        # Compute daily VWAP and SD bands
        tp = (day_df['high'] + day_df['low'] + day_df['close']) / 3
        tpvol = tp * day_df['volume']
        cum_tpvol = tpvol.cumsum()
        cum_vol = day_df['volume'].cumsum().replace(0, 1e-10)
        vwap = cum_tpvol / cum_vol

        # Cumulative variance
        variance = (day_df['volume'] * (tp - vwap) ** 2).cumsum() / cum_vol
        sd = np.sqrt(variance)

        day_df = day_df.copy()
        day_df['vwap'] = vwap.values
        day_df['sd'] = sd.values
        day_df['sd2_upper'] = vwap.values + 2 * sd.values
        day_df['sd2_lower'] = vwap.values - 2 * sd.values
        day_df['sd3_upper'] = vwap.values + 3 * sd.values
        day_df['sd3_lower'] = vwap.values - 3 * sd.values
        day_df['sd1_upper'] = vwap.values + 1 * sd.values
        day_df['sd1_lower'] = vwap.values - 1 * sd.values

        position = None
        day_loss = 0

        for i in range(len(day_df)):
            row = day_df.iloc[i]
            t = row['time']
            idx = day_df.index[i]

            # Daily loss cap
            if day_loss <= -MAX_DAILY_LOSS:
                break

            # Time stop: square off at 13:25
            if position and t >= SQUAREOFF_TIME:
                exit_price = row['close']
                if position['side'] == 'LONG':
                    pnl = (exit_price - position['entry']) * LOT_SIZE * position['remaining_frac'] - CHARGES_PER_LOT * position['remaining_frac']
                else:
                    pnl = (position['entry'] - exit_price) * LOT_SIZE * position['remaining_frac'] - CHARGES_PER_LOT * position['remaining_frac']

                trades.append({
                    'day': day, 'side': position['side'],
                    'entry': position['entry'], 'exit': exit_price,
                    'entry_time': position['entry_time'], 'exit_time': t,
                    'pnl': round(pnl, 2), 'result': 'EOD',
                    'sl': position['sl'], 'vwap_at_entry': position['vwap'],
                })
                daily_pnl[day] += pnl
                day_loss += pnl
                position = None
                continue

            # Manage open position
            if position:
                if position['side'] == 'LONG':
                    # SL check
                    if row['low'] <= position['sl']:
                        pnl = (position['sl'] - position['entry']) * LOT_SIZE * position['remaining_frac'] - CHARGES_PER_LOT * position['remaining_frac']
                        trades.append({
                            'day': day, 'side': 'LONG',
                            'entry': position['entry'], 'exit': position['sl'],
                            'entry_time': position['entry_time'], 'exit_time': t,
                            'pnl': round(pnl, 2), 'result': 'SL',
                            'sl': position['sl'], 'vwap_at_entry': position['vwap'],
                        })
                        daily_pnl[day] += pnl
                        day_loss += pnl
                        position = None
                        continue

                    # T1: VWAP hit — take 60%
                    if not position['t1_hit'] and row['high'] >= row['vwap']:
                        t1_pnl = (row['vwap'] - position['entry']) * LOT_SIZE * T1_FRACTION - CHARGES_PER_LOT * T1_FRACTION
                        position['t1_hit'] = True
                        position['remaining_frac'] = 1 - T1_FRACTION
                        position['sl'] = position['entry']  # move to breakeven
                        daily_pnl[day] += t1_pnl
                        day_loss += t1_pnl

                    # T2: +1SD hit — take remaining
                    if position and position['t1_hit'] and row['high'] >= row['sd1_upper']:
                        t2_pnl = (row['sd1_upper'] - position['entry']) * LOT_SIZE * position['remaining_frac'] - CHARGES_PER_LOT * position['remaining_frac']
                        trades.append({
                            'day': day, 'side': 'LONG',
                            'entry': position['entry'], 'exit': row['sd1_upper'],
                            'entry_time': position['entry_time'], 'exit_time': t,
                            'pnl': round(t1_pnl + t2_pnl if position['t1_hit'] else t2_pnl, 2),
                            'result': 'T2',
                            'sl': position['sl'], 'vwap_at_entry': position['vwap'],
                        })
                        daily_pnl[day] += t2_pnl
                        day_loss += t2_pnl
                        position = None
                        continue

                elif position['side'] == 'SHORT':
                    # SL check
                    if row['high'] >= position['sl']:
                        pnl = (position['entry'] - position['sl']) * LOT_SIZE * position['remaining_frac'] - CHARGES_PER_LOT * position['remaining_frac']
                        trades.append({
                            'day': day, 'side': 'SHORT',
                            'entry': position['entry'], 'exit': position['sl'],
                            'entry_time': position['entry_time'], 'exit_time': t,
                            'pnl': round(pnl, 2), 'result': 'SL',
                            'sl': position['sl'], 'vwap_at_entry': position['vwap'],
                        })
                        daily_pnl[day] += pnl
                        day_loss += pnl
                        position = None
                        continue

                    # T1: VWAP hit
                    if not position['t1_hit'] and row['low'] <= row['vwap']:
                        t1_pnl = (position['entry'] - row['vwap']) * LOT_SIZE * T1_FRACTION - CHARGES_PER_LOT * T1_FRACTION
                        position['t1_hit'] = True
                        position['remaining_frac'] = 1 - T1_FRACTION
                        position['sl'] = position['entry']  # breakeven
                        daily_pnl[day] += t1_pnl
                        day_loss += t1_pnl

                    # T2: -1SD hit
                    if position and position['t1_hit'] and row['low'] <= row['sd1_lower']:
                        t2_pnl = (position['entry'] - row['sd1_lower']) * LOT_SIZE * position['remaining_frac'] - CHARGES_PER_LOT * position['remaining_frac']
                        trades.append({
                            'day': day, 'side': 'SHORT',
                            'entry': position['entry'], 'exit': row['sd1_lower'],
                            'entry_time': position['entry_time'], 'exit_time': t,
                            'pnl': round(t1_pnl + t2_pnl if position['t1_hit'] else t2_pnl, 2),
                            'result': 'T2',
                            'sl': position['sl'], 'vwap_at_entry': position['vwap'],
                        })
                        daily_pnl[day] += t2_pnl
                        day_loss += t2_pnl
                        position = None
                        continue

                continue  # don't enter while in position

            # ── Entry signals (only 09:30 - 13:00, no position open) ──
            if t < ENTRY_START or t > ENTRY_END:
                continue

            # Skip first 30 min (SD bands too narrow)
            if t < dtime(9, 30):
                continue

            # SD must be meaningful (at least ₹5/gram)
            if row['sd'] < 5:
                continue

            # ADX filter: find most recent 15-min ADX
            bar_15m = row['date'].floor('15min')
            adx_val = adx_map.get(bar_15m, 0)
            if adx_val > ADX_THRESHOLD:
                continue

            # Volume filter
            if row['volume'] <= VOL_MULTIPLIER * row['vol_avg10']:
                continue

            # LONG: low pierces -2SD, close above -2SD, RSI < 35
            if (row['low'] <= row['sd2_lower'] and
                row['close'] > row['sd2_lower'] and
                row['rsi'] < RSI_LONG_THRESHOLD):

                entry = row['high'] + 1  # 1 tick above reversal candle
                sl = row['sd3_lower']
                position = {
                    'side': 'LONG', 'entry': entry, 'sl': sl,
                    'vwap': row['vwap'], 'entry_time': t,
                    't1_hit': False, 'remaining_frac': 1.0,
                }

            # SHORT: high pierces +2SD, close below +2SD, RSI > 65
            elif (row['high'] >= row['sd2_upper'] and
                  row['close'] < row['sd2_upper'] and
                  row['rsi'] > RSI_SHORT_THRESHOLD):

                entry = row['low'] - 1  # 1 tick below reversal candle
                sl = row['sd3_upper']
                position = {
                    'side': 'SHORT', 'entry': entry, 'sl': sl,
                    'vwap': row['vwap'], 'entry_time': t,
                    't1_hit': False, 'remaining_frac': 1.0,
                }

    # ── Results ──
    print_results(trades, daily_pnl)
    return trades


def print_results(trades, daily_pnl):
    """Print backtest results."""
    if not trades:
        print("\nNo trades generated!")
        return

    df = pd.DataFrame(trades)
    total_pnl = df['pnl'].sum()
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] <= 0]

    print(f"\n{'=' * 70}")
    print(f"GOLDM VWAP SD BACKTEST RESULTS")
    print(f"{'=' * 70}")
    print(f"Period: {df['day'].min()} to {df['day'].max()}")
    print(f"Capital: ₹{CAPITAL:,} | Charges: ₹{CHARGES_PER_LOT}/lot")
    print(f"{'=' * 70}")

    print(f"\n  Total trades:    {len(df)}")
    print(f"  Wins:            {len(wins)} ({len(wins)/len(df)*100:.0f}%)")
    print(f"  Losses:          {len(losses)} ({len(losses)/len(df)*100:.0f}%)")
    print(f"  Total P&L:       ₹{total_pnl:+,.0f}")
    print(f"  Avg win:         ₹{wins['pnl'].mean():+,.0f}" if len(wins) > 0 else "")
    print(f"  Avg loss:        ₹{losses['pnl'].mean():+,.0f}" if len(losses) > 0 else "")
    print(f"  Profit Factor:   {wins['pnl'].sum() / abs(losses['pnl'].sum()):.2f}" if len(losses) > 0 and losses['pnl'].sum() != 0 else "")
    print(f"  Avg P&L/trade:   ₹{df['pnl'].mean():+,.0f}")

    # By result type
    print(f"\n  Exit breakdown:")
    for result, group in df.groupby('result'):
        print(f"    {result:<6}: {len(group):>3} trades  ₹{group['pnl'].sum():>+8,.0f}")

    # By side
    print(f"\n  By side:")
    for side, group in df.groupby('side'):
        w = len(group[group['pnl'] > 0])
        print(f"    {side:<6}: {len(group):>3} trades  W:{w} L:{len(group)-w}  ₹{group['pnl'].sum():>+8,.0f}")

    # Monthly breakdown
    df['month'] = pd.to_datetime(df['day']).dt.to_period('M')
    print(f"\n  Monthly P&L:")
    for month, group in df.groupby('month'):
        print(f"    {month}: {len(group):>3} trades  ₹{group['pnl'].sum():>+8,.0f}")

    # Daily P&L stats
    daily_vals = list(daily_pnl.values())
    if daily_vals:
        pos_days = sum(1 for d in daily_vals if d > 0)
        neg_days = sum(1 for d in daily_vals if d < 0)
        zero_days = sum(1 for d in daily_vals if d == 0)
        print(f"\n  Daily stats:")
        print(f"    Trading days:  {len(daily_vals)}")
        print(f"    Positive days: {pos_days} ({pos_days/len(daily_vals)*100:.0f}%)")
        print(f"    Negative days: {neg_days}")
        print(f"    No-trade days: {zero_days}")
        print(f"    Best day:      ₹{max(daily_vals):+,.0f}")
        print(f"    Worst day:     ₹{min(daily_vals):+,.0f}")
        print(f"    Avg day:       ₹{np.mean(daily_vals):+,.0f}")


if __name__ == "__main__":
    trades = run_backtest()
