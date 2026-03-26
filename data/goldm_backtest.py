"""
GOLDM Intraday Backtest Engine
================================
Strategies:
  1. ORB  — Opening Range Breakout (trending regime)
  2. VWAP_SD — VWAP ±2SD Mean Reversion with tight stop (ranging regime)
  3. HYBRID — Regime detector routes to ORB or VWAP_SD each day

Usage:
  python data/goldm_backtest.py --strategy all

Dependencies:
  pip install pandas numpy
"""

import argparse
import json
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
import itertools

import numpy as np
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

CFG = {
    # Contract
    "lot_size"          : 100,
    "charges_per_lot"   : 220,
    "tick_size"         : 1,

    # Risk
    "capital"           : 50_000,
    "max_daily_loss_pct": 0.02,
    "max_risk_per_trade": 0.01,

    # Session
    "session_start"     : "09:00",
    "session_end"       : "23:30",
    "trade_start"       : "09:30",
    "trade_end"         : "23:00",
    "squareoff_time"    : "23:20",

    # ORB params
    "orb_minutes"       : 30,         # 30 min = 2 bars at 15-min
    "orb_target_mult"   : 1.5,
    "orb_sl_mult"       : 1.0,
    "orb_long_only"     : False,       # trade both sides
    "slippage_ticks"    : 2,          # 2 ticks slippage per side

    # VWAP SD params (improved)
    "sd_entry_band"     : 2.0,
    "sd_t1"             : 0.0,
    "sd_t2_band"        : 1.0,
    "sd_hard_stop_pct"  : 0.002,     # 0.2% of price as max stop
    "sd_rsi_long"       : 35,
    "sd_rsi_short"      : 65,
    "sd_vol_mult"       : 1.2,
    "sd_min_sd_gram"    : 3,

    # Regime detection
    "regime_adx_trend"  : 28,
    "regime_slope_pct"  : 0.15,
    "regime_crosses_min": 3,
}

# =============================================================================
# DATA LOADER — use pre-fetched CSV
# =============================================================================

def load_data(resample='15min') -> pd.DataFrame:
    """Load GOLDM 5-min data from CSV, optionally resample to 15-min."""
    csv_path = Path(__file__).parent / "goldm_5min.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run fetch_goldm_data.py first.")
        raise SystemExit(1)

    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    if df['date'].dt.tz is not None:
        df['date'] = df['date'].dt.tz_convert('Asia/Kolkata')
    else:
        df['date'] = df['date'].dt.tz_localize('Asia/Kolkata')
    df = df.set_index('date').sort_index()

    if resample and resample != '5min':
        df = df.resample(resample).agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()
        print(f"  Resampled to {resample}: {len(df)} bars")

    return df

# =============================================================================
# INDICATORS
# =============================================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['date_only'] = df.index.date

    # VWAP + bands (daily reset)
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['tpvol'] = df['tp'] * df['volume']

    vwap_list = []
    sd_list = []
    for date, grp in df.groupby('date_only'):
        cum_vol = grp['volume'].cumsum()
        cum_tpvol = grp['tpvol'].cumsum()
        vwap = cum_tpvol / cum_vol.replace(0, np.nan)
        sq_dev = grp['volume'] * (grp['tp'] - vwap) ** 2
        variance = sq_dev.cumsum() / cum_vol.replace(0, np.nan)
        sd = np.sqrt(variance.fillna(0))
        vwap_list.append(vwap)
        sd_list.append(sd)

    df['vwap'] = pd.concat(vwap_list)
    df['sd'] = pd.concat(sd_list)

    for mult in [1, 2, 3]:
        df[f'sd{mult}_upper'] = df['vwap'] + mult * df['sd']
        df[f'sd{mult}_lower'] = df['vwap'] - mult * df['sd']

    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    # ADX (simplified)
    plus_dm = df['high'].diff().clip(lower=0)
    minus_dm = (-df['low'].diff()).clip(lower=0)
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr14
    minus_di = 100 * minus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr14
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-10)) * 100
    df['adx'] = dx.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    # VWAP slope (% per 6 bars = 30 min)
    df['vwap_slope_pct'] = df['vwap'].diff(6) / df['vwap'].shift(6) * 100

    # Volume avg
    df['vol_avg10'] = df['volume'].rolling(10).mean()

    return df

# =============================================================================
# REGIME DETECTOR
# =============================================================================

def detect_regime(day_df: pd.DataFrame, cfg: dict) -> str:
    if len(day_df) < 20:
        return 'SKIP'

    mid = day_df.iloc[12:]
    slope_abs = mid['vwap_slope_pct'].abs().median()
    adx_med = mid['adx'].median() if mid['adx'].notna().any() else 20

    cross_series = ((day_df['close'] - day_df['vwap']).shift(1) *
                    (day_df['close'] - day_df['vwap']) < 0)
    vwap_crosses = cross_series.sum()

    trending = (slope_abs > cfg['regime_slope_pct'] or adx_med > cfg['regime_adx_trend'])
    ranging = (slope_abs < cfg['regime_slope_pct'] * 0.6 and
               vwap_crosses >= cfg['regime_crosses_min'])

    if trending:
        return 'TRENDING'
    elif ranging:
        return 'RANGING'
    else:
        return 'NEUTRAL'

# =============================================================================
# STRATEGY 1: ORB
# =============================================================================

def strategy_orb(day_df: pd.DataFrame, cfg: dict) -> list:
    """
    ORB with realistic execution:
    - Entry on NEXT bar open after breakout confirmation (not same bar)
    - Max 1 trade per day (no re-entry after SL/target)
    - Slippage on entry and exit
    """
    trades = []
    date_str = str(day_df.index[0].date())
    slippage = cfg['slippage_ticks'] * cfg['tick_size']

    or_end = day_df.index[0] + pd.Timedelta(minutes=cfg['orb_minutes'])
    or_data = day_df[day_df.index <= or_end]
    trade_data = day_df[
        (day_df.index > or_end) &
        (day_df.index.time <= pd.Timestamp(cfg['squareoff_time']).time())
    ]

    if len(or_data) < 2 or len(trade_data) < 2:
        return []

    or_high = or_data['high'].max()
    or_low = or_data['low'].min()
    or_range = or_high - or_low

    if or_range <= 0:
        return []

    # Skip extreme range days (news spikes) — OR > 0.5% of price
    or_range_pct = or_range / or_data['close'].iloc[-1] * 100
    if or_range_pct > 0.3:
        return []

    target_dist = or_range * cfg['orb_target_mult']
    sl_dist = or_range * cfg['orb_sl_mult']

    position = None
    breakout_signal = None  # 'LONG' or 'SHORT' — wait for next bar to enter
    traded_today = False    # max 1 trade per day

    trade_bars = list(trade_data.iterrows())

    for bar_idx, (ts, row) in enumerate(trade_bars):
        # Execute pending entry on this bar's open (next bar after signal)
        if breakout_signal and not traded_today:
            if breakout_signal == 'LONG':
                entry = row['open'] + slippage
                position = dict(side='LONG', entry=entry,
                                sl=entry - sl_dist,       # SL from entry, not OR
                                target=entry + target_dist,
                                entry_ts=ts, date=date_str)
            elif breakout_signal == 'SHORT':
                entry = row['open'] - slippage
                position = dict(side='SHORT', entry=entry,
                                sl=entry + sl_dist,       # SL from entry, not OR
                                target=entry - target_dist,
                                entry_ts=ts, date=date_str)
            traded_today = True
            breakout_signal = None

        # Manage open position
        if position is not None:
            side = position['side']
            if side == 'LONG':
                if row['low'] <= position['sl']:
                    exit_p = position['sl'] - slippage
                    pnl = (exit_p - position['entry']) * cfg['lot_size'] - cfg['charges_per_lot']
                    trades.append({**position, 'exit': exit_p, 'pnl': pnl,
                                   'result': 'SL', 'exit_ts': ts, 'strategy': 'ORB'})
                    position = None
                elif row['high'] >= position['target']:
                    exit_p = position['target']
                    pnl = (exit_p - position['entry']) * cfg['lot_size'] - cfg['charges_per_lot']
                    trades.append({**position, 'exit': exit_p, 'pnl': pnl,
                                   'result': 'TARGET', 'exit_ts': ts, 'strategy': 'ORB'})
                    position = None
            elif side == 'SHORT':
                if row['high'] >= position['sl']:
                    exit_p = position['sl'] + slippage
                    pnl = (position['entry'] - exit_p) * cfg['lot_size'] - cfg['charges_per_lot']
                    trades.append({**position, 'exit': exit_p, 'pnl': pnl,
                                   'result': 'SL', 'exit_ts': ts, 'strategy': 'ORB'})
                    position = None
                elif row['low'] <= position['target']:
                    exit_p = position['target']
                    pnl = (position['entry'] - exit_p) * cfg['lot_size'] - cfg['charges_per_lot']
                    trades.append({**position, 'exit': exit_p, 'pnl': pnl,
                                   'result': 'TARGET', 'exit_ts': ts, 'strategy': 'ORB'})
                    position = None
            continue  # don't generate new signals while in position

        # Detect breakout (signal only — enter on NEXT bar)
        if not traded_today and breakout_signal is None:
            if row['close'] > or_high:
                breakout_signal = 'LONG'
            elif row['close'] < or_low and not cfg.get('orb_long_only', False):
                breakout_signal = 'SHORT'

    # EOD square-off
    if position is not None and len(trade_data) > 0:
        exit_p = trade_data.iloc[-1]['close'] + (
            -slippage if position['side'] == 'LONG' else slippage)
        mult = 1 if position['side'] == 'LONG' else -1
        pnl = mult * (exit_p - position['entry']) * cfg['lot_size'] - cfg['charges_per_lot']
        trades.append({**position, 'exit': exit_p, 'pnl': pnl,
                       'result': 'EOD', 'exit_ts': trade_data.index[-1], 'strategy': 'ORB'})
    return trades

# =============================================================================
# STRATEGY 2: VWAP SD (IMPROVED)
# =============================================================================

def strategy_vwap_sd(day_df: pd.DataFrame, cfg: dict) -> list:
    """
    VWAP SD with realistic execution:
    - Entry on NEXT bar open after signal
    - Max 1 trade per day
    - Slippage on entry and SL exits
    """
    trades = []
    date_str = str(day_df.index[0].date())
    slippage = cfg['slippage_ticks'] * cfg['tick_size']

    trade_data = day_df[
        (day_df.index.time >= pd.Timestamp(cfg['trade_start']).time()) &
        (day_df.index.time <= pd.Timestamp(cfg['trade_end']).time())
    ].copy()

    if len(trade_data) < 10:
        return []

    position = None
    pending_signal = None  # dict with signal info — enter on next bar
    traded_today = False

    trade_bars = list(trade_data.iterrows())

    for bar_idx, (ts, row) in enumerate(trade_bars):
        if pd.isna(row.get('rsi', np.nan)) or pd.isna(row['vwap']):
            continue

        # Execute pending entry on this bar's open
        if pending_signal and not traded_today:
            sig = pending_signal
            if sig['side'] == 'LONG':
                entry = row['open'] + slippage
            else:
                entry = row['open'] - slippage
            hard_stop_dist = entry * cfg['sd_hard_stop_pct']
            if sig['side'] == 'LONG':
                sl = max(sig['sd3'], entry - hard_stop_dist)
            else:
                sl = min(sig['sd3'], entry + hard_stop_dist)
            position = dict(side=sig['side'], entry=entry, sl=sl,
                            t1=sig['vwap'], t2=sig['t2'],
                            t1_hit=False, entry_ts=ts, date=date_str)
            traded_today = True
            pending_signal = None

        # Manage open position
        if position is not None:
            side = position['side']
            if side == 'LONG':
                if row['low'] <= position['sl']:
                    exit_p = position['sl'] - slippage
                    pnl = (exit_p - position['entry']) * cfg['lot_size'] - cfg['charges_per_lot']
                    trades.append({**position, 'exit': exit_p, 'pnl': pnl,
                                   'result': 'SL', 'exit_ts': ts, 'strategy': 'VWAP_SD'})
                    position = None
                elif not position['t1_hit'] and row['high'] >= position['t1']:
                    position['t1_hit'] = True
                    position['sl'] = position['entry']
                elif position['t1_hit'] and row['high'] >= position['t2']:
                    pnl = (position['t2'] - position['entry']) * cfg['lot_size'] - cfg['charges_per_lot']
                    trades.append({**position, 'exit': position['t2'], 'pnl': pnl,
                                   'result': 'T2', 'exit_ts': ts, 'strategy': 'VWAP_SD'})
                    position = None
            elif side == 'SHORT':
                if row['high'] >= position['sl']:
                    exit_p = position['sl'] + slippage
                    pnl = (position['entry'] - exit_p) * cfg['lot_size'] - cfg['charges_per_lot']
                    trades.append({**position, 'exit': exit_p, 'pnl': pnl,
                                   'result': 'SL', 'exit_ts': ts, 'strategy': 'VWAP_SD'})
                    position = None
                elif not position['t1_hit'] and row['low'] <= position['t1']:
                    position['t1_hit'] = True
                    position['sl'] = position['entry']
                elif position['t1_hit'] and row['low'] <= position['t2']:
                    pnl = (position['entry'] - position['t2']) * cfg['lot_size'] - cfg['charges_per_lot']
                    trades.append({**position, 'exit': position['t2'], 'pnl': pnl,
                                   'result': 'T2', 'exit_ts': ts, 'strategy': 'VWAP_SD'})
                    position = None
            continue  # don't generate signals while in position

        # Generate signal (enter on NEXT bar)
        if traded_today or pending_signal:
            continue
        if row['sd'] < cfg['sd_min_sd_gram']:
            continue

        slope_ok = abs(row.get('vwap_slope_pct', 0)) < cfg['regime_slope_pct'] * 1.5
        vol_ok = row['volume'] >= row.get('vol_avg10', 0) * cfg['sd_vol_mult']

        if slope_ok:
            if (row['low'] <= row['sd2_lower'] and row['close'] > row['sd2_lower'] and
                row['rsi'] < cfg['sd_rsi_long'] and vol_ok):
                pending_signal = dict(side='LONG', vwap=row['vwap'],
                                      t2=row['sd1_upper'], sd3=row['sd3_lower'])
            elif (row['high'] >= row['sd2_upper'] and row['close'] < row['sd2_upper'] and
                  row['rsi'] > cfg['sd_rsi_short'] and vol_ok):
                pending_signal = dict(side='SHORT', vwap=row['vwap'],
                                      t2=row['sd1_lower'], sd3=row['sd3_upper'])

    # EOD square-off
    if position is not None and len(trade_data) > 0:
        exit_p = trade_data.iloc[-1]['close'] + (
            -slippage if position['side'] == 'LONG' else slippage)
        mult = 1 if position['side'] == 'LONG' else -1
        pnl = mult * (exit_p - position['entry']) * cfg['lot_size'] - cfg['charges_per_lot']
        trades.append({**position, 'exit': exit_p, 'pnl': pnl,
                       'result': 'EOD', 'exit_ts': trade_data.index[-1], 'strategy': 'VWAP_SD'})
    return trades

# =============================================================================
# BACKTEST RUNNER
# =============================================================================

def run_backtest(df: pd.DataFrame, mode: str = 'hybrid', cfg: dict = CFG) -> pd.DataFrame:
    all_trades = []
    daily_pnl = {}

    df = add_indicators(df)

    for date, day_df in df.groupby('date_only'):
        day_df = day_df.between_time(cfg['session_start'], cfg['session_end'])
        if len(day_df) < 10:
            continue

        today_pnl = daily_pnl.get(date, 0)
        if today_pnl <= -(cfg['capital'] * cfg['max_daily_loss_pct']):
            continue

        trades = []
        if mode == 'orb':
            trades = strategy_orb(day_df, cfg)
        elif mode == 'vwap_sd':
            trades = strategy_vwap_sd(day_df, cfg)
        elif mode == 'hybrid':
            regime = detect_regime(day_df, cfg)
            if regime == 'TRENDING':
                trades = strategy_orb(day_df, cfg)
            elif regime == 'RANGING':
                trades = strategy_vwap_sd(day_df, cfg)

        all_trades.extend(trades)
        daily_pnl[date] = daily_pnl.get(date, 0) + sum(t['pnl'] for t in trades)

    return pd.DataFrame(all_trades)

# =============================================================================
# STATS + PRINTING
# =============================================================================

def compute_stats(trades_df: pd.DataFrame, label: str = "") -> dict:
    if trades_df.empty:
        return {"label": label, "trades": 0}

    tdf = trades_df.copy()
    tdf['pnl'] = tdf['pnl'].astype(float)
    wins = tdf[tdf['pnl'] > 0]
    losses = tdf[tdf['pnl'] < 0]
    gross_win = wins['pnl'].sum()
    gross_loss = losses['pnl'].sum()
    pf = abs(gross_win / gross_loss) if gross_loss != 0 else float('inf')

    cum = tdf['pnl'].cumsum()
    peak = cum.cummax()
    max_dd = (cum - peak).min()

    tdf['month'] = pd.to_datetime(tdf['date']).dt.to_period('M')
    monthly = tdf.groupby('month')['pnl'].sum()

    streak = tdf['pnl'].apply(lambda x: 1 if x > 0 else -1)
    max_win_streak = max((sum(1 for _ in g) for k, g in itertools.groupby(streak) if k == 1), default=0)
    max_loss_streak = max((sum(1 for _ in g) for k, g in itertools.groupby(streak) if k == -1), default=0)

    tdf['trade_date'] = pd.to_datetime(tdf['date'])
    daily = tdf.groupby('trade_date')['pnl'].sum()
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0

    return {
        "label": label, "trades": len(tdf),
        "wins": len(wins), "losses": len(losses),
        "win_rate": len(wins) / len(tdf) * 100,
        "total_pnl": tdf['pnl'].sum(),
        "avg_win": wins['pnl'].mean() if len(wins) else 0,
        "avg_loss": losses['pnl'].mean() if len(losses) else 0,
        "profit_factor": pf,
        "max_drawdown": max_dd,
        "max_dd_pct": max_dd / CFG['capital'] * 100,
        "sharpe": sharpe,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "sl_count": len(tdf[tdf['result'] == 'SL']),
        "target_count": len(tdf[tdf['result'].isin(['TARGET', 'T2'])]),
        "eod_count": len(tdf[tdf['result'] == 'EOD']),
        "monthly": monthly.to_dict(),
        "avg_pnl_trade": tdf['pnl'].mean(),
        "return_pct": tdf['pnl'].sum() / CFG['capital'] * 100,
    }


def print_stats(stats: dict):
    if stats.get('trades', 0) == 0:
        print(f"\n[{stats['label']}] No trades generated.")
        return

    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  BACKTEST RESULTS — {stats['label']}")
    print(sep)
    print(f"  Total trades     : {stats['trades']}")
    print(f"  Wins             : {stats['wins']}  ({stats['win_rate']:.1f}%)")
    print(f"  Losses           : {stats['losses']}")
    print(f"  Total P&L        : ₹{stats['total_pnl']:>10,.0f}   ({stats['return_pct']:.1f}% on capital)")
    print(f"  Avg win          : ₹{stats['avg_win']:>10,.0f}")
    print(f"  Avg loss         : ₹{stats['avg_loss']:>10,.0f}")
    print(f"  Avg P&L/trade    : ₹{stats['avg_pnl_trade']:>10,.0f}")
    print(f"  Profit Factor    : {stats['profit_factor']:.2f}")
    print(f"  Max Drawdown     : ₹{stats['max_drawdown']:>10,.0f}   ({stats['max_dd_pct']:.1f}%)")
    print(f"  Sharpe Ratio     : {stats['sharpe']:.2f}")
    print(f"  Win streak (max) : {stats['max_win_streak']}")
    print(f"  Loss streak(max) : {stats['max_loss_streak']}")
    print(f"\n  Exit breakdown:")
    print(f"    SL hits        : {stats['sl_count']}")
    print(f"    Target hits    : {stats['target_count']}")
    print(f"    EOD exits      : {stats['eod_count']}")
    print(f"\n  Monthly P&L:")
    for month, pnl in sorted(stats.get('monthly', {}).items(), key=lambda x: str(x[0])):
        bar = "█" * max(1, int(abs(pnl) / 1000))
        sign = "+" if pnl >= 0 else ""
        color = "" if pnl >= 0 else ""
        print(f"    {month}  {sign}₹{pnl:>8,.0f}  {bar}")
    print(sep)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="GOLDM Intraday Backtest")
    parser.add_argument("--strategy", default="all",
                        choices=["orb", "vwap_sd", "hybrid", "all"])
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  GOLDM INTRADAY BACKTEST ENGINE")
    print(f"  Capital : ₹{CFG['capital']:,}  |  Charges: ₹{CFG['charges_per_lot']}/lot")
    print(f"{'='*65}")

    df = load_data(resample='15min')
    print(f"  Loaded  : {len(df):,} bars  |  {df.index[0].date()} → {df.index[-1].date()}")

    strategies = ["orb", "vwap_sd", "hybrid"] if args.strategy == "all" else [args.strategy]
    all_stats = {}

    for strat in strategies:
        trades_df = run_backtest(df, mode=strat)
        stats = compute_stats(trades_df, label=strat.upper())
        all_stats[strat] = stats
        print_stats(stats)

    if len(strategies) > 1:
        print(f"\n{'='*65}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'='*65}")
        print(f"  {'Strategy':<12} {'Trades':>7} {'Win%':>7} {'P&L':>12} {'PF':>6} {'Sharpe':>8} {'MaxDD%':>8}")
        print(f"  {'-'*62}")
        for strat, s in all_stats.items():
            if s.get('trades', 0) == 0:
                print(f"  {strat.upper():<12} {'No trades':>7}")
                continue
            print(f"  {strat.upper():<12} "
                  f"{s['trades']:>7} "
                  f"{s['win_rate']:>6.1f}% "
                  f"₹{s['total_pnl']:>10,.0f} "
                  f"{s['profit_factor']:>6.2f} "
                  f"{s['sharpe']:>8.2f} "
                  f"{s['max_dd_pct']:>7.1f}%")
        print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
