"""
Compare Strategy J portfolio backtest: WITH vs WITHOUT the 3 new filters.
Filters: (1) wick >5% excluded, (2) support-declining skip, (3) IBS 0.6 vs 0.5
Period: Jan 1 2025 - Dec 30 2025, Nifty 50, J-only, 10L capital, 50K/trade
"""
import sys, importlib, shutil, os, json
sys.path.insert(0, "/Users/jay/Desktop/relative_strength")

from datetime import datetime

BACKTEST_FILE = "/Users/jay/Desktop/relative_strength/data/momentum_backtest.py"
BACKUP_FILE = BACKTEST_FILE + ".comparison_bak"

END_DATE = datetime(2025, 12, 30)
PERIOD_DAYS = (END_DATE - datetime(2025, 1, 1)).days


def run_backtest(label):
    """Run J-only portfolio backtest for 2025."""
    # Force reload the module to pick up file changes
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith('data'):
            del sys.modules[mod_name]

    from data.momentum_backtest import MomentumBacktester

    print(f"  [{label}] Running Jan 2025 - Dec 2025...", flush=True)
    bt = MomentumBacktester()
    result = bt.run_portfolio_backtest(
        PERIOD_DAYS, universe=50, capital_lakhs=10, per_stock=50000,
        strategies=["J"], entries_per_day=1, end_date=END_DATE
    )

    if "error" in result:
        print(f"  ERROR: {result['error']}", flush=True)
        return {"signals": 0, "wins": 0, "losses": 0, "win_rate": 0, "total_pnl": 0, "trades": []}

    trades = result.get("trades", [])
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total_pnl = sum(t["pnl"] for t in trades)
    win_rate = round(len(wins) / len(trades) * 100, 1) if trades else 0

    print(f"  Trades={len(trades)}, Wins={len(wins)}, Losses={len(losses)}, "
          f"WR={win_rate}%, PnL={total_pnl:.0f}", flush=True)

    return {
        "signals": len(trades), "wins": len(wins), "losses": len(losses),
        "win_rate": win_rate, "total_pnl": round(total_pnl, 2),
        "trades": trades
    }


def revert_j_filters():
    """Remove the 3 J filters from the portfolio backtest section."""
    with open(BACKTEST_FILE, 'r') as f:
        src = f.read()

    # --- Revert 1: Remove wick filter from portfolio support calc ---
    src = src.replace(
        """            # Exclude huge-wick weeks (close - low > 5% of close) from support calc
            wick_pct = (weekly["Close"] - weekly["Low"]) / weekly["Close"]
            valid_week = wick_pct <= 0.05
            filtered_close = weekly["Close"].where(valid_week)
            filtered_low = weekly["Low"].where(valid_week)
            w_support = filtered_close.rolling(window=26, min_periods=13).min()
            weekly_support_series = w_support.reindex(daily.index, method="ffill")
            weekly_support_prev_series = w_support.shift(1).reindex(daily.index, method="ffill")
            w_low_stop = filtered_low.rolling(window=26, min_periods=13).min()
            weekly_low_stop_series = w_low_stop.reindex(daily.index, method="ffill")""",
        """            w_support = weekly["Close"].rolling(window=26, min_periods=26).min()
            weekly_support_series = w_support.reindex(daily.index, method="ffill")
            w_low_stop = weekly["Low"].rolling(window=26, min_periods=26).min()
            weekly_low_stop_series = w_low_stop.reindex(daily.index, method="ffill")"""
    )

    # --- Revert 2: Remove weekly_support_prev from indicators dict ---
    src = src.replace(
        '                "weekly_support_prev": weekly_support_prev_series,\n', ''
    )

    # --- Revert 3: Remove declining filter + support_prev + IBS back to 0.5 ---
    src = src.replace(
        """                if "J" in strategies:
                    w_support = ind["weekly_support"]
                    w_support_prev = ind["weekly_support_prev"]
                    w_low_stop = ind["weekly_low_stop"]
                    if w_support is not None and not pd.isna(w_support.iloc[i]):
                        ws = float(w_support.iloc[i])
                        wsp = float(w_support_prev.iloc[i]) if w_support_prev is not None and not pd.isna(w_support_prev.iloc[i]) else ws
                        wls = float(w_low_stop.iloc[i]) if w_low_stop is not None and not pd.isna(w_low_stop.iloc[i]) else ws
                        if ws > 0 and ws >= wsp:  # support must not be declining
                            close_near = ((price - ws) / ws) * 100
                            if (close_near >= 0 and close_near <= 3.0
                                    and ibs > 0.6 and is_green):""",
        """                if "J" in strategies:
                    w_support = ind["weekly_support"]
                    w_low_stop = ind["weekly_low_stop"]
                    if w_support is not None and not pd.isna(w_support.iloc[i]):
                        ws = float(w_support.iloc[i])
                        wls = float(w_low_stop.iloc[i]) if w_low_stop is not None and not pd.isna(w_low_stop.iloc[i]) else ws
                        if ws > 0:
                            close_near = ((price - ws) / ws) * 100
                            if (close_near >= 0 and close_near <= 3.0
                                    and ibs > 0.5 and is_green):"""
    )

    with open(BACKTEST_FILE, 'w') as f:
        f.write(src)
    print("  [Reverted J filters in portfolio section]", flush=True)


def print_trades(label, trades):
    """Print trade list."""
    print(f"\n  --- {label} Trades ---")
    print(f"  {'Entry':<12} {'Exit':<12} {'Stock':<12} {'EntPx':>8} {'ExPx':>8} {'PnL':>10} {'Reason':<22}")
    print(f"  {'-'*90}")
    for t in trades:
        print(f"  {t['entry_date']:<12} {t['exit_date']:<12} {t.get('symbol',''):>12} "
              f"{t['entry_price']:>8.2f} {t['exit_price']:>8.2f} {t['pnl']:>10.2f} {t.get('exit_reason',''):<22}")


if __name__ == "__main__":
    shutil.copy2(BACKTEST_FILE, BACKUP_FILE)

    try:
        # Phase 1: WITH filters (current code)
        print("=" * 70, flush=True)
        print("PHASE 1: WITH filters (wick, declining, IBS 0.6)", flush=True)
        print("=" * 70, flush=True)
        with_res = run_backtest("WITH")
        print_trades("WITH", with_res["trades"])

        # Phase 2: Revert filters
        print("\n" + "=" * 70, flush=True)
        print("Reverting J filters...", flush=True)
        revert_j_filters()

        # Phase 3: WITHOUT filters (original code)
        print("=" * 70, flush=True)
        print("PHASE 2: WITHOUT filters (original IBS 0.5, no wick/declining)", flush=True)
        print("=" * 70, flush=True)
        without_res = run_backtest("WITHOUT")
        print_trades("WITHOUT", without_res["trades"])

        # Phase 4: Comparison
        print("\n" + "=" * 70)
        print("COMPARISON: Jan 2025 - Dec 2025 (Strategy J, Nifty 50)")
        print("=" * 70)
        print(f"{'Metric':<25} {'WITH filters':>15} {'WITHOUT filters':>15} {'Delta':>12}")
        print("-" * 70)
        print(f"{'Signals/Trades':<25} {with_res['signals']:>15} {without_res['signals']:>15} {with_res['signals']-without_res['signals']:>+12}")
        print(f"{'Wins':<25} {with_res['wins']:>15} {without_res['wins']:>15} {with_res['wins']-without_res['wins']:>+12}")
        print(f"{'Losses':<25} {with_res['losses']:>15} {without_res['losses']:>15} {with_res['losses']-without_res['losses']:>+12}")
        print(f"{'Win Rate':<25} {str(with_res['win_rate'])+'%':>15} {str(without_res['win_rate'])+'%':>15}")
        print(f"{'Total P&L':<25} {with_res['total_pnl']:>15.0f} {without_res['total_pnl']:>15.0f} {with_res['total_pnl']-without_res['total_pnl']:>+12.0f}")
        print("=" * 70)

        # Show signals that were filtered out
        with_entries = set((t['entry_date'], t.get('symbol','')) for t in with_res['trades'])
        without_entries = set((t['entry_date'], t.get('symbol','')) for t in without_res['trades'])
        filtered_out = without_entries - with_entries
        if filtered_out:
            print(f"\nSignals FILTERED OUT by new filters ({len(filtered_out)}):")
            for entry_date, symbol in sorted(filtered_out):
                t = next((t for t in without_res['trades']
                          if t['entry_date'] == entry_date and t.get('symbol','') == symbol), None)
                if t:
                    print(f"  {entry_date} {symbol:<12} PnL={t['pnl']:>+10.2f} ({t.get('exit_reason','')})")

        new_entries = with_entries - without_entries
        if new_entries:
            print(f"\nNEW signals from filter changes ({len(new_entries)}):")
            for entry_date, symbol in sorted(new_entries):
                t = next((t for t in with_res['trades']
                          if t['entry_date'] == entry_date and t.get('symbol','') == symbol), None)
                if t:
                    print(f"  {entry_date} {symbol:<12} PnL={t['pnl']:>+10.2f} ({t.get('exit_reason','')})")

    finally:
        shutil.copy2(BACKUP_FILE, BACKTEST_FILE)
        os.remove(BACKUP_FILE)
        print(f"\nRestored original file.", flush=True)
