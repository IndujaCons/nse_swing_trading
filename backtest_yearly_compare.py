"""
JT 3-Stage vs JT Original — 11 Year Comparison.
20L capital, 2L per trade, 2 entries/day, Nifty 100, 2015-2025.
Usage: python3 backtest_yearly_compare.py [seed]
"""
import os, sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from data.momentum_backtest import MomentumBacktester


def run_year(bt, strategies, year, three_stage=True, seed=42):
    end = datetime(year, 12, 31)
    result = bt.run_portfolio_backtest(
        period_days=365,
        universe=100,
        capital_lakhs=20,
        per_stock=200000,
        strategies=strategies,
        entries_per_day=2,
        end_date=end,
        three_stage_exit=three_stage,
        seed=seed,
    )
    if "error" in result:
        return None

    s = result["summary"]
    return {
        "trades": s["total_trades"],
        "win_rate": s["win_rate"],
        "total_pnl": s["total_pnl"],
        "return_pct": s["total_return_pct"],
        "avg_win": s["avg_win"],
        "avg_loss": s["avg_loss"],
        "profit_factor": s["profit_factor"],
    }


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    print(f"\n  Seed: {seed}\n")

    bt = MomentumBacktester()
    years = list(range(2025, 2014, -1))
    jt3_results = {}
    jt_results = {}

    for year in years:
        print(f"  [{year}] JT(3-Stage)...", end=" ", flush=True)
        r = run_year(bt, ["J", "T"], year, three_stage=True, seed=seed)
        jt3_results[year] = r
        if r:
            print(f"{r['trades']}tr {r['win_rate']:.1f}%WR Rs{r['total_pnl']:,.0f}", end="  |  ")
        else:
            print("FAILED", end="  |  ")

        print(f"JT(Original)...", end=" ", flush=True)
        r = run_year(bt, ["J", "T"], year, three_stage=False, seed=seed)
        jt_results[year] = r
        if r:
            print(f"{r['trades']}tr {r['win_rate']:.1f}%WR Rs{r['total_pnl']:,.0f}")
        else:
            print("FAILED")

    print(f"\n\n{'='*120}")
    print(f"  JT 3-Stage vs JT Original — Nifty 100, 11 Years, 20L capital, 2L/trade, 2/day, seed={seed}")
    print(f"{'='*120}")
    print(f"  {'Year':<6} | {'--- JT (3-Stage) ---':^42} | {'--- JT (Original) ---':^42} | {'dPnL':>12}")
    print(f"  {'':6} | {'WR%':>6} {'Ret%':>7} {'P&L':>12} {'AvgWin':>8} {'AvgLoss':>9} | {'WR%':>6} {'Ret%':>7} {'P&L':>12} {'AvgWin':>8} {'AvgLoss':>9} |")
    print(f"  {'-'*120}")

    a_pnl = a_tr = a_wins = 0
    b_pnl = b_tr = b_wins = 0
    a_rets = []
    b_rets = []

    for year in years:
        a = jt3_results.get(year)
        b = jt_results.get(year)
        if a and b:
            a_pnl += a["total_pnl"]; a_tr += a["trades"]; a_wins += round(a["trades"] * a["win_rate"] / 100); a_rets.append(a["return_pct"])
            b_pnl += b["total_pnl"]; b_tr += b["trades"]; b_wins += round(b["trades"] * b["win_rate"] / 100); b_rets.append(b["return_pct"])
            dpnl = a["total_pnl"] - b["total_pnl"]
            print(f"  {year:<6} | {a['win_rate']:>5.1f}% {a['return_pct']:>6.2f}% {a['total_pnl']:>12,.0f} {a['avg_win']:>8,.0f} {a['avg_loss']:>9,.0f} | {b['win_rate']:>5.1f}% {b['return_pct']:>6.2f}% {b['total_pnl']:>12,.0f} {b['avg_win']:>8,.0f} {b['avg_loss']:>9,.0f} | {dpnl:>+12,.0f}")

    print(f"  {'-'*120}")
    a_wr = (a_wins / a_tr * 100) if a_tr else 0
    b_wr = (b_wins / b_tr * 100) if b_tr else 0
    print(f"  {'Total':<6} | {a_wr:>5.1f}% {a_pnl/2000000*100:>6.2f}% {a_pnl:>12,.0f} {'':>8} {'':>9} | {b_wr:>5.1f}% {b_pnl/2000000*100:>6.2f}% {b_pnl:>12,.0f}")
    a_avg_ret = sum(a_rets) / len(a_rets) if a_rets else 0
    b_avg_ret = sum(b_rets) / len(b_rets) if b_rets else 0
    print(f"  {'Avg/yr':<6} | {'':>6} {a_avg_ret:>6.2f}% {a_pnl/len(a_rets):>12,.0f} {'':>8} {'':>9} | {'':>6} {b_avg_ret:>6.2f}% {b_pnl/len(b_rets):>12,.0f}")
    a_win_yrs = sum(1 for r in a_rets if r > 0)
    b_win_yrs = sum(1 for r in b_rets if r > 0)
    print(f"\n  JT(3-Stage): {a_win_yrs}/{len(a_rets)} winning yrs | Best: {max(a_rets):.2f}% | Worst: {min(a_rets):.2f}%")
    print(f"  JT(Original): {b_win_yrs}/{len(b_rets)} winning yrs | Best: {max(b_rets):.2f}% | Worst: {min(b_rets):.2f}%")
    print()


if __name__ == "__main__":
    main()
