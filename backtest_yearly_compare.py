"""
JTU Original — 11 Year Backtest.
20L capital, 2L per trade, 2 entries/day, Nifty 50, 2015-2025.
"""
import os, sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from data.momentum_backtest import MomentumBacktester


def run_year(bt, strategies, year):
    end = datetime(year, 12, 31)
    result = bt.run_portfolio_backtest(
        period_days=365,
        universe=50,
        capital_lakhs=20,
        per_stock=200000,
        strategies=strategies,
        entries_per_day=2,
        end_date=end,
    )
    if "error" in result:
        return None

    s = result["summary"]
    strat_breakdown = {}
    for t in result["trades"]:
        st = t.get("strategy", "?")
        if st not in strat_breakdown:
            strat_breakdown[st] = {"trades": 0, "pnl": 0, "wins": 0}
        strat_breakdown[st]["trades"] += 1
        strat_breakdown[st]["pnl"] += t.get("pnl", 0)
        if t.get("pnl", 0) > 0:
            strat_breakdown[st]["wins"] += 1
    for st in strat_breakdown:
        b = strat_breakdown[st]
        b["win_rate"] = (b["wins"] / b["trades"] * 100) if b["trades"] > 0 else 0

    return {
        "trades": s["total_trades"],
        "win_rate": s["win_rate"],
        "total_pnl": s["total_pnl"],
        "return_pct": s["total_return_pct"],
        "avg_win": s["avg_win"],
        "avg_loss": s["avg_loss"],
        "profit_factor": s["profit_factor"],
        "breakdown": strat_breakdown,
    }


def main():
    bt = MomentumBacktester()
    years = list(range(2025, 2014, -1))
    results = {}

    for year in years:
        print(f"  [{year}] Running J+T+U...", end=" ", flush=True)
        r = run_year(bt, ["J", "T", "U"], year)
        results[year] = r
        if r:
            print(f"{r['trades']} trades, {r['win_rate']:.1f}% WR, Rs {r['total_pnl']:,.0f}, {r['return_pct']:.2f}%")
        else:
            print("FAILED")

    print(f"\n\n{'='*80}")
    print(f"  J+T+U Original — 11 Year Backtest (20L capital, 2L/trade, 2 trades/day, Nifty 50)")
    print(f"{'='*80}")
    print(f"  {'Year':<6} {'Trades':>7} {'WR%':>7} {'P&L Rs':>13} {'Ret%':>8} {'PF':>6} {'AvgWin':>9} {'AvgLoss':>9}")
    print(f"  {'-'*80}")

    total_pnl = 0
    total_trades = 0
    total_wins = 0
    yearly_rets = []

    for year in years:
        r = results[year]
        if r:
            total_pnl += r["total_pnl"]
            total_trades += r["trades"]
            total_wins += round(r["trades"] * r["win_rate"] / 100)
            yearly_rets.append(r["return_pct"])
            print(f"  {year:<6} {r['trades']:>7} {r['win_rate']:>5.1f}% {r['total_pnl']:>13,.0f} {r['return_pct']:>7.2f}% {r['profit_factor']:>6.2f} {r['avg_win']:>9,.0f} {r['avg_loss']:>9,.0f}")
        else:
            print(f"  {year:<6}  -- no data --")

    print(f"  {'-'*80}")
    avg_ret = sum(yearly_rets) / len(yearly_rets) if yearly_rets else 0
    overall_wr = (total_wins / total_trades * 100) if total_trades else 0
    print(f"  {'Total':<6} {total_trades:>7} {overall_wr:>5.1f}% {total_pnl:>13,.0f} {total_pnl/2000000*100:>7.2f}%")
    print(f"  {'Avg/yr':<6} {total_trades//len(yearly_rets):>7} {'':>7} {total_pnl/len(yearly_rets):>13,.0f} {avg_ret:>7.2f}%")
    win_years = sum(1 for r in yearly_rets if r > 0)
    print(f"\n  Winning years: {win_years}/{len(yearly_rets)}  |  Best: {max(yearly_rets):.2f}%  |  Worst: {min(yearly_rets):.2f}%")

    # Per-strategy breakdown
    print(f"\n  Per-strategy breakdown:")
    print(f"  {'Year':<6} | {'J':^24} | {'T':^24} | {'U':^24}")
    print(f"  {'':6} | {'Tr':>4} {'WR%':>5} {'P&L':>11} | {'Tr':>4} {'WR%':>5} {'P&L':>11} | {'Tr':>4} {'WR%':>5} {'P&L':>11}")
    print(f"  {'-'*6}-+-{'-'*24}-+-{'-'*24}-+-{'-'*24}")
    for year in years:
        r = results[year]
        if r:
            parts = []
            for st in ["J", "T", "U"]:
                b = r["breakdown"].get(st, {"trades": 0, "win_rate": 0, "pnl": 0})
                parts.append(f"{b['trades']:>4} {b['win_rate']:>4.0f}% {b['pnl']:>11,.0f}")
            print(f"  {year:<6} | {parts[0]} | {parts[1]} | {parts[2]}")

    print()


if __name__ == "__main__":
    main()
