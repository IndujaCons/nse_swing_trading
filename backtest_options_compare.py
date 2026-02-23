"""
Baseline (UW 10d) vs Option 1 (Tighten SL to 4% after first +6% exit)
11-year comparison: 2015-2025, 20L capital, 2L/trade, 2 entries/day, Nifty 100.
Usage: python3 backtest_options_compare.py [seed]
"""
import os, sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from data.momentum_backtest import MomentumBacktester


def run_year(bt, year, seed=42, **kwargs):
    end = datetime(year, 12, 31)
    result = bt.run_portfolio_backtest(
        period_days=365,
        universe=100,
        capital_lakhs=20,
        per_stock=200000,
        strategies=["J", "T"],
        entries_per_day=2,
        end_date=end,
        three_stage_exit=True,
        seed=seed,
        underwater_exit_days=10,
        **kwargs,
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

    configs = [
        ("Baseline (UW 10d)", {}),
        ("+ T SL 4% after 1st", {"t_tight_sl": 0.04}),
    ]

    all_results = {name: {} for name, _ in configs}

    for year in years:
        for name, kwargs in configs:
            print(f"  [{year}] {name}...", end=" ", flush=True)
            r = run_year(bt, year, seed=seed, **kwargs)
            all_results[name][year] = r
            if r:
                print(f"{r['trades']}tr {r['win_rate']:.1f}%WR Rs{r['total_pnl']:,.0f}", end="  ")
            else:
                print("FAILED", end="  ")
        print()

    # === Display comparison table ===
    col_w = 48
    total_w = 8 + (col_w + 3) * 2 + 14
    print(f"\n{'=' * total_w}")
    print(f"  Baseline (UW 10d) vs + Tighten T SL to 4% after first +6% exit")
    print(f"  Nifty 100, 11 Years, 20L capital, 2L/trade, 2/day, seed={seed}")
    print(f"{'=' * total_w}")

    hdr = f"  {'Year':<6}"
    for name, _ in configs:
        hdr += f" | {name:^{col_w}}"
    hdr += f" | {'Delta':>12}"
    print(hdr)

    sub = f"  {'':6}"
    for _ in configs:
        sub += f" | {'Tr':>4} {'WR%':>6} {'Ret%':>7} {'P&L':>12} {'AvgLoss':>9} {'PF':>5}"
    sub += f" | {'':>12}"
    print(sub)
    print(f"  {'-' * (total_w - 2)}")

    acc = {name: {"pnl": 0, "tr": 0, "wins": 0, "rets": []} for name, _ in configs}

    for year in years:
        line = f"  {year:<6}"
        for name, _ in configs:
            r = all_results[name].get(year)
            if r:
                a = acc[name]
                a["pnl"] += r["total_pnl"]
                a["tr"] += r["trades"]
                a["wins"] += round(r["trades"] * r["win_rate"] / 100)
                a["rets"].append(r["return_pct"])
                line += f" | {r['trades']:>4} {r['win_rate']:>5.1f}% {r['return_pct']:>6.2f}% {r['total_pnl']:>12,.0f} {r['avg_loss']:>9,.0f} {r['profit_factor']:>5.2f}"
            else:
                line += f" | {'FAILED':^{col_w}}"
        b = all_results[configs[0][0]].get(year)
        u = all_results[configs[1][0]].get(year)
        if b and u:
            d = u["total_pnl"] - b["total_pnl"]
            line += f" | {d:>+12,.0f}"
        print(line)

    print(f"  {'-' * (total_w - 2)}")

    line = f"  {'Total':<6}"
    for name, _ in configs:
        a = acc[name]
        wr = (a["wins"] / a["tr"] * 100) if a["tr"] else 0
        ret = a["pnl"] / 2000000 * 100
        line += f" | {a['tr']:>4} {wr:>5.1f}% {ret:>6.2f}% {a['pnl']:>12,.0f} {'':>9} {'':>5}"
    dpnl = acc[configs[1][0]]["pnl"] - acc[configs[0][0]]["pnl"]
    line += f" | {dpnl:>+12,.0f}"
    print(line)

    line = f"  {'Avg/yr':<6}"
    for name, _ in configs:
        a = acc[name]
        avg_ret = sum(a["rets"]) / len(a["rets"]) if a["rets"] else 0
        avg_pnl = a["pnl"] / len(a["rets"]) if a["rets"] else 0
        line += f" | {'':>4} {'':>6} {avg_ret:>6.2f}% {avg_pnl:>12,.0f} {'':>9} {'':>5}"
    print(line)

    print()
    for name, _ in configs:
        a = acc[name]
        rets = a["rets"]
        if rets:
            win_yrs = sum(1 for r in rets if r > 0)
            print(f"  {name}: {win_yrs}/{len(rets)} winning yrs | Best: {max(rets):.2f}% | Worst: {min(rets):.2f}% | Total P&L: Rs{a['pnl']:,.0f}")
    print()


if __name__ == "__main__":
    main()
