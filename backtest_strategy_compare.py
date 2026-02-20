"""
Compare all strategy combos against J+T+U baseline.
20L capital, 2L per trade, 2 entries/day, Nifty 50, last 1 year.

Tests:
  1. J+T+U (baseline)
  2. J+T+U+O
  3. J+T+U+V
  4. J+T+U+O+V (full combo)
  5. V standalone

Goal: find if V (Donchian Breakout) adds value to J+T+U+O.
"""
import os, sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from data.momentum_backtest import MomentumBacktester


def run_combo(bt, strategies, label):
    result = bt.run_portfolio_backtest(
        period_days=365,
        universe=50,
        capital_lakhs=20,
        per_stock=200000,
        strategies=strategies,
        entries_per_day=2,
    )
    if "error" in result:
        return None

    s = result["summary"]

    # Per-strategy breakdown
    strat_counts = {}
    strat_pnl = {}
    strat_wins = {}
    for t in result["trades"]:
        st = t.get("strategy", "?")
        strat_counts[st] = strat_counts.get(st, 0) + 1
        strat_pnl[st] = strat_pnl.get(st, 0) + t.get("pnl", 0)
        if t.get("pnl", 0) > 0:
            strat_wins[st] = strat_wins.get(st, 0) + 1

    breakdown = {}
    for st in sorted(strat_counts.keys()):
        cnt = strat_counts[st]
        pnl = strat_pnl.get(st, 0)
        wins = strat_wins.get(st, 0)
        wr = (wins / cnt * 100) if cnt > 0 else 0
        breakdown[st] = {"trades": cnt, "pnl": pnl, "win_rate": wr}

    return {
        "label": label,
        "strategies": strategies,
        "trades": s["total_trades"],
        "win_rate": s["win_rate"],
        "total_pnl": s["total_pnl"],
        "return_pct": s["total_return_pct"],
        "total_signals": result["total_signals"],
        "missed": result["missed_signals"],
        "max_pos": result["max_positions_used"],
        "breakdown": breakdown,
    }


def main():
    bt = MomentumBacktester()

    combos = [
        (["J", "T", "U"], "J+T+U (baseline)"),
        (["J", "T", "U", "O"], "J+T+U+O"),
        (["J", "T", "U", "V"], "J+T+U+V"),
        (["J", "T", "U", "O", "V"], "J+T+U+O+V"),
        (["V"], "V standalone"),
        (["O"], "O standalone"),
    ]

    results = []
    for i, (strats, label) in enumerate(combos):
        print(f"[{i+1}/{len(combos)}] Running {label}...")
        r = run_combo(bt, strats, label)
        if r:
            results.append(r)
        print(f"  -> {r['trades']} trades, {r['win_rate']:.1f}% WR, Rs {r['total_pnl']:,.0f} P&L, {r['return_pct']:.2f}% return")

    print()
    print("=" * 90)
    print(f"  {'Combo':<22} {'Trades':>6} {'WR%':>6} {'P&L Rs':>12} {'Ret%':>7} {'Signals':>8} {'Missed':>7}")
    print("=" * 90)
    for r in results:
        print(f"  {r['label']:<22} {r['trades']:>6} {r['win_rate']:>5.1f}% {r['total_pnl']:>12,.0f} {r['return_pct']:>6.2f}% {r['total_signals']:>8} {r['missed']:>7}")
    print("=" * 90)

    # Standalone strategies detail
    print()
    print("Standalone strategy details:")
    for r in results:
        if len(r["strategies"]) == 1:
            st = r["strategies"][0]
            b = r["breakdown"].get(st, {})
            print(f"  {st}: {b.get('trades',0)} trades, {b.get('win_rate',0):.0f}% WR, Rs {b.get('pnl',0):,.0f}")

    # Show what each added strategy contributes in combo
    baseline = next((r for r in results if r["label"] == "J+T+U (baseline)"), None)
    if baseline:
        print()
        print("Marginal contribution (vs J+T+U baseline):")
        print(f"  {'Combo':<22} {'dTrades':>8} {'dWR%':>7} {'dP&L Rs':>12} {'dRet%':>8}")
        print("-" * 60)
        for r in results:
            if r["label"] != baseline["label"] and len(r["strategies"]) > 1:
                dt = r["trades"] - baseline["trades"]
                dwr = r["win_rate"] - baseline["win_rate"]
                dpnl = r["total_pnl"] - baseline["total_pnl"]
                dret = r["return_pct"] - baseline["return_pct"]
                sign_pnl = "+" if dpnl >= 0 else ""
                sign_wr = "+" if dwr >= 0 else ""
                sign_ret = "+" if dret >= 0 else ""
                print(f"  {r['label']:<22} {dt:>+8} {sign_wr}{dwr:>5.1f}% {sign_pnl}{dpnl:>11,.0f} {sign_ret}{dret:>6.2f}%")

    # Per-strategy breakdown for the full combo
    full = next((r for r in results if r["label"] == "J+T+U+O+V"), None)
    if full:
        print()
        print("Per-strategy breakdown in J+T+U+O+V:")
        print(f"  {'Strategy':<12} {'Trades':>7} {'WR%':>7} {'P&L Rs':>12}")
        print("-" * 42)
        for st, b in sorted(full["breakdown"].items()):
            print(f"  {st:<12} {b['trades']:>7} {b['win_rate']:>5.1f}% {b['pnl']:>12,.0f}")

    print()


if __name__ == "__main__":
    main()
