"""
Backtest ONLY the T+U overlap signals (same stock, same day).
Uses Strategy T exit rules (Keltner upper) for the trade.
Also runs with U exit rules for comparison.
"""
import os, sys
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf

sys.path.insert(0, os.path.dirname(__file__))
from data.momentum_backtest import MomentumBacktester
from data.momentum_engine import NIFTY_50_TICKERS


def run():
    bt = MomentumBacktester()

    end_date = datetime.now()
    period_days = 365
    daily_start = end_date - timedelta(days=period_days + 500)
    bt_start_date = (end_date - timedelta(days=period_days)).date()

    tickers = NIFTY_50_TICKERS
    total = len(tickers)

    try:
        nifty_raw = yf.Ticker("^NSEI").history(start=daily_start, end=end_date)
    except Exception:
        nifty_raw = pd.DataFrame()

    print(f"Fetching {total} stocks...")
    stock_data = {}
    for idx, ticker in enumerate(tickers):
        print(f"  [{idx+1}/{total}] {ticker}", end="\r")
        try:
            daily = yf.Ticker(f"{ticker}.NS").history(start=daily_start, end=end_date)
        except Exception:
            continue
        if daily.empty or len(daily) < 210:
            continue
        stock_data[ticker] = daily

    print(f"\nLoaded {len(stock_data)} stocks. Computing indicators...")

    indicators = {}
    for ticker, daily in stock_data.items():
        closes = daily["Close"]
        opens = daily["Open"]
        highs = daily["High"]
        lows = daily["Low"]

        hl_range = highs - lows
        ibs_series = ((closes - lows) / hl_range).where(hl_range > 0, 0.5)

        ema20_series = closes.ewm(span=20, adjust=False).mean()
        ema200_series = closes.ewm(span=200, adjust=False).mean()

        rsi14_series = bt._calculate_rsi_series(closes, 14)

        prev_close_s = closes.shift(1)
        tr1 = highs - lows
        tr2 = (highs - prev_close_s).abs()
        tr3 = (lows - prev_close_s).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14_series = true_range.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

        rsi14_min = rsi14_series.rolling(window=14, min_periods=14).min()
        rsi14_max = rsi14_series.rolling(window=14, min_periods=14).max()
        stoch_rsi_raw = (rsi14_series - rsi14_min) / (rsi14_max - rsi14_min)
        stoch_rsi_raw = stoch_rsi_raw.fillna(0.5)
        stoch_k_series = stoch_rsi_raw.rolling(window=3, min_periods=3).mean() * 100
        stoch_d_series = stoch_k_series.rolling(window=3, min_periods=3).mean()

        bt_indices = [i for i, ts in enumerate(daily.index)
                      if ts.date() >= bt_start_date and i >= 200]

        indicators[ticker] = {
            "daily": daily, "closes": closes, "opens": opens,
            "highs": highs, "lows": lows, "ibs": ibs_series,
            "ema20": ema20_series, "ema200": ema200_series,
            "atr14": atr14_series,
            "stoch_k": stoch_k_series, "stoch_d": stoch_d_series,
            "rsi14": rsi14_series,
            "bt_indices": bt_indices,
        }

    # Build date mappings
    all_dates = set()
    date_to_idx = {}
    for ticker, ind in indicators.items():
        mapping = {}
        for i in ind["bt_indices"]:
            d = ind["daily"].index[i].date()
            all_dates.add(d)
            mapping[d] = i
        date_to_idx[ticker] = mapping
    all_dates = sorted(all_dates)

    # Find all T+U overlap entries
    overlap_entries = []  # (date, ticker, entry_price, entry_idx)

    for day in all_dates:
        for ticker, ind in indicators.items():
            if day not in date_to_idx.get(ticker, {}):
                continue

            i = date_to_idx[ticker][day]
            price = float(ind["closes"].iloc[i])
            open_price = float(ind["opens"].iloc[i])
            is_green = price > open_price

            # Check T
            t_fires = False
            ema20_val = float(ind["ema20"].iloc[i])
            atr14_val = float(ind["atr14"].iloc[i]) if not pd.isna(ind["atr14"].iloc[i]) else 0.0
            if atr14_val > 0:
                near_ema20 = abs(price - ema20_val) / ema20_val <= 0.01
                was_at_upper = False
                for lb in range(max(0, i - 10), i):
                    past_high = float(ind["highs"].iloc[lb])
                    past_ema20 = float(ind["ema20"].iloc[lb])
                    past_atr14 = float(ind["atr14"].iloc[lb]) if not pd.isna(ind["atr14"].iloc[lb]) else 0.0
                    if past_high >= past_ema20 + 2 * past_atr14:
                        was_at_upper = True
                        break
                if near_ema20 and was_at_upper and is_green:
                    t_fires = True

            # Check U
            u_fires = False
            stk = float(ind["stoch_k"].iloc[i]) if not pd.isna(ind["stoch_k"].iloc[i]) else 50.0
            std = float(ind["stoch_d"].iloc[i]) if not pd.isna(ind["stoch_d"].iloc[i]) else 50.0
            stk_prev = float(ind["stoch_k"].iloc[i-1]) if i > 0 and not pd.isna(ind["stoch_k"].iloc[i-1]) else 50.0
            std_prev = float(ind["stoch_d"].iloc[i-1]) if i > 0 and not pd.isna(ind["stoch_d"].iloc[i-1]) else 50.0
            ema200_val = float(ind["ema200"].iloc[i])
            k_cross_up = (stk_prev <= std_prev and stk > std)
            if k_cross_up and stk < 30 and std < 30 and price > ema200_val:
                u_fires = True

            if t_fires and u_fires:
                overlap_entries.append({
                    "date": day,
                    "ticker": ticker,
                    "entry_price": price,
                    "entry_idx": i,
                })

    print(f"\nFound {len(overlap_entries)} T+U overlap entries. Simulating trades...\n")

    # Simulate each overlap entry with BOTH exit rule sets
    def simulate_trade(entry, exit_mode):
        """exit_mode: 'T' (Keltner upper), 'U' (StochRSI overbought), 'combined' (first to fire)"""
        ticker = entry["ticker"]
        ind = indicators[ticker]
        entry_price = entry["entry_price"]
        entry_idx = entry["entry_idx"]
        daily = ind["daily"]
        max_idx = len(daily) - 1

        partial_done = False
        shares = int(200000 // entry_price)
        remaining = shares
        total_pnl = 0

        for j in range(entry_idx + 1, max_idx + 1):
            price = float(ind["closes"].iloc[j])
            high = float(ind["highs"].iloc[j])

            # 5% hard SL
            if price <= entry_price * 0.95:
                pnl = (price - entry_price) * remaining
                return {
                    "exit_reason": "HARD_SL_5PCT",
                    "exit_date": daily.index[j].date(),
                    "pnl_pct": round(((price - entry_price) / entry_price) * 100, 2),
                    "pnl_amount": round(total_pnl + pnl, 0),
                    "hold_days": (daily.index[j].date() - entry["date"]).days,
                    "win": False,
                }

            # +5% partial
            if not partial_done and price >= entry_price * 1.05:
                half = shares // 2
                total_pnl += (price - entry_price) * half
                remaining = shares - half
                partial_done = True

            # Exit conditions based on mode
            should_exit = False
            exit_reason = ""

            if exit_mode in ("T", "combined"):
                ema20_val = float(ind["ema20"].iloc[j])
                atr14_val = float(ind["atr14"].iloc[j]) if not pd.isna(ind["atr14"].iloc[j]) else 0.0
                upper_keltner = ema20_val + 2 * atr14_val
                if price >= upper_keltner:
                    should_exit = True
                    exit_reason = "KELTNER_UPPER"

            if not should_exit and exit_mode in ("U", "combined"):
                stk = float(ind["stoch_k"].iloc[j]) if not pd.isna(ind["stoch_k"].iloc[j]) else 50.0
                std_val = float(ind["stoch_d"].iloc[j]) if not pd.isna(ind["stoch_d"].iloc[j]) else 50.0
                stk_prev = float(ind["stoch_k"].iloc[j-1]) if j > 0 and not pd.isna(ind["stoch_k"].iloc[j-1]) else 50.0
                std_prev = float(ind["stoch_d"].iloc[j-1]) if j > 0 and not pd.isna(ind["stoch_d"].iloc[j-1]) else 50.0
                k_cross_dn = (stk_prev >= std_prev and stk < std_val)
                if k_cross_dn and stk > 70 and std_val > 70:
                    should_exit = True
                    exit_reason = "STOCHRSI_OB"

            if should_exit:
                pnl = (price - entry_price) * remaining
                total_pnl_final = total_pnl + pnl
                pnl_pct = round(((price - entry_price) / entry_price) * 100, 2)
                return {
                    "exit_reason": exit_reason,
                    "exit_date": daily.index[j].date(),
                    "pnl_pct": pnl_pct,
                    "pnl_amount": round(total_pnl_final, 0),
                    "hold_days": (daily.index[j].date() - entry["date"]).days,
                    "win": total_pnl_final > 0,
                }

        # Still open at end
        last_price = float(ind["closes"].iloc[max_idx])
        pnl = (last_price - entry_price) * remaining
        total_pnl_final = total_pnl + pnl
        pnl_pct = round(((last_price - entry_price) / entry_price) * 100, 2)
        return {
            "exit_reason": "STILL_OPEN",
            "exit_date": daily.index[max_idx].date(),
            "pnl_pct": pnl_pct,
            "pnl_amount": round(total_pnl_final, 0),
            "hold_days": (daily.index[max_idx].date() - entry["date"]).days,
            "win": total_pnl_final > 0,
        }

    # Run all three exit modes
    for mode, label in [("T", "T exits (Keltner upper)"),
                         ("U", "U exits (StochRSI overbought)"),
                         ("combined", "Combined (first T or U exit)")]:
        results = []
        for entry in overlap_entries:
            r = simulate_trade(entry, mode)
            r["ticker"] = entry["ticker"]
            r["entry_date"] = entry["date"]
            r["entry_price"] = entry["entry_price"]
            results.append(r)

        wins = sum(1 for r in results if r["win"])
        losses = len(results) - wins
        total_pnl = sum(r["pnl_amount"] for r in results)
        avg_pnl_pct = np.mean([r["pnl_pct"] for r in results]) if results else 0
        avg_hold = np.mean([r["hold_days"] for r in results]) if results else 0
        win_pcts = [r["pnl_pct"] for r in results if r["win"]]
        loss_pcts = [r["pnl_pct"] for r in results if not r["win"]]
        avg_win = np.mean(win_pcts) if win_pcts else 0
        avg_loss = np.mean(loss_pcts) if loss_pcts else 0

        print(f"{'=' * 60}")
        print(f"  EXIT MODE: {label}")
        print(f"{'=' * 60}")
        print(f"  Trades       : {len(results)}")
        print(f"  Winners      : {wins}")
        print(f"  Losers       : {losses}")
        print(f"  Win rate     : {wins/len(results)*100:.1f}%" if results else "  Win rate     : N/A")
        print(f"  Total P&L    : Rs {total_pnl:,.0f}")
        print(f"  Avg P&L %    : {avg_pnl_pct:.2f}%")
        print(f"  Avg win %    : +{avg_win:.2f}%")
        print(f"  Avg loss %   : {avg_loss:.2f}%")
        print(f"  Avg hold     : {avg_hold:.0f} days")
        print()

        # Show each trade
        print(f"  {'Date':<12} {'Ticker':<14} {'Entry':>8} {'P&L%':>7} {'Rs P&L':>10} {'Days':>5} {'Exit Reason'}")
        print(f"  {'-'*12} {'-'*14} {'-'*8} {'-'*7} {'-'*10} {'-'*5} {'-'*20}")
        for r in sorted(results, key=lambda x: x["entry_date"]):
            pnl_sign = "+" if r["pnl_pct"] >= 0 else ""
            print(f"  {str(r['entry_date']):<12} {r['ticker']:<14} {r['entry_price']:>8.1f} {pnl_sign}{r['pnl_pct']:>6.1f}% {r['pnl_amount']:>10,.0f} {r['hold_days']:>5} {r['exit_reason']}")
        print()

    # Compare: T-only signals (no U), U-only (no T), overlap
    print("=" * 60)
    print("  COMPARISON: Overlap vs T-only vs U-only (combined exits)")
    print("=" * 60)

    # Collect T-only and U-only entries
    overlap_set = {(e["date"], e["ticker"]) for e in overlap_entries}

    t_only_entries = []
    u_only_entries = []

    for day in all_dates:
        for ticker, ind in indicators.items():
            if day not in date_to_idx.get(ticker, {}):
                continue

            i = date_to_idx[ticker][day]
            price = float(ind["closes"].iloc[i])
            open_price = float(ind["opens"].iloc[i])
            is_green = price > open_price

            t_fires = False
            ema20_val = float(ind["ema20"].iloc[i])
            atr14_val = float(ind["atr14"].iloc[i]) if not pd.isna(ind["atr14"].iloc[i]) else 0.0
            if atr14_val > 0:
                near_ema20 = abs(price - ema20_val) / ema20_val <= 0.01
                was_at_upper = False
                for lb in range(max(0, i - 10), i):
                    past_high = float(ind["highs"].iloc[lb])
                    past_ema20 = float(ind["ema20"].iloc[lb])
                    past_atr14 = float(ind["atr14"].iloc[lb]) if not pd.isna(ind["atr14"].iloc[lb]) else 0.0
                    if past_high >= past_ema20 + 2 * past_atr14:
                        was_at_upper = True
                        break
                if near_ema20 and was_at_upper and is_green:
                    t_fires = True

            u_fires = False
            stk = float(ind["stoch_k"].iloc[i]) if not pd.isna(ind["stoch_k"].iloc[i]) else 50.0
            std = float(ind["stoch_d"].iloc[i]) if not pd.isna(ind["stoch_d"].iloc[i]) else 50.0
            stk_prev = float(ind["stoch_k"].iloc[i-1]) if i > 0 and not pd.isna(ind["stoch_k"].iloc[i-1]) else 50.0
            std_prev = float(ind["stoch_d"].iloc[i-1]) if i > 0 and not pd.isna(ind["stoch_d"].iloc[i-1]) else 50.0
            ema200_val = float(ind["ema200"].iloc[i])
            k_cross_up = (stk_prev <= std_prev and stk > std)
            if k_cross_up and stk < 30 and std < 30 and price > ema200_val:
                u_fires = True

            key = (day, ticker)
            if t_fires and not u_fires:
                t_only_entries.append({"date": day, "ticker": ticker, "entry_price": price, "entry_idx": i})
            if u_fires and not t_fires:
                u_only_entries.append({"date": day, "ticker": ticker, "entry_price": price, "entry_idx": i})

    # Simulate T-only and U-only with combined exits
    for entries, label in [(t_only_entries, "T-only (no U)"),
                            (u_only_entries, "U-only (no T)"),
                            (overlap_entries, "T+U overlap")]:
        results = [simulate_trade(e, "combined") for e in entries]
        wins = sum(1 for r in results if r["win"])
        total_pnl = sum(r["pnl_amount"] for r in results)
        wr = (wins / len(results) * 100) if results else 0
        avg_pnl = np.mean([r["pnl_pct"] for r in results]) if results else 0
        print(f"  {label:<20}: {len(results):>4} trades, {wr:>5.1f}% win rate, avg {avg_pnl:>+6.2f}%, Rs {total_pnl:>10,.0f}")

    print()


if __name__ == "__main__":
    run()
