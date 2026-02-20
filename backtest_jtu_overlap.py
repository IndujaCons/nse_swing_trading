"""
J + T + U Portfolio Backtest with Overlap Analysis
Capital: 20L, Per trade: 2L, 2 entries/day, Nifty 50, Last 1 year

Measures how often T and U signals fire on the SAME stock on the SAME day.
"""
import os, sys, json
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf

sys.path.insert(0, os.path.dirname(__file__))
from data.momentum_backtest import MomentumBacktester
from data.momentum_engine import NIFTY_50_TICKERS


def run_overlap_analysis():
    bt = MomentumBacktester()

    end_date = datetime.now()
    period_days = 365
    daily_start = end_date - timedelta(days=period_days + 500)
    bt_start_date = (end_date - timedelta(days=period_days)).date()

    tickers = NIFTY_50_TICKERS
    total = len(tickers)

    # Fetch Nifty
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

    # Pre-compute indicators (same as backtest engine)
    indicators = {}
    for ticker, daily in stock_data.items():
        closes = daily["Close"]
        opens = daily["Open"]
        highs = daily["High"]
        lows = daily["Low"]

        hl_range = highs - lows
        ibs_series = ((closes - lows) / hl_range).where(hl_range > 0, 0.5)

        # J: weekly support
        weekly = daily.resample("W-FRI").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum"
        }).dropna()
        w_support = weekly["Close"].rolling(window=26, min_periods=26).min()
        weekly_support_series = w_support.reindex(daily.index, method="ffill")
        w_low_stop = weekly["Low"].rolling(window=26, min_periods=26).min()
        weekly_low_stop_series = w_low_stop.reindex(daily.index, method="ffill")

        # CCI(20)
        cci20_series = bt._calculate_cci_series(highs, lows, closes, 20)

        # EMAs
        ema20_series = closes.ewm(span=20, adjust=False).mean()
        ema200_series = closes.ewm(span=200, adjust=False).mean()

        # RSI(14) for stoch RSI
        rsi14_series = bt._calculate_rsi_series(closes, 14)

        # ATR(14) for Keltner
        prev_close_s = closes.shift(1)
        tr1 = highs - lows
        tr2 = (highs - prev_close_s).abs()
        tr3 = (lows - prev_close_s).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14_series = true_range.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

        # Stochastic RSI
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
            "weekly_support": weekly_support_series,
            "weekly_low_stop": weekly_low_stop_series,
            "cci20": cci20_series,
            "ema20": ema20_series, "ema200": ema200_series,
            "atr14": atr14_series,
            "stoch_k": stoch_k_series, "stoch_d": stoch_d_series,
            "bt_indices": bt_indices,
        }

    # Build all trading dates
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

    print(f"Scanning {len(all_dates)} trading days for J / T / U signals...\n")

    # Track signals WITHOUT dedup — raw count per strategy
    j_total = 0
    t_total = 0
    u_total = 0
    t_and_u_overlap = 0  # same stock, same day
    j_and_t_overlap = 0
    j_and_u_overlap = 0
    all_three_overlap = 0

    overlap_details = []  # (date, ticker) where T+U both fire

    for day in all_dates:
        day_j = set()
        day_t = set()
        day_u = set()

        for ticker, ind in indicators.items():
            if day not in date_to_idx.get(ticker, {}):
                continue

            i = date_to_idx[ticker][day]
            price = float(ind["closes"].iloc[i])
            open_price = float(ind["opens"].iloc[i])
            is_green = price > open_price
            ibs = float(ind["ibs"].iloc[i])

            # ---- Strategy J ----
            w_support = ind["weekly_support"]
            if not pd.isna(w_support.iloc[i]):
                ws = float(w_support.iloc[i])
                if ws > 0:
                    close_near = ((price - ws) / ws) * 100
                    cci_val = float(ind["cci20"].iloc[i]) if not pd.isna(ind["cci20"].iloc[i]) else 0.0
                    if (close_near >= 0 and close_near <= 3.0
                            and ibs > 0.5 and is_green
                            and cci_val > -100):
                        day_j.add(ticker)

            # ---- Strategy T ----
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
                    day_t.add(ticker)

            # ---- Strategy U ----
            stk = float(ind["stoch_k"].iloc[i]) if not pd.isna(ind["stoch_k"].iloc[i]) else 50.0
            std = float(ind["stoch_d"].iloc[i]) if not pd.isna(ind["stoch_d"].iloc[i]) else 50.0
            stk_prev = float(ind["stoch_k"].iloc[i-1]) if i > 0 and not pd.isna(ind["stoch_k"].iloc[i-1]) else 50.0
            std_prev = float(ind["stoch_d"].iloc[i-1]) if i > 0 and not pd.isna(ind["stoch_d"].iloc[i-1]) else 50.0
            ema200_val = float(ind["ema200"].iloc[i])
            k_cross_up = (stk_prev <= std_prev and stk > std)
            if k_cross_up and stk < 30 and std < 30 and price > ema200_val:
                day_u.add(ticker)

        j_total += len(day_j)
        t_total += len(day_t)
        u_total += len(day_u)

        tu_overlap_today = day_t & day_u
        jt_overlap_today = day_j & day_t
        ju_overlap_today = day_j & day_u
        all3_today = day_j & day_t & day_u

        t_and_u_overlap += len(tu_overlap_today)
        j_and_t_overlap += len(jt_overlap_today)
        j_and_u_overlap += len(ju_overlap_today)
        all_three_overlap += len(all3_today)

        for tk in tu_overlap_today:
            overlap_details.append({"date": str(day), "ticker": tk})

    # Print overlap report
    print("=" * 60)
    print("  SIGNAL OVERLAP ANALYSIS  (Nifty 50, last 1 year)")
    print("=" * 60)
    print(f"  Trading days scanned : {len(all_dates)}")
    print(f"  Stocks in universe   : {len(stock_data)}")
    print()
    print(f"  J signals (raw)      : {j_total}")
    print(f"  T signals (raw)      : {t_total}")
    print(f"  U signals (raw)      : {u_total}")
    print()
    print(f"  T & U overlap        : {t_and_u_overlap}  (same stock, same day)")
    print(f"  J & T overlap        : {j_and_t_overlap}")
    print(f"  J & U overlap        : {j_and_u_overlap}")
    print(f"  J & T & U overlap    : {all_three_overlap}")
    print()

    if t_total > 0:
        print(f"  T signals also in U  : {t_and_u_overlap}/{t_total} = {t_and_u_overlap/t_total*100:.1f}%")
    if u_total > 0:
        print(f"  U signals also in T  : {t_and_u_overlap}/{u_total} = {t_and_u_overlap/u_total*100:.1f}%")

    print()

    if overlap_details:
        print("  T+U overlap instances:")
        for d in overlap_details[:30]:
            print(f"    {d['date']}  {d['ticker']}")
        if len(overlap_details) > 30:
            print(f"    ... and {len(overlap_details) - 30} more")
    else:
        print("  No T+U overlaps found.")

    print()
    print("=" * 60)
    print("  PORTFOLIO BACKTEST  J+T+U  20L / 2L per trade / 2 per day")
    print("=" * 60)

    bt2 = MomentumBacktester()

    def progress(cur, tot, msg):
        print(f"  [{cur}/{tot}] {msg}    ", end="\r")

    result = bt2.run_portfolio_backtest(
        period_days=365,
        universe=50,
        capital_lakhs=20,
        per_stock=200000,
        strategies=["J", "T", "U"],
        entries_per_day=2,
        progress_callback=progress,
    )
    print()

    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return

    s = result["summary"]
    print(f"  Period         : {result['start_date']} to {result['end_date']}")
    print(f"  Capital        : 20L (effective: {result['effective_capital_lakhs']}L)")
    print(f"  Total trades   : {s['total_trades']}")
    print(f"  Winners        : {s['winning_trades']}")
    print(f"  Losers         : {s['losing_trades']}")
    print(f"  Win rate       : {s['win_rate']:.1f}%")
    print(f"  Total P&L      : Rs {s['total_pnl']:,.0f}")
    print(f"  Return         : {s['total_return_pct']:.2f}%")
    print(f"  Avg win        : {s.get('avg_win_pct', 0):.2f}%")
    print(f"  Avg loss       : {s.get('avg_loss_pct', 0):.2f}%")
    print(f"  Max positions  : {result['max_positions_used']}/{result['max_positions']}")
    print(f"  Total signals  : {result['total_signals']}")
    print(f"  Missed signals : {result['missed_signals']}")

    # Strategy breakdown
    strategy_trades = {}
    for t in result["trades"]:
        strat = t.get("strategy", "?")
        if strat not in strategy_trades:
            strategy_trades[strat] = {"count": 0, "pnl": 0, "wins": 0}
        strategy_trades[strat]["count"] += 1
        strategy_trades[strat]["pnl"] += t.get("pnl_amount", 0)
        if t.get("pnl_pct", 0) > 0:
            strategy_trades[strat]["wins"] += 1

    print()
    print("  Per-strategy breakdown:")
    for strat in ["J", "T", "U"]:
        st = strategy_trades.get(strat, {"count": 0, "pnl": 0, "wins": 0})
        wr = (st["wins"] / st["count"] * 100) if st["count"] > 0 else 0
        print(f"    {strat}: {st['count']} trades, Rs {st['pnl']:,.0f} P&L, {wr:.0f}% win rate")

    print()


if __name__ == "__main__":
    run_overlap_analysis()
