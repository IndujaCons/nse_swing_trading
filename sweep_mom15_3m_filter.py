"""
Mom15 3M Return Filter Sweep
Tests whether filtering stocks by minimum 3M price return
improves Mom15 performance across different thresholds.

ret_3m = price_now / price_63d_ago - 1
Filter: only include stocks where ret_3m >= threshold

Sweep thresholds: [None, -0.10, -0.05, 0.0, 0.02, 0.05]
Filter applied AFTER beta_cap, BEFORE Z-scoring.
Stocks with no 63d history pass through (same as EPS filter logic).

Mom15 frozen params:
  top_n=15, buffer_in=10, buffer_out=30, rebalance_months=2,
  beta_cap=1.0, pit_universe=True, eps_filter=0.0, w12=0.50, w6=0.0, w3=0.50
"""

import sys
import os
from datetime import datetime, timedelta, date as _date

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "/Users/jay/Desktop/relative_strength")

from data.momentum_backtest import (
    load_pit_nifty200,
    get_all_pit_tickers,
    get_pit_universe,
    TICKER_ALIASES,
    _fetch_alias,
)

# ── Constants ────────────────────────────────────────────────────────────────
CAPITAL = 20_00_000          # 20 lakhs
TOP_N = 15
BUFFER_IN = 10
BUFFER_OUT = 30
REBAL_MONTHS = 2
BETA_CAP = 1.0
W12 = 0.50
W6 = 0.0
W3 = 0.50
LONG_PERIOD = 252            # 12m return lookback
SHORT_PERIOD = 126           # 6m return lookback
VOL_LOOKBACK = 252
PERIOD_DAYS = 365 * 11

# None = baseline (no filter); float = minimum ret_3m required
SWEEP_THRESHOLDS = [None, -0.10, -0.05, 0.0, 0.02, 0.05]

# ── Charges (Zerodha equity delivery) ────────────────────────────────────────

def compute_net_profit(gross_pnl, entry_price, exit_price, shares):
    B = entry_price * shares
    S = exit_price * shares
    total_turnover = B + S
    stt      = 0.001      * total_turnover
    exchange = 0.0000307  * total_turnover
    sebi     = 0.000001   * total_turnover
    stamp    = 0.00015    * B
    gst      = 0.18       * (exchange + sebi)
    total_charges = stt + exchange + sebi + stamp + gst
    deductible = exchange + sebi + stamp + gst
    taxable_profit = gross_pnl - deductible
    stcg_tax = 0.20 * taxable_profit if taxable_profit > 0 else 0.0
    net = gross_pnl - total_charges - stcg_tax
    return net, total_charges

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(trades, label, avg_filtered):
    if not trades:
        print(f"\n{label}: No trades")
        return None

    df = pd.DataFrame(trades)
    nets = []
    for _, row in df.iterrows():
        net, _ = compute_net_profit(row["pnl"], row["entry_price"], row["exit_price"], row["shares"])
        nets.append(net)
    df["net_profit"] = nets

    n = len(df)
    wr = (df["pnl"] > 0).mean() * 100
    gross_win  = df[df["pnl"] > 0]["pnl"].sum()
    gross_loss = df[df["pnl"] < 0]["pnl"].abs().sum()
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    df["exit_year"] = pd.to_datetime(df["exit_date"].astype(str)).dt.year
    yearly = df.groupby("exit_year")["net_profit"].sum()
    yearly_pct = (yearly / CAPITAL * 100).round(1)

    net_per_yr = yearly_pct.mean()
    neg_years  = int((yearly_pct < 0).sum())

    print(f"\n{'='*60}")
    print(f"  {label}  |  avg filtered/rebal: {avg_filtered:.1f} stocks")
    print(f"{'='*60}")
    print(f"  Trades: {n}  |  WR: {wr:.1f}%  |  PF: {pf:.2f}")
    print(f"  Avg Net %/yr: {net_per_yr:.1f}%  |  Neg years: {neg_years}")
    print(f"\n  Year-by-year net returns:")
    for yr, pct in yearly_pct.items():
        marker = " <--" if pct < 0 else ""
        print(f"    {yr}: {pct:+.1f}%{marker}")

    return {
        "label": label,
        "trades": n,
        "wr": round(wr, 1),
        "pf": round(pf, 2),
        "net_per_yr": round(net_per_yr, 1),
        "neg_years": neg_years,
        "yearly": yearly_pct.to_dict(),
        "avg_filtered": round(avg_filtered, 1),
    }

# ── Rebalancing loop ──────────────────────────────────────────────────────────

def run_mom15_loop(stock_data, nifty50_data,
                   date_to_iloc, n50_map, trading_days, rebal_dates,
                   pit_data, min_ret_3m):
    """
    Run Mom15 rebalancing loop with optional minimum 3M return filter.

    min_ret_3m: None = no filter; float = minimum ret_3m required to pass
    Stocks with no 63d history pass through (same as EPS filter logic).

    Returns (trades_list, avg_filtered_per_rebal)
    """
    PER_STOCK_CAPITAL = CAPITAL / TOP_N

    portfolio = {}
    trades = []
    cash = float(CAPITAL)
    filter_counts = []

    def compute_scores(day):
        """Compute Normalized Momentum Score for all stocks on a given day."""
        pit_set = get_pit_universe(pit_data, day) if pit_data is not None else None
        scores = {}

        for ticker, df in stock_data.items():
            if pit_set is not None and ticker not in pit_set:
                continue

            idx_map = date_to_iloc.get(ticker, {})
            ci = idx_map.get(day)
            if ci is None:
                for offset in range(1, 6):
                    prev = day - timedelta(days=offset)
                    if prev in idx_map:
                        ci = idx_map[prev]
                        break

            if ci is None or ci < VOL_LOOKBACK + 18:
                continue

            closes = df["Close"]
            if ci - LONG_PERIOD < 0 or ci - SHORT_PERIOD < 0 or ci < 63:
                continue

            p_now   = float(closes.iloc[ci])
            p_long  = float(closes.iloc[ci - LONG_PERIOD])
            p_short = float(closes.iloc[ci - SHORT_PERIOD])
            p_3m    = float(closes.iloc[ci - 63])

            if p_long <= 0 or p_short <= 0 or p_now <= 0 or p_3m <= 0:
                continue

            ret_long  = p_now / p_long  - 1
            ret_short = p_now / p_short - 1
            ret_3m    = p_now / p_3m    - 1

            vol_bars = min(VOL_LOOKBACK, ci)
            log_rets = np.diff(np.log(
                np.maximum(closes.iloc[ci - vol_bars:ci + 1].values.astype(float), 0.01)
            ))
            sigma = float(np.std(log_rets)) * np.sqrt(252)
            if sigma < 0.01:
                continue

            mr_12 = ret_long  / sigma
            mr_6  = ret_short / sigma
            mr_3  = ret_3m    / sigma

            # Beta cap filter (BETA_CAP = 1.0)
            beta = None
            n50_ci = n50_map.get(day)
            if n50_ci is None:
                for offset in range(1, 6):
                    prev2 = day - timedelta(days=offset)
                    if prev2 in n50_map:
                        n50_ci = n50_map[prev2]
                        break
            if n50_ci is not None and n50_ci >= 252:
                n50_closes = nifty50_data["Close"].iloc[n50_ci - 252:n50_ci + 1].values.astype(float)
                stk_closes = closes.iloc[ci - 252:ci + 1].values.astype(float)
                n50_rets   = np.diff(n50_closes) / np.maximum(n50_closes[:-1], 0.01)
                stk_rets   = np.diff(stk_closes)  / np.maximum(stk_closes[:-1],  0.01)
                if len(stk_rets) == len(n50_rets):
                    cov_val = np.cov(stk_rets, n50_rets)
                    if cov_val.shape == (2, 2) and cov_val[1, 1] > 1e-10:
                        beta = cov_val[0, 1] / cov_val[1, 1]

            if beta is None or beta > BETA_CAP:
                continue   # beta_cap=1.0 filter

            scores[ticker] = {
                "mr_12":  mr_12,
                "mr_6":   mr_6,
                "mr_3":   mr_3,
                "ret_3m": ret_3m,
                "price":  p_now,
                "beta":   beta,
            }

        return scores

    for rebal_day in rebal_dates:
        scores = compute_scores(rebal_day)
        if not scores:
            continue

        # ── 3M Return Filter ─────────────────────────────────────────────────
        filtered_out = 0
        if min_ret_3m is not None:
            filtered_scores = {}
            for ticker, s in scores.items():
                r3m = s.get("ret_3m")
                if r3m is None:
                    filtered_scores[ticker] = s  # no 63d history → pass through
                    continue
                if r3m >= min_ret_3m:
                    filtered_scores[ticker] = s  # passes filter
                else:
                    filtered_out += 1            # filtered out
            scores = filtered_scores
        filter_counts.append(filtered_out)

        if len(scores) < 20:
            continue

        # ── Z-scoring ────────────────────────────────────────────────────────
        mr12_vals = np.array([s["mr_12"] for s in scores.values()])
        mr6_vals  = np.array([s["mr_6"]  for s in scores.values()])

        mu_12, std_12 = np.mean(mr12_vals), np.std(mr12_vals)
        mu_6,  std_6  = np.mean(mr6_vals),  np.std(mr6_vals)

        if std_12 < 0.001 or std_6 < 0.001:
            continue

        mr3_vals = np.array([s["mr_3"] for s in scores.values()])
        if len(mr3_vals) > 10 and np.std(mr3_vals) > 0.001:
            mu_3, std_3 = np.mean(mr3_vals), np.std(mr3_vals)
        else:
            mu_3, std_3 = 0.0, 1.0

        for ticker, s in scores.items():
            z_12 = (s["mr_12"] - mu_12) / std_12
            z_6  = (s["mr_6"]  - mu_6)  / std_6
            z_3  = (s["mr_3"]  - mu_3)  / std_3

            wt_z = W12 * z_12 + W6 * z_6 + W3 * z_3
            norm_score = (1 + wt_z) if wt_z >= 0 else 1.0 / (1 - wt_z)
            s["wt_z"]       = wt_z
            s["norm_score"] = norm_score

        # ── Buffer + rank ─────────────────────────────────────────────────────
        ranked      = sorted(scores.items(), key=lambda x: -x[1]["norm_score"])
        ticker_rank = {t: rank + 1 for rank, (t, _) in enumerate(ranked)}

        current_tickers       = set(portfolio.keys())
        new_portfolio_tickers = set()

        # Keep existing if rank <= buffer_out
        for t in current_tickers:
            rank = ticker_rank.get(t, 999)
            if rank <= BUFFER_OUT:
                new_portfolio_tickers.add(t)

        # Add new if rank <= buffer_in
        for rank, (t, _) in enumerate(ranked):
            if rank + 1 <= BUFFER_IN and t not in current_tickers:
                new_portfolio_tickers.add(t)

        # Fill remaining slots
        for rank, (t, _) in enumerate(ranked):
            if len(new_portfolio_tickers) >= TOP_N:
                break
            new_portfolio_tickers.add(t)

        # Cap at top_n (keep highest-ranked)
        if len(new_portfolio_tickers) > TOP_N:
            ranked_new = sorted(new_portfolio_tickers, key=lambda t: ticker_rank.get(t, 999))
            new_portfolio_tickers = set(ranked_new[:TOP_N])

        # ── Execute rebalance ─────────────────────────────────────────────────
        # Sell stocks leaving portfolio
        to_sell = current_tickers - new_portfolio_tickers
        for t in to_sell:
            pos = portfolio[t]
            exit_price = scores.get(t, {}).get("price")
            if exit_price is None:
                idx_map = date_to_iloc.get(t, {})
                ci = idx_map.get(rebal_day)
                if ci is not None:
                    exit_price = float(stock_data[t]["Close"].iloc[ci])
                else:
                    exit_price = pos["entry_price"]
            pnl = (exit_price - pos["entry_price"]) * pos["shares"]
            trades.append({
                "symbol":      t,
                "strategy":    "Mom15",
                "entry_date":  pos["entry_date"],
                "exit_date":   rebal_day,
                "entry_price": pos["entry_price"],
                "exit_price":  exit_price,
                "shares":      pos["shares"],
                "pnl":         pnl,
                "exit_reason": "REBALANCE_OUT",
                "hold_days":   (rebal_day - pos["entry_date"]).days,
            })
            cash += pos["shares"] * exit_price
            del portfolio[t]

        # Buy new stocks
        to_buy = new_portfolio_tickers - set(portfolio.keys())
        for t in to_buy:
            s = scores.get(t)
            if s is None:
                continue
            price = s["price"]
            if price <= 0:
                continue
            shares = int(PER_STOCK_CAPITAL // price)
            if shares <= 0:
                continue
            cost = shares * price
            if cost > cash:
                continue
            portfolio[t] = {
                "entry_date":  rebal_day,
                "entry_price": price,
                "shares":      shares,
                "rank":        ticker_rank.get(t, 999),
                "norm_score":  s["norm_score"],
            }
            cash -= cost

    # Close remaining positions at end
    last_day = trading_days[-1]
    for t in list(portfolio.keys()):
        pos = portfolio[t]
        idx_map = date_to_iloc.get(t, {})
        ci = idx_map.get(last_day)
        if ci is not None:
            exit_price = float(stock_data[t]["Close"].iloc[ci])
        else:
            exit_price = pos["entry_price"]
            for offset in range(1, 10):
                prev = last_day - timedelta(days=offset)
                if prev in idx_map:
                    exit_price = float(stock_data[t]["Close"].iloc[idx_map[prev]])
                    break
        pnl = (exit_price - pos["entry_price"]) * pos["shares"]
        trades.append({
            "symbol":      t,
            "strategy":    "Mom15",
            "entry_date":  pos["entry_date"],
            "exit_date":   last_day,
            "entry_price": pos["entry_price"],
            "exit_price":  exit_price,
            "shares":      pos["shares"],
            "pnl":         pnl,
            "exit_reason": "BACKTEST_END",
            "hold_days":   (last_day - pos["entry_date"]).days,
        })
    portfolio.clear()

    avg_filtered = float(np.mean(filter_counts)) if filter_counts else 0.0
    return trades, avg_filtered

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    end_date      = datetime.now()
    daily_start   = end_date - timedelta(days=PERIOD_DAYS + 600)
    bt_start_date = (end_date - timedelta(days=PERIOD_DAYS)).date()

    print("=" * 60)
    print("Mom15 3M Return Filter Sweep")
    print("=" * 60)

    # ── Phase 1: PIT universe ─────────────────────────────────────────────
    pit_data = load_pit_nifty200()
    if pit_data is None:
        print("ERROR: nifty200_pit.json not found")
        sys.exit(1)

    all_pit_tickers = get_all_pit_tickers(pit_data)
    tickers = sorted(all_pit_tickers)
    print(f"PIT universe: {len(tickers)} unique tickers")

    # ── Phase 2: Fetch stock data ──────────────────────────────────────────
    stock_data = {}
    total = len(tickers)
    for idx, ticker in enumerate(tickers):
        nse_symbol = f"{ticker}.NS"
        try:
            daily = yf.Ticker(nse_symbol).history(start=daily_start, end=end_date)
        except Exception:
            daily = pd.DataFrame()
        if (daily.empty or len(daily) < 300) and ticker in TICKER_ALIASES:
            daily = _fetch_alias(ticker, daily_start, end_date)
        if not daily.empty and len(daily) >= 300:
            stock_data[ticker] = daily
        if (idx + 1) % 50 == 0:
            print(f"  Loaded {idx+1}/{total} stocks...")

    print(f"Data loaded: {len(stock_data)} stocks with sufficient history")

    # ── Phase 3: Nifty 50 for beta calculation ────────────────────────────
    nifty50_data = None
    n50_map = {}
    try:
        nifty50_data = yf.Ticker("^NSEI").history(start=daily_start, end=end_date)
        for iloc_idx in range(len(nifty50_data)):
            dt = nifty50_data.index[iloc_idx].date()
            n50_map[dt] = iloc_idx
        print(f"Nifty 50 loaded: {len(nifty50_data)} bars (for beta_cap)")
    except Exception:
        print("WARNING: Could not fetch Nifty 50 data")

    # ── Phase 4: Build date mappings and trading_days ─────────────────────
    date_to_iloc = {}
    for ticker, df in stock_data.items():
        mapping = {}
        for iloc_idx in range(len(df)):
            dt = df.index[iloc_idx].date()
            mapping[dt] = iloc_idx
        date_to_iloc[ticker] = mapping

    all_dates_count = {}
    for ticker, df in stock_data.items():
        for d in df.index:
            dt = d.date()
            all_dates_count[dt] = all_dates_count.get(dt, 0) + 1
    trading_days = sorted(d for d, c in all_dates_count.items() if c >= 100)
    trading_days = [d for d in trading_days if d >= bt_start_date]
    print(f"Trading days: {len(trading_days)} ({trading_days[0]} to {trading_days[-1]})")

    # ── Phase 5: Determine rebalance dates ────────────────────────────────
    rebal_dates = [trading_days[0]]
    next_rebal = trading_days[0]
    for d in trading_days[1:]:
        months_diff = (d.year - next_rebal.year) * 12 + (d.month - next_rebal.month)
        if months_diff >= REBAL_MONTHS:
            rebal_dates.append(d)
            next_rebal = d
    print(f"Rebalance dates: {len(rebal_dates)}")

    # ── Phase 6: Sweep ────────────────────────────────────────────────────
    results = []

    for threshold in SWEEP_THRESHOLDS:
        if threshold is None:
            label = "None (baseline)"
        elif threshold == 0.0:
            label = "ret_3m>=0%"
        elif threshold > 0:
            label = f"ret_3m>=+{threshold*100:.0f}%"
        else:
            label = f"ret_3m>={threshold*100:.0f}%"

        print(f"\n--- Running {label} ---")

        trades, avg_filtered = run_mom15_loop(
            stock_data=stock_data,
            nifty50_data=nifty50_data,
            date_to_iloc=date_to_iloc,
            n50_map=n50_map,
            trading_days=trading_days,
            rebal_dates=rebal_dates,
            pit_data=pit_data,
            min_ret_3m=threshold,
        )
        print(f"  Completed: {len(trades)} trades, avg filtered: {avg_filtered:.1f}")
        r = compute_metrics(trades, label=label, avg_filtered=avg_filtered)
        if r:
            results.append(r)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n\n" + "=" * 75)
    print("3M Return Filter Sweep — Mom15 (11yr PIT, 20L, 2-monthly)")
    print("=" * 75)
    print(f"{'Threshold':<22} {'Trades':>7} {'WR%':>6} {'PF':>6} {'Net%/yr':>9} {'NegYrs':>7} {'AvgFilt':>8}")
    print("-" * 75)
    for r in results:
        print(f"{r['label']:<22} {r['trades']:>7} {r['wr']:>6.1f} "
              f"{r['pf']:>6.2f} {r['net_per_yr']:>+9.1f} {r['neg_years']:>7} {r['avg_filtered']:>8.1f}")

    # Year-by-year table
    print("\n\nYEAR-BY-YEAR NET RETURNS (%)")
    all_years = sorted(set(yr for r in results for yr in r["yearly"]))
    header = f"{'Year':>6}" + "".join(f"{r['label'][:14]:>15}" for r in results)
    print(header)
    print("-" * (6 + 15 * len(results)))
    for yr in all_years:
        row = f"{yr:>6}"
        for r in results:
            val = r["yearly"].get(yr)
            if val is not None:
                row += f"{val:>+15.1f}"
            else:
                row += f"{'N/A':>15}"
        print(row)

    print("\n\nDone.")


if __name__ == "__main__":
    main()
