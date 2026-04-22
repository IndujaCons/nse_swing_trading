#!/usr/bin/env python3
"""
ETF Core Z-Score ↔ Mom20 — Correlation & Diversification Analysis
=================================================================
Usage:
    python3 correlation_analysis.py

Inputs  : etf_zscore_rebal.csv  (from data/etf_core_zscore_backtest.py)
          mom20_rebal.csv        (from mom15_pit_report.py --mom20)
Outputs : correlation_report.md
          correlation_analysis.png
"""

import sys, os
import pandas as pd
import numpy as np

# ── Load CSVs ─────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

etf_path = os.path.join(BASE, "etf_zscore_rebal.csv")
mom_path = os.path.join(BASE, "mom20_rebal.csv")

for p, label in [(etf_path, "etf_zscore_rebal.csv"), (mom_path, "mom20_rebal.csv")]:
    if not os.path.exists(p):
        print(f"ERROR: {label} not found.")
        print("Run: python3 data/etf_core_zscore_backtest.py   → generates etf_zscore_rebal.csv")
        print("Run: python3 mom15_pit_report.py --mom20         → generates mom20_rebal.csv")
        sys.exit(1)

etf = pd.read_csv(etf_path, parse_dates=["Date"]).set_index("Date").rename(columns={"NAV": "ETF"})
mom = pd.read_csv(mom_path, parse_dates=["date"]).set_index("date").rename(columns={"nav": "Mom20"})

# ── Common window: Jan 2016 – Apr 2026 ────────────────────────────────────────
COMMON_START = "2016-01-01"
COMMON_END   = "2026-04-30"
etf = etf.loc[COMMON_START:COMMON_END, ["ETF"]]
mom = mom.loc[COMMON_START:COMMON_END, ["Mom20"]]

# ── Monthly returns ───────────────────────────────────────────────────────────
etf_ret = etf["ETF"].pct_change().dropna()
mom_ret = mom["Mom20"].pct_change().dropna()

combined = pd.concat([etf_ret, mom_ret], axis=1, join="inner")
combined.columns = ["ETF", "Mom20"]
print(f"Aligned periods: {len(combined)} months  {combined.index.min().date()} → {combined.index.max().date()}")

# ── Pearson / Spearman ────────────────────────────────────────────────────────
pearson  = combined["ETF"].corr(combined["Mom20"], method="pearson")
try:
    spearman = combined["ETF"].corr(combined["Mom20"], method="spearman")
except Exception:
    # scipy not available — compute manually via rank
    spearman = combined["ETF"].rank().corr(combined["Mom20"].rank(), method="pearson")
print(f"\nPearson  correlation: {pearson:.3f}")
print(f"Spearman correlation: {spearman:.3f}")

# ── Rolling 12-month correlation ──────────────────────────────────────────────
rolling_corr = combined["ETF"].rolling(12).corr(combined["Mom20"]).dropna()
print(f"\nRolling 12m correlation:")
print(f"  Mean  : {rolling_corr.mean():.3f}")
print(f"  Median: {rolling_corr.median():.3f}")
print(f"  Min   : {rolling_corr.min():.3f}  (year ending {rolling_corr.idxmin().date()})")
print(f"  Max   : {rolling_corr.max():.3f}  (year ending {rolling_corr.idxmax().date()})")
print(f"  StDev : {rolling_corr.std():.3f}")

# ── Year-by-year correlation ──────────────────────────────────────────────────
combined["Year"] = combined.index.year
yearly_corr = combined.groupby("Year").apply(
    lambda x: x["ETF"].corr(x["Mom20"]) if len(x) >= 6 else np.nan
)
print("\nYearly correlation (≥6 obs):")
for yr, c in yearly_corr.items():
    tag = "  ← high" if c > 0.7 else ("  ← low/neg" if c < 0.2 else "")
    print(f"  {yr}: {c:+.3f}{tag}")

# ── Co-movement counts ────────────────────────────────────────────────────────
both_up        = ((combined["ETF"] > 0) & (combined["Mom20"] > 0)).sum()
both_down      = ((combined["ETF"] < 0) & (combined["Mom20"] < 0)).sum()
etf_up_mom_dn  = ((combined["ETF"] > 0) & (combined["Mom20"] < 0)).sum()
etf_dn_mom_up  = ((combined["ETF"] < 0) & (combined["Mom20"] > 0)).sum()
total = len(combined)

print(f"\nCo-movement ({total} months):")
print(f"  Both up        : {both_up:3d} ({100*both_up/total:.0f}%)")
print(f"  Both down      : {both_down:3d} ({100*both_down/total:.0f}%)")
print(f"  Diversified    : {etf_up_mom_dn + etf_dn_mom_up:3d} ({100*(etf_up_mom_dn+etf_dn_mom_up)/total:.0f}%)")
print(f"    ETF up, Mom down : {etf_up_mom_dn}")
print(f"    Mom up, ETF down : {etf_dn_mom_up}")

# ── 50/50 portfolio stats ─────────────────────────────────────────────────────
combined["Portfolio_5050"] = 0.5 * combined["ETF"] + 0.5 * combined["Mom20"]

def ann_stats(returns, label):
    mean   = returns.mean() * 12
    std    = returns.std() * np.sqrt(12)
    sharpe = mean / std if std > 0 else np.nan
    cum    = (1 + returns).cumprod()
    mdd    = (cum / cum.cummax() - 1).min()
    total_ret = cum.iloc[-1] - 1
    print(f"  {label:22s}  CAGR ~{mean*100:5.1f}%  Vol {std*100:5.1f}%  Sharpe {sharpe:.2f}  MaxDD {mdd*100:5.1f}%  Total {total_ret*100:+.0f}%")

print("\nAnnualized stats (arithmetic CAGR proxy, geometric slightly lower):")
ann_stats(combined["ETF"],         "ETF Core Z-Score")
ann_stats(combined["Mom20"],       "Mom20")
ann_stats(combined["Portfolio_5050"], "50/50 portfolio")

# ── Per-year returns (last NAV of year / last NAV of prior year) ───────────────
def yearly_from_nav(df, col):
    s = df[col]
    last = s.groupby(s.index.year).last()
    return last.pct_change().dropna()

etf_raw  = pd.read_csv(etf_path, parse_dates=["Date"]).set_index("Date")["NAV"]
mom_raw  = pd.read_csv(mom_path, parse_dates=["date"]).set_index("date")["nav"]
etf_raw  = etf_raw.loc[COMMON_START:COMMON_END]
mom_raw  = mom_raw.loc[COMMON_START:COMMON_END]

etf_yr_s = etf_raw.groupby(etf_raw.index.year).last().pct_change().dropna()
mom_yr_s = mom_raw.groupby(mom_raw.index.year).last().pct_change().dropna()

all_years = sorted(set(etf_yr_s.index) | set(mom_yr_s.index))

print("\nPer-year returns (NAV-based):")
print(f"  {'Year':<6}  {'ETF Core':>9}  {'Mom20':>9}  {'50/50':>9}  {'Winner':<10}  Diversified?")
print(f"  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*10}  {'─'*12}")
for yr in all_years:
    etf_yr = etf_yr_s.get(yr, float("nan"))
    mom_yr = mom_yr_s.get(yr, float("nan"))
    if np.isnan(etf_yr) or np.isnan(mom_yr):
        continue
    p50_yr  = 0.5 * etf_yr + 0.5 * mom_yr
    winner  = "ETF" if etf_yr > mom_yr else "Mom20"
    gap     = abs(etf_yr - mom_yr)
    diversified = "YES ✓" if gap > 0.10 else ("slight" if gap > 0.05 else "—")
    print(f"  {yr:<6}  {etf_yr*100:>+8.1f}%  {mom_yr*100:>+8.1f}%  {p50_yr*100:>+8.1f}%  {winner:<10}  {diversified}")

# ── Diversification verdict ───────────────────────────────────────────────────
print("\n── Diversification Verdict ──")
if pearson < 0.30:
    verdict = "EXCELLENT (<0.30) — 50/50 well-supported; could even tilt more to Mom20"
elif pearson < 0.50:
    verdict = "STRONG (0.30-0.50) — 50/50 is appropriate"
elif pearson < 0.70:
    verdict = "MODERATE (0.50-0.70) — 50/50 defensible, partial diversification benefit"
elif pearson < 0.85:
    verdict = "LIMITED (0.70-0.85) — reconsider overlap; add third uncorrelated strategy"
else:
    verdict = "NONE (>0.85) — effectively same trade; treat as one position"
print(f"  Pearson {pearson:.3f} → {verdict}")

# ── Save outputs ──────────────────────────────────────────────────────────────
combined.drop(columns="Year").to_csv(os.path.join(BASE, "correlation_monthly.csv"))
rolling_corr.to_csv(os.path.join(BASE, "correlation_rolling12m.csv"))
print("\nSaved: correlation_monthly.csv, correlation_rolling12m.csv")

# ── Plot ──────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    fig.suptitle("ETF Core Z-Score ↔ Mom20 — Diversification Analysis", fontsize=13)

    # Cumulative returns
    (1 + combined["ETF"]).cumprod().plot(ax=axes[0], label="ETF Core Z-Score", color="#2196F3")
    (1 + combined["Mom20"]).cumprod().plot(ax=axes[0], label="Mom20", color="#FF6B35")
    (1 + combined["Portfolio_5050"]).cumprod().plot(ax=axes[0], label="50/50 blend",
                                                     color="#4CAF50", linewidth=2.5, linestyle="--")
    axes[0].set_title("Cumulative Returns (Jan 2016 base = 1.0)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3); axes[0].set_ylabel("Growth of ₹1")

    # Rolling 12m correlation
    rolling_corr.plot(ax=axes[1], color="#9C27B0")
    axes[1].axhline(y=0,    color="gray",   linestyle=":", alpha=0.6)
    axes[1].axhline(y=0.5,  color="orange", linestyle="--", alpha=0.5, label="0.5 threshold")
    axes[1].axhline(y=0.70, color="red",    linestyle="--", alpha=0.4, label="0.7 threshold")
    axes[1].fill_between(rolling_corr.index, rolling_corr, 0,
                         where=(rolling_corr > 0), alpha=0.15, color="#9C27B0")
    axes[1].set_title("Rolling 12-Month Pearson Correlation")
    axes[1].set_ylim(-1, 1); axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    out_png = os.path.join(BASE, "correlation_analysis.png")
    plt.savefig(out_png, dpi=130)
    print(f"Chart saved: correlation_analysis.png")
except ImportError:
    print("matplotlib not available — skipping chart")
