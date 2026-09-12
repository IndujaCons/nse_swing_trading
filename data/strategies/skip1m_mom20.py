"""Skip1M variant of Mom20 raw feature collection — research/scan only.

Identical to compute_mom20_features (data/strategies/mom20.py) except the
3-month leg skips the most recent ~25 trading days (classical 12-2-style
skip, applied to the 3m leg only — "full Option C", validated in
mom15_pit_report.py via `--mom20 --skip-1m --skip-3m-only --skip-days 25`).
The 12-month leg, sigma, ema20_ext, beta and high_52w are all unchanged —
only ret_3m/mr_3 differ. Pure function; no IO, no side effects.
"""

import numpy as np

SKIP_DAYS = 25


def compute_skip1m_features(ticker, closes, i, price, bench_ret_series, bench_var, highs=None):
    """Returns the skip1m_raw entry for one ticker, or None if insufficient
    data / degenerate σ. Same shape as compute_mom20_features's return dict."""
    if i < 252 or i < SKIP_DAYS + 63:
        return None
    try:
        ret_12m = price / float(closes.iloc[i - 252]) - 1
        ret_6m  = price / float(closes.iloc[i - 126]) - 1
        p_ref_3m = float(closes.iloc[i - SKIP_DAYS])
        ret_3m   = p_ref_3m / float(closes.iloc[i - SKIP_DAYS - 63]) - 1
        log_rets = np.log(closes.iloc[i - 251:i + 1] / closes.iloc[i - 252:i].values)
        sigma = float(np.std(log_rets)) * np.sqrt(252)
        if sigma <= 0.001:
            return None
        sigma_3m = float(np.std(log_rets.iloc[-63:])) * np.sqrt(252) if len(log_rets) >= 63 else sigma

        ema20     = float(closes.ewm(span=20, adjust=False).mean().iloc[i])
        ema20_ext = round((price / ema20 - 1) * 100, 1) if ema20 > 0 else 0.0
        high_source = highs if highs is not None else closes
        high_52w  = float(high_source.iloc[i - 252:i + 1].max())

        # Beta vs regime benchmark (Nifty200), date-aligned — unskipped, same as compute_mom20_features.
        mom_beta = None
        if bench_ret_series is not None and bench_var > 1e-10:
            stock_ret_series = closes.astype(float).pct_change().iloc[i - 251:i + 1]
            common_dates = stock_ret_series.index.intersection(bench_ret_series.index)
            if len(common_dates) > 50:
                sr = stock_ret_series.loc[common_dates].values
                nr = bench_ret_series.loc[common_dates].values
                mask = ~(np.isnan(sr) | np.isnan(nr))
                if mask.sum() > 50:
                    cov_val = np.cov(sr[mask], nr[mask])
                    if cov_val.shape == (2, 2) and cov_val[1, 1] > 1e-10:
                        mom_beta = cov_val[0, 1] / cov_val[1, 1]

        return {
            "ticker":   ticker,
            "price":    round(price, 2),
            "ret_12m":  ret_12m,
            "ret_6m":   ret_6m,
            "ret_3m":   ret_3m,
            "sigma":    sigma,
            "sigma_3m": sigma_3m,
            "ema20_ext": ema20_ext,
            "mr_12":    ret_12m / sigma,
            "mr_6":     ret_6m / sigma,
            "mr_3":     ret_3m / sigma,
            "beta":     round(mom_beta, 2) if mom_beta is not None else None,
            "high_52w": high_52w,
        }
    except Exception:
        return None
