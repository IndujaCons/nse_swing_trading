"""Alpha20: CAPM Jensen's alpha per ticker.

Pure compute — produces the dict accumulated in `alpha20_raw` for ranking
after the per-ticker loop.
"""

import numpy as np


def compute_alpha20_features(ticker, closes, i, price, n50_ret_series, n50_var):
    """Returns the alpha20_raw entry for one ticker, or None.
    Math identical to the original inline block."""
    if i < 252 or n50_ret_series is None or n50_var <= 1e-10:
        return None
    try:
        stock_ret_series_a = closes.astype(float).pct_change().iloc[i - 251:i + 1]
        common_dates_a = stock_ret_series_a.index.intersection(n50_ret_series.index)
        if len(common_dates_a) < 100:
            return None
        sr_a = stock_ret_series_a.loc[common_dates_a].values
        nr_a = n50_ret_series.loc[common_dates_a].values
        mask_a = ~(np.isnan(sr_a) | np.isnan(nr_a))
        if mask_a.sum() < 100:
            return None
        rf_daily = 0.065 / 252
        cov_val = np.cov(sr_a[mask_a], nr_a[mask_a])
        if not (cov_val.shape == (2, 2) and cov_val[1, 1] > 1e-10):
            return None
        beta = cov_val[0, 1] / cov_val[1, 1]
        alpha_daily = np.mean(sr_a[mask_a]) - (rf_daily + beta * (float(np.mean(nr_a[mask_a])) - rf_daily))
        alpha_annual = alpha_daily * 252
        return {
            "ticker": ticker,
            "price":  round(price, 2),
            "alpha":  round(alpha_annual * 100, 2),
            "beta":   round(beta, 2),
        }
    except Exception:
        return None
