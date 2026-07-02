"""Mom20/Mom15 raw feature collection.

Per-ticker compute step — produces the dict that gets accumulated in
`mom20_raw` for cross-sectional Z-scoring after the loop. Pure function;
no IO, no side effects.
"""

import numpy as np


def compute_mom20_features(ticker, closes, i, price, n50_ret_series, n50_var):
    """Returns the mom20_raw entry for one ticker, or None if insufficient
    data / degenerate σ. Math identical to the original inline block."""
    if i < 252:
        return None
    try:
        ret_12m = price / float(closes.iloc[i - 252]) - 1
        ret_6m  = price / float(closes.iloc[i - 126]) - 1
        ret_3m  = price / float(closes.iloc[i - 63])  - 1 if i >= 63 else None
        log_rets = np.log(closes.iloc[i - 251:i + 1] / closes.iloc[i - 252:i].values)
        sigma = float(log_rets.std()) * np.sqrt(252)
        if sigma <= 0.001:
            return None
        sigma_3m = float(log_rets.iloc[-63:].std()) * np.sqrt(252) if len(log_rets) >= 63 else sigma

        ema20     = float(closes.ewm(span=20, adjust=False).mean().iloc[i])
        ema20_ext = round((price / ema20 - 1) * 100, 1) if ema20 > 0 else 0.0
        high_52w  = float(closes.iloc[i - 252:i + 1].max())

        # Beta vs Nifty 50 (date-aligned).
        mom_beta = None
        if n50_ret_series is not None and n50_var > 1e-10:
            stock_ret_series = closes.astype(float).pct_change().iloc[i - 251:i + 1]
            common_dates = stock_ret_series.index.intersection(n50_ret_series.index)
            if len(common_dates) >= 100:
                sr = stock_ret_series.loc[common_dates].values
                nr = n50_ret_series.loc[common_dates].values
                mask = ~(np.isnan(sr) | np.isnan(nr))
                if mask.sum() >= 100:
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
            "mr_3":     (ret_3m / sigma) if ret_3m is not None else None,
            "beta":     round(mom_beta, 2) if mom_beta is not None else None,
            "high_52w": high_52w,
        }
    except Exception:
        return None
