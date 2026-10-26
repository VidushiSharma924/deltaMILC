"""Evaluation metrics for streamflow prediction."""

import numpy as np
import scipy.stats


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Compute a suite of hydrological performance metrics.

    Parameters
    ----------
    pred   : [n_basins, n_time]  — predicted streamflow (mm/day)
    target : [n_basins, n_time]  — observed streamflow (mm/day)

    Returns
    -------
    dict with keys: NSE, KGE, Bias, RMSE, ubRMSE, Corr, FLV, FHV, lowRMSE, highRMSE
    """
    n_basins, n_time = pred.shape
    NSE = np.full(n_basins, np.nan)
    KGE = np.full(n_basins, np.nan)
    Bias = np.full(n_basins, np.nan)
    RMSE = np.full(n_basins, np.nan)
    ubRMSE = np.full(n_basins, np.nan)
    Corr = np.full(n_basins, np.nan)
    FLV = np.full(n_basins, np.nan)
    FHV = np.full(n_basins, np.nan)
    lowRMSE = np.full(n_basins, np.nan)
    highRMSE = np.full(n_basins, np.nan)

    for k in range(n_basins):
        p = pred[k]
        t = target[k]
        valid = np.where(np.logical_and(~np.isnan(p), ~np.isnan(t)))[0]
        if len(valid) < 2:
            continue
        pp, tt = p[valid], t[valid]

        Bias[k] = np.mean(pp - tt)
        RMSE[k] = np.sqrt(np.mean((pp - tt) ** 2))
        ubRMSE[k] = np.sqrt(np.mean(((pp - pp.mean()) - (tt - tt.mean())) ** 2))

        corr, _ = scipy.stats.pearsonr(pp, tt)
        Corr[k] = corr

        tt_mean = tt.mean()
        sst = np.sum((tt - tt_mean) ** 2)
        if sst > 0:
            NSE[k] = 1.0 - np.sum((pp - tt) ** 2) / sst

        if tt_mean > 0:
            KGE[k] = 1.0 - np.sqrt(
                (corr - 1) ** 2
                + (np.std(pp) / np.std(tt) - 1) ** 2
                + (pp.mean() / tt_mean - 1) ** 2
            )

        # Flow duration curve fractions
        ps = np.sort(pp)
        ts = np.sort(tt)
        n = len(ps)
        idx_low = round(0.3 * n)
        idx_high = round(0.98 * n)

        if np.sum(ts[:idx_low]) > 0:
            FLV[k] = np.sum(ps[:idx_low] - ts[:idx_low]) / np.sum(ts[:idx_low]) * 100
        if np.sum(ts[idx_high:]) > 0:
            FHV[k] = np.sum(ps[idx_high:] - ts[idx_high:]) / np.sum(ts[idx_high:]) * 100

        lowRMSE[k] = np.sqrt(np.mean((ps[:idx_low] - ts[:idx_low]) ** 2))
        highRMSE[k] = np.sqrt(np.mean((ps[idx_high:] - ts[idx_high:]) ** 2))

    return dict(NSE=NSE, KGE=KGE, Bias=Bias, RMSE=RMSE, ubRMSE=ubRMSE,
                Corr=Corr, FLV=FLV, FHV=FHV, lowRMSE=lowRMSE, highRMSE=highRMSE)


def print_metrics_summary(metrics: dict) -> None:
    keys = ["NSE", "KGE", "FLV", "FHV", "lowRMSE", "highRMSE"]
    header = "  ".join(f"{k:>10}" for k in keys)
    values = "  ".join(f"{np.nanmedian(metrics[k]):>10.3f}" for k in keys)
    print(f"Metric (median):  {header}")
    print(f"                  {values}")
