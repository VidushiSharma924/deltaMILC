"""Hydrology-specific data scaler (mean/std normalization with optional log-gamma transform)."""

import numpy as np


def _calc_stat(x: np.ndarray) -> list:
    """Compute [p10, p90, mean, std] for a flattened array, ignoring NaNs."""
    flat = x.flatten()
    valid = flat[~np.isnan(flat)]
    std = float(np.std(valid))
    if std < 1e-3:
        std = 1.0
    return [
        float(np.percentile(valid, 10)),
        float(np.percentile(valid, 90)),
        float(np.mean(valid)),
        std,
    ]


def _calc_stat_gamma(x: np.ndarray) -> list:
    """Log-gamma transform then compute stats — used for precipitation in dMILC1.0."""
    flat = x.flatten()
    valid = flat[~np.isnan(flat)]
    transformed = np.log10(np.sqrt(valid) + 0.1)
    std = float(np.std(transformed))
    if std < 1e-3:
        std = 1.0
    return [
        float(np.percentile(transformed, 10)),
        float(np.percentile(transformed, 90)),
        float(np.mean(transformed)),
        std,
    ]


def _build_stat_dict(
    attr_vars: list,
    attr_data: np.ndarray,
    series_vars: list,
    series_data: np.ndarray,
    log_norm_cols: list,
) -> dict:
    stat = {}
    for k, var in enumerate(series_vars):
        stat[var] = (
            _calc_stat_gamma(series_data[:, :, k])
            if var in log_norm_cols
            else _calc_stat(series_data[:, :, k])
        )
    for k, var in enumerate(attr_vars):
        stat[var] = _calc_stat(attr_data[:, k])
    return stat


def _normalize(x: np.ndarray, var_list: list, stat: dict, log_norm_cols: list) -> np.ndarray:
    out = np.full(x.shape, np.nan)
    for k, var in enumerate(var_list):
        s = stat[var]
        if x.ndim == 3:
            col = x[:, :, k].copy()
            if var in log_norm_cols:
                col = np.log10(np.sqrt(col) + 0.1)
            out[:, :, k] = (col - s[2]) / s[3]
        elif x.ndim == 2:
            col = x[:, k].copy()
            if var in log_norm_cols:
                col = np.log10(np.sqrt(col) + 0.1)
            out[:, k] = (col - s[2]) / s[3]
    return out


def _denormalize(x: np.ndarray, var_list: list, stat: dict, log_norm_cols: list) -> np.ndarray:
    out = np.full(x.shape, np.nan)
    for k, var in enumerate(var_list):
        s = stat[var]
        if x.ndim == 3:
            col = x[:, :, k] * s[3] + s[2]
            if var in log_norm_cols:
                col = (np.power(10.0, col) - 0.1) ** 2
            out[:, :, k] = col
        elif x.ndim == 2:
            col = x[:, k] * s[3] + s[2]
            if var in log_norm_cols:
                col = (np.power(10.0, col) - 0.1) ** 2
            out[:, k] = col
    return out


class HydroScaler:
    """Fit-transform scaler for hydrology data (forcing series + static attributes).

    Parameters
    ----------
    attr_vars : list[str]   — names of static attribute columns
    series_vars : list[str] — names of time-series forcing columns
    log_norm_cols : list[str] — variables to apply log-gamma transform before normalizing
    """

    def __init__(self, attr_vars: list, series_vars: list, log_norm_cols: list = None):
        self.attr_vars = attr_vars
        self.series_vars = series_vars
        self.log_norm_cols = log_norm_cols or []
        self.stat_dict: dict = {}

    def fit(self, attr_data: np.ndarray, series_data: np.ndarray) -> None:
        """Compute statistics from training data."""
        self.stat_dict = _build_stat_dict(
            self.attr_vars, attr_data, self.series_vars, series_data, self.log_norm_cols
        )

    def transform(self, data: np.ndarray, var_list: list) -> np.ndarray:
        if not self.stat_dict:
            raise RuntimeError("HydroScaler.fit() must be called before transform().")
        return _normalize(data, var_list, self.stat_dict, self.log_norm_cols)

    def inverse_transform(self, data: np.ndarray, var_list: list) -> np.ndarray:
        if not self.stat_dict:
            raise RuntimeError("HydroScaler.fit() must be called before inverse_transform().")
        return _denormalize(data, var_list, self.stat_dict, self.log_norm_cols)

    def fit_transform(self, attr_data: np.ndarray, series_data: np.ndarray):
        """Fit on training data and return (norm_attrs, norm_series)."""
        self.fit(attr_data, series_data)
        return self.transform(attr_data, self.attr_vars), self.transform(series_data, self.series_vars)
