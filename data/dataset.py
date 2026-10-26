"""CAMELS dataset loading and caching utilities for dMILC."""

import glob
import logging
import os
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

# =============================================================================
# UNIT CONVERSION
# =============================================================================

def basin_norm(
    x: np.ndarray,
    basin_area: np.ndarray,
    to_norm: bool
) -> np.ndarray:
    """
    Convert streamflow between m³/s and mm/day.

    Parameters
    ----------
    x          : [B,T] or [B,T,1]
    basin_area : [B,1] in km²
    to_norm    : True  -> m³/s → mm/day
                 False -> mm/day → m³/s
    """

    if np.isnan(x).any():
        raise ValueError("NaN detected in streamflow input")

    if np.isnan(basin_area).any():
        raise ValueError("NaN detected in basin_area")

    if np.any(basin_area <= 0):
        raise ValueError("Invalid basin area <= 0")

    nd = x.ndim

    if nd == 3:
        if x.shape[2] != 1:
            raise ValueError(f"Expected last dim=1, got {x.shape}")
        x = x[:, :, 0]

    if basin_area.ndim != 2 or basin_area.shape[1] != 1:
        raise ValueError(
            f"basin_area must be [B,1], got {basin_area.shape}"
        )

    if x.shape[0] != basin_area.shape[0]:
        raise ValueError(
            f"Basin mismatch x={x.shape[0]} area={basin_area.shape[0]}"
        )

    area_tiled = np.tile(basin_area, (1, x.shape[1]))

    if to_norm:
        # m³/s → mm/day
        flow = (x * 86400.0) / (area_tiled * 1e6) * 1e3
    else:
        # mm/day → m³/s
        flow = x * (area_tiled * 1e6) / (86400.0 * 1e3)

    if np.isnan(flow).any():
        raise ValueError("NaN generated during unit conversion")

    if np.isinf(flow).any():
        raise ValueError("Inf generated during unit conversion")

    return np.expand_dims(flow, axis=2) if nd == 3 else flow



def build_area_month_tensors(
    basin_ids,
    forcing_root,
    attr_path,
    t_range=None,         
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor
]:
    """
    Build tensors used by MILC.

    Returns
    -------
    area_tensor  : [B,1] km²
    month_tensor : [B,T]
    lat_tensor   : [B]
    elev_tensor  : [B]
    doy_tensor   : [B,T]
    """

    forcing_files = sorted(glob.glob(os.path.join(forcing_root, "*.csv")))

    if not forcing_files:
        raise FileNotFoundError(
            f"No forcing files found under {forcing_root}"
        )

    file_map = {
        os.path.basename(f).split(".")[0]: f
        for f in forcing_files
    }

    def _camels_ind_id(bid):
        return str(int(bid)).zfill(5)

    mapped_ids = [_camels_ind_id(bid) for bid in basin_ids]

    missing = [bid for bid in mapped_ids if bid not in file_map]

    if missing:
        raise KeyError(
            f"Missing forcing files for basins: {missing[:5]}"
        )

    selected = [file_map[bid] for bid in mapped_ids]

    # =========================================================================
    # REFERENCE TIME AXIS
    # =========================================================================

    df0 = pd.read_csv(selected[0])

    date0 = pd.to_datetime(df0[["year", "month", "day"]])

    # ─── APPLY t_range SLICE ────────────────────────────────────────
    if t_range is not None:
        start_date = pd.to_datetime(str(t_range[0]), format="%Y%m%d")
        end_date   = pd.to_datetime(str(t_range[1]), format="%Y%m%d")

        keep_mask = (date0 >= start_date) & (date0 < end_date)

        if keep_mask.sum() == 0:
            raise ValueError(
                f"No forcing data in t_range {t_range} for reference basin "
                f"{mapped_ids[0]}"
            )

        df0 = df0.loc[keep_mask].reset_index(drop=True)
        date0 = date0.loc[keep_mask].reset_index(drop=True)
    
    print("\n=== REFERENCE TIME AXIS ===")
    print("Reference basin:", mapped_ids[0])
    print("Start:", date0.iloc[0])
    print("End:", date0.iloc[-1])

    month_arr = df0["month"].values.astype(np.int32)
    doy_arr = date0.dt.dayofyear.values.astype(np.int32)

    T = len(month_arr)

    # =========================================================================
    # VALIDATE ALL FORCING FILES
    # =========================================================================

    for fp in selected[1:]:

        df = pd.read_csv(fp)

        date = pd.to_datetime(df[["year", "month", "day"]])

        if t_range is not None:
            keep_mask = (date >= start_date) & (date < end_date)
            df = df.loc[keep_mask].reset_index(drop=True)
            date = date.loc[keep_mask].reset_index(drop=True)

        if len(df) != T:
            raise ValueError(f"Length mismatch in {fp}")

        if not np.array_equal(date.values, date0.values):
            raise ValueError(f"Time axis mismatch in {fp}")

    # =========================================================================
    # LOAD ATTRIBUTES
    # =========================================================================

    attr = pd.read_csv(attr_path)

    required_cols = [
        "gauge_id",
        "cwc_area",
        "cwc_lat",
        "elev_mean"
    ]
    

    for col in required_cols:
        if col not in attr.columns:
            raise ValueError(f"Missing column {col}")

    attr["gauge_id"] = (
        attr["gauge_id"]
        .astype(str)
        .str.zfill(5)
    )

    area_dict = dict(zip(attr["gauge_id"], attr["cwc_area"]))
    lat_dict = dict(zip(attr["gauge_id"], attr["cwc_lat"]))
    elev_dict = dict(zip(attr["gauge_id"], attr["elev_mean"]))

    missing_attr = [
        bid for bid in mapped_ids
        if bid not in area_dict
    ]

    if missing_attr:
        raise KeyError(
            f"Missing attributes for basins: {missing_attr[:5]}"
        )

    areas = np.array(
        [area_dict[bid] for bid in mapped_ids],
        dtype=np.float32
    )

    lats = np.array(
        [lat_dict[bid] for bid in mapped_ids],
        dtype=np.float32
    )

    elevs = np.array(
        [elev_dict[bid] for bid in mapped_ids],
        dtype=np.float32
    )

    # =========================================================================
    # AREA VALIDATION
    # =========================================================================

    print("\n=== AREA VALIDATION ===")

    print("Min area :", np.min(areas))
    print("Max area :", np.max(areas))
    print("Mean area:", np.mean(areas))

    if np.any(areas <= 0):
        raise ValueError("Invalid basin area <= 0")

    if np.any(areas > 1e7):
        raise ValueError(
            "Area values unrealistically large. "
            "Possible m² instead of km²."
        )

    if np.isnan(areas).any():
        raise ValueError("NaN in basin areas")

    if np.isinf(areas).any():
        raise ValueError("Inf in basin areas")

    # =========================================================================
    # BUILD TENSORS
    # =========================================================================

    B = len(mapped_ids)

    area_tensor = torch.from_numpy(
        areas.astype(np.float32)
    ).unsqueeze(-1)

    lat_tensor = torch.from_numpy(
        lats.astype(np.float32)
    )

    elev_tensor = torch.from_numpy(elevs.astype(np.float32))

    month_tensor = (
        torch.from_numpy(month_arr)
        .unsqueeze(0)
        .repeat(B, 1)
        .long()
    )

    doy_tensor = (
        torch.from_numpy(doy_arr)
        .unsqueeze(0)
        .repeat(B, 1)
        .long()
    )

    print("\n=== FINAL TENSOR SHAPES ===")
    print("area_tensor :", area_tensor.shape)
    print("month_tensor:", month_tensor.shape)
    print("doy_tensor  :", doy_tensor.shape)

    return (
        area_tensor,
        month_tensor,
        lat_tensor,
        elev_tensor,
        doy_tensor,
    )

def build_doy_tensor(forcing_files, basin_ids, t_range=None):
    file_map = {os.path.basename(f).split(".")[0]: f for f in forcing_files}

    def _id(bid):
        return str(int(bid)).zfill(5)

    mapped_ids = [_id(bid) for bid in basin_ids]

    missing = [bid for bid in mapped_ids if bid not in file_map]
    if missing:
        raise KeyError(f"Missing forcing files for basins: {missing[:5]}")

    # --- reference ---
    df0 = pd.read_csv(file_map[mapped_ids[0]])
    date0 = pd.to_datetime(df0[["year","month","day"]])

    if t_range is not None:
        start_date = pd.to_datetime(str(t_range[0]), format="%Y%m%d")
        end_date   = pd.to_datetime(str(t_range[1]), format="%Y%m%d")

        keep_mask = (date0 >= start_date) & (date0 < end_date)

        if keep_mask.sum() == 0:
            raise ValueError(
                f"No forcing data in t_range {t_range} for reference basin "
                f"{mapped_ids[0]}"
            )

        df0 = df0.loc[keep_mask].reset_index(drop=True)
        date0 = date0.loc[keep_mask].reset_index(drop=True)


    doy = date0.dt.dayofyear.values.astype(np.int32)
    T = len(doy)

    # --- validation ---
    for bid in mapped_ids[1:]:
        df = pd.read_csv(file_map[bid])
        date = pd.to_datetime(df[["year","month","day"]])

        if t_range is not None:
            keep_mask = (date >= start_date) & (date < end_date)
            df = df.loc[keep_mask].reset_index(drop=True)
            date = date.loc[keep_mask].reset_index(drop=True)

        if len(df) != T:
            raise ValueError(f"Length mismatch for basin {bid}")

        if not np.array_equal(date.values, date0.values):
            raise ValueError(f"Time mismatch for basin {bid}")

    B = len(basin_ids)

    doy_tensor = torch.from_numpy(doy).unsqueeze(0).repeat(B, 1).long()

    return doy_tensor

    
class CAMELSDataset:
    def __init__(
        self,
        root_db: str,
        forcing_root: str,
        attr_path: str,
        flow_path: str = None,   
        pet_subdir: str = "pet_harg",
        pet_range: list = None,
    ):
        self.root_db = root_db
        self.forcing_root = forcing_root
        self.flow_path = flow_path       
        self.attr_path = attr_path       
        self.pet_subdir = pet_subdir
        self.pet_range = pet_range
        

    def extract(self, basin_ids, t_range, attr_vars, forcing_vars):
        x_list = []
        y_list = []
        qmask_list = []          
        valid_ids_out = []
        reference_dates = None
        full_date_index = None

        full_flow_df = None
        if self.flow_path:
            full_flow_df = pd.read_csv(self.flow_path)
            full_flow_df["Date"] = pd.to_datetime(full_flow_df["Date"])
            full_flow_df["Basin_ID"] = full_flow_df["Basin_ID"].astype(str).str.zfill(5)

        for basin in basin_ids:
            basin_str = str(int(basin)).zfill(5)
            fp = f"{self.forcing_root}/{basin_str}.csv"
            if not os.path.exists(fp):
                raise FileNotFoundError(fp)

            df = pd.read_csv(fp, encoding="utf-8-sig")
            df.columns = df.columns.str.strip()
            df["date"] = pd.to_datetime(df[["year", "month", "day"]])
            if df["date"].duplicated().any():
                raise ValueError(f"Duplicate dates in basin {basin_str}")
            if not df["date"].is_monotonic_increasing:
                raise ValueError(f"Dates not sorted in basin {basin_str}")
            df = df.set_index("date")

            start_date = pd.to_datetime(str(t_range[0]), format="%Y%m%d")
            end_date   = pd.to_datetime(str(t_range[1]), format="%Y%m%d")
            df = df.loc[(df.index >= start_date) & (df.index < end_date)]
            if len(df) == 0:
                raise ValueError(
                    f"No forcing data for basin {basin_str} "
                    f"between {start_date.date()} and {end_date.date()}"
                )

            if reference_dates is None:
                reference_dates = df.index
                full_date_index = df.index
            else:
                if not df.index.equals(full_date_index):
                    raise ValueError(f"Forcing date mismatch in basin {basin_str}")

            required_cols = ["prcp(mm/day)", "tmax(C)", "tmin(C)",
                            "wind(m/s)", "rel_hum(%)", "srad_sw(w/m2)"]
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"{col} missing in basin {basin_str}")

            df = df.rename(columns={
                "prcp(mm/day)": "prcp", "tmax(C)": "tmax", "tmin(C)": "tmin",
                "tavg(C)": "tavg", "wind(m/s)": "wind", "rel_hum(%)": "rel_hum",
                "srad_sw(w/m2)": "srad_sw", "srad_lw(w/m2)": "srad_lw",
            })

            if "tavg" not in df.columns:
                df["tavg"] = (df["tmax"] + df["tmin"]) / 2

            bad_temp = df["tmax"] < df["tmin"]
            if np.any(bad_temp):
                df.loc[bad_temp, ["tmax", "tmin"]] = np.nan
                df["tmax"] = df["tmax"].interpolate(limit_direction="both")
                df["tmin"] = df["tmin"].interpolate(limit_direction="both")
                df["tavg"] = (df["tmax"] + df["tmin"]) / 2

            srad_lw = df["srad_lw"].values if "srad_lw" in df.columns else np.zeros(len(df))

            x_basin = np.stack([
                df["prcp"].values, df["tmax"].values, df["tmin"].values,
                df["tavg"].values, df["srad_sw"].values, srad_lw,
                df["wind"].values, df["rel_hum"].values,
            ], axis=1)

            if np.isnan(x_basin).any():
                raise ValueError(f"NaN forcings in basin {basin_str}")

            if "Q" in df.columns:
                q_series = pd.to_numeric(
                    df["Q"].astype(str).str.strip(), errors="coerce"
                ).astype(np.float32).reindex(full_date_index)
                q_mask_series = (~q_series.isna()).astype(np.float32)
            else:
                if full_flow_df is None:
                    raise ValueError(
                        f"Q missing in basin {basin_str} forcing file, and no "
                        f"flow_path was provided as a fallback source."
                    )
                basin_flow = full_flow_df[full_flow_df["Basin_ID"] == basin_str]
                if basin_flow.empty:
                    raise ValueError(f"Basin {basin_str} has no Q in forcing file or flow_path")
                basin_flow = basin_flow.set_index("Date")

                q_col = "Q_filled" if "Q_filled" in basin_flow.columns else "Q"
                q_series = pd.to_numeric(
                    basin_flow[q_col], errors="coerce"
                ).astype(np.float32).reindex(full_date_index)

                if "mask" in basin_flow.columns:
                    q_mask_series = pd.to_numeric(
                        basin_flow["mask"], errors="coerce"
                    ).astype(np.float32).reindex(full_date_index)
                    q_mask_series = q_mask_series.fillna(0.0)
                else:
                    q_mask_series = (~q_series.isna()).astype(np.float32)

            n_missing = (q_mask_series == 0).sum()
   
            x_list.append(x_basin)
            y_list.append(q_series.values)
            qmask_list.append(q_mask_series.values)
            valid_ids_out.append(str(int(basin)))

        # ── STACK ─────────────────────────────────────────────────────
        x     = np.stack(x_list, axis=0)
        y_raw = np.stack(y_list, axis=0)[:, :, np.newaxis]
        mask  = np.stack(qmask_list, axis=0)[:, :, np.newaxis]
        mask  = np.nan_to_num(mask, nan=0.0)

        print("\n=== MASK VALIDATION ===")
        print("Valid fraction:", np.mean(mask))
        per_basin = mask[:, :, 0].mean(axis=1)
        print("Per-basin valid fraction (min/mean/max): "
            f"{per_basin.min():.3f} / {per_basin.mean():.3f} / {per_basin.max():.3f}")
        if np.mean(mask) < 0.3:
            raise ValueError(f"Too much missing streamflow: {np.mean(mask):.3f}")

        attr = pd.read_csv(self.attr_path)
        attr["gauge_id"] = attr["gauge_id"].astype(str).str.zfill(5)
        attr = attr.set_index("gauge_id")

        valid_ids_zfill = [str(int(b)).zfill(5) for b in valid_ids_out]
        missing_attr = [b for b in valid_ids_zfill if b not in attr.index]
        if missing_attr:
            raise KeyError(f"Missing attributes for basins: {missing_attr[:5]}")

        c = attr.loc[
            valid_ids_zfill,
            [v for v in attr_vars if v in attr.columns]
        ].values.astype(np.float32)

        if "cwc_area" not in attr.columns:
            raise ValueError("cwc_area missing from attributes")

        basin_area = attr.loc[valid_ids_zfill, ["cwc_area"]].values.astype(np.float32)
        if np.any(basin_area <= 0):
            raise ValueError("Invalid basin area detected")

        y_filled = np.where(np.isnan(y_raw), 0.0, y_raw)
        y_mm = basin_norm(y_filled, basin_area, to_norm=True)
        y_mm[mask == 0] = np.nan         

        print("\n=== STREAMFLOW UNIT CHECK ===")
        print(f"Raw mean m³/s (obs only): {np.nansum(np.where(mask == 1, y_raw, np.nan)) / np.sum(mask):.3f}")
        print(f"Conv mean mm/day (obs only): {np.nanmean(y_mm):.3f}")

        runoff_ratio = np.nanmean(y_mm) / np.mean(x[..., 0])
        print(f"\nRunoff ratio (obs-only Q): {runoff_ratio:.3f}")
        if not np.isfinite(runoff_ratio) or runoff_ratio <= 0 or runoff_ratio > 1.5:
            raise ValueError(f"Runoff ratio out of range: {runoff_ratio:.3f}")

        print("\n=== FINAL SHAPES ===")
        print("X:", x.shape, "Y:", y_mm.shape, "C:", c.shape, "Mask:", mask.shape)

        return x, y_mm, c, mask, valid_ids_out
    


# =============================================================================
# CACHE
# =============================================================================

def load_or_build_cache(
    dataset: CAMELSDataset,
    cache_path: str,
    basin_ids: List[str],
    t_range: List[int],
    attr_vars: List[str],
    forcing_vars: List[str],
    rebuild: bool = False,
):
    print("\n=== CACHE STATUS ===")
    print("REBUILD CACHE:", rebuild)
    print("CACHE PATH:", cache_path)

    if not rebuild and os.path.exists(cache_path):

        logger.info("Loading cache: %s", cache_path)

        with open(cache_path, "rb") as f:
            x, y, c, mask, valid_ids = pickle.load(f)

        logger.info(
            "Loaded x%s y%s c%s mask%s",
            x.shape,
            y.shape,
            c.shape,
            mask.shape,
        )

        return x, y, c, mask, valid_ids

    logger.info("Building cache → %s", cache_path)

    x, y, c, mask, valid_ids = dataset.extract(
        basin_ids,
        t_range,
        attr_vars,
        forcing_vars,
    )
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump((x, y, c, mask, valid_ids), f)  
        

    logger.info("Cache saved: %s", cache_path)

    return x, y, c, mask, valid_ids


