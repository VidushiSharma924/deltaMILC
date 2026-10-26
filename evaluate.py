
"""Evaluation / inference for dMILC."""

import logging
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils.metrics import compute_metrics, print_metrics_summary

logger = logging.getLogger(__name__)



def _prediction_date_columns(prediction_dates, warmup, pred_length):
    """
    Return dates aligned EXACTLY with prediction output.
    """
    if prediction_dates is None:
        return None

    dates = pd.to_datetime(prediction_dates)


    dates = dates[warmup : warmup + pred_length]

    if len(dates) != pred_length:
        raise ValueError(
            f"Date mismatch: got {len(dates)}, expected {pred_length}"
        )

    return dates.strftime("%Y-%m-%d").tolist()

def _build_z_full(
    z: np.ndarray,
    c: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Concatenate normalized forcings with repeated static attrs.

    z : [B, T, nz]
    c : [B, nc]
    → [T, B, nz+nc]
    """
    B, T, nz = z.shape
    nc = c.shape[-1]
    c_rep = np.tile(c[:, np.newaxis, :], (1, T, 1))     
    zc    = np.concatenate([z, c_rep], axis=2)           
    return torch.from_numpy(zc.swapaxes(0, 1)).float().to(device)

   
def run_inference(
    model: nn.Module,
    x: np.ndarray,
    z: np.ndarray,
    c: np.ndarray,
    month_tensor: torch.Tensor,
    area_tensor: torch.Tensor,
    lat_tensor,
    doy_tensor,
    elev_tensor,
    batch_size: int = 10,
    gpu: int = 0,
) -> np.ndarray:
    """Run model inference in batches.

    Returns
    -------
    pred : [B, T, 4]  — (Qs, qd, qb, et) in mm/day
    """
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    warmup = getattr(model, "warmup", 0)  

    B, T, _ = x.shape
    all_preds = []
    print("x:", x.shape)
    print("z:", z.shape)
    print("c:", c.shape)

    z_full = _build_z_full(z, c, device)   
    print("final z_full:", z_full.shape)

    with torch.no_grad():
        print("B =", B)
        print("batch_size =", batch_size)
        for start in range(0, B, batch_size):

            end = min(start + batch_size, B)
            logger.debug("Inference batch %d–%d", start, end)

            x_b     = torch.from_numpy(x[start:end].swapaxes(0, 1)).float().to(device)
            z_b     = z_full[:, start:end, :]
            month_b = month_tensor[start:end, warmup:T].to(device) 
            area_b  = area_tensor[start:end].to(device)
            elev_b  = elev_tensor[start:end].to(device)
            lat_b   = lat_tensor[start:end].to(device)
            doy_b   = doy_tensor[start:end, warmup:T].to(device)   

            out = model(
                x_b,
                z_b,
                month_b,
                area_b,
                lat=lat_b,
                elev=elev_b,
                doy=doy_b,
            )
            
            all_preds.append(out.cpu().numpy().swapaxes(0, 1)) 

    return np.concatenate(all_preds, axis=0) 



def evaluate(
    model: nn.Module,
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    month_tensor: torch.Tensor,
    area_tensor: torch.Tensor,
    lat_tensor, doy_tensor, elev_tensor,
    save_folder: Optional[str] = None,
    skip_warmup_days: int = 0,
    batch_size: int = 10,
    gpu: int = 0,
    basin_ids: Optional[List[str]] = None,
    prediction_dates: Optional[List] = None,
    y_std: np.ndarray = None,
    basin_area_km2=None,
) -> dict:
    """Run inference, compute metrics, optionally save predictions.

    Metrics and saved CSVs are in physical units (mm/day for Qs/qd/qb,
    ET is already physical since it was never divided by y_std).
    """
    assert batch_size > 0, f"batch_size corrupted: {batch_size}"

    pred = run_inference(
        model=model,
        x=x,
        z=z,
        c=c,
        month_tensor=month_tensor,
        area_tensor=area_tensor,
        lat_tensor=lat_tensor,
        doy_tensor=doy_tensor,
        elev_tensor=elev_tensor,
        batch_size=batch_size,
        gpu=gpu,
    )

    if y_std is None:
        raise ValueError(
            "y_std must be passed to evaluate() — predictions and targets "
            "are in normalized units and cannot be scored/saved without it."
        )

    y_std_arr = np.asarray(y_std)
    if y_std_arr.ndim == 3:
        scale = y_std_arr[:, 0, 0][:, None]   
    elif y_std_arr.ndim == 2:
        scale = y_std_arr[:, 0][:, None]     
    else:
        scale = y_std_arr[:, None]            

    if scale.shape[0] != pred.shape[0]:
        raise ValueError(
            f"y_std basin count {scale.shape[0]} does not match "
            f"prediction basin count {pred.shape[0]}"
        )

    pred_physical = pred.copy()
    pred_physical[:, :, :3] *= scale[:, :, None]   
    pred_physical[:, :, 3]  = pred[:, :, 3]      
    y_physical = y.copy()
    y_physical[:, :, 0] *= scale                   

    model_warmup = getattr(model, "warmup", 0)
    if model_warmup < 0:
        raise ValueError(
            f"Prediction is longer than target: pred={pred.shape[1]} target={y.shape[1]}"
        )

    qs_pred   = pred_physical[:, skip_warmup_days:, 0]
    qs_target = y_physical[:, model_warmup + skip_warmup_days:, 0]
    if qs_pred.shape != qs_target.shape:
        raise ValueError(
            f"Prediction/target length mismatch after alignment: "
            f"pred={qs_pred.shape} target={qs_target.shape} "
            f"model_warmup={model_warmup} skip={skip_warmup_days}"
        )

    metrics = compute_metrics(qs_pred, qs_target)
    print_metrics_summary(metrics)

    if save_folder:
        os.makedirs(save_folder, exist_ok=True)

        if basin_ids is None:
            basin_ids = [str(i) for i in range(pred_physical.shape[0])]

        if len(basin_ids) != pred_physical.shape[0]:
            raise ValueError(
                f"basin_ids length {len(basin_ids)} "
                f"does not match predictions {pred_physical.shape[0]}"
            )

        columns = _prediction_date_columns(
            prediction_dates,
            model_warmup,
            pred_physical.shape[1],
        )

        print("Pred shape:", pred_physical.shape)
        print("Dates length:", len(columns) if columns is not None else None)
        if columns is not None:
            print("First date:", columns[0])
            print("Last date:", columns[-1])

        for name, data in [
            ("Qs", pred_physical[:, :, 0]),
            ("qd", pred_physical[:, :, 1]),
            ("qb", pred_physical[:, :, 2]),
            ("ET", pred_physical[:, :, 3]),
        ]:
            out_path = os.path.join(save_folder, f"{name}.csv")
            pd.DataFrame(
                data,
                index=basin_ids,
                columns=columns,
            ).to_csv(
                out_path,
                index=True,
                index_label="basin_id",
                header=True,
            )
            logger.info("Saved %s → %s", name, out_path)

        metric_df = pd.DataFrame({k: v for k, v in metrics.items()}, index=basin_ids)
        metric_df.to_csv(os.path.join(save_folder, "metrics.csv"), index_label="basin_id")
        logger.info("Metrics saved → %s", os.path.join(save_folder, "metrics.csv"))

    return metrics



def load_checkpoint(checkpoint_path: str, gpu: int = 0) -> nn.Module:
    """Load a saved model checkpoint."""
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    logger.info("Loaded checkpoint from %s", checkpoint_path)
    return model
