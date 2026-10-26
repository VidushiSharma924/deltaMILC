"""Training engine for dMILC."""

import logging
import os
import time
from typing import Optional, Tuple


import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mini-batch helpers
# ---------------------------------------------------------------------------

def _random_indices(
    n_basins: int,
    n_time: int,
    batch_size: int,
    seq_len: int,
    warmup: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample random (basin, start-time) indices for a mini-batch."""
    i_grid = np.random.randint(0, n_basins, [batch_size])
    i_t    = np.random.randint(warmup, n_time - seq_len, [batch_size])
    return i_grid, i_t


def _select_subset(
    data: np.ndarray,
    i_grid: np.ndarray,
    i_t: np.ndarray,
    seq_len: int,
    warmup: int = 0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Extract a [T+warmup, B, F] tensor from [B, T, F] data array."""
    B, T, F = data.shape if data.ndim == 3 else (*data.shape, 1)
    total = seq_len + warmup
    batch = len(i_grid)

    out = np.zeros((total, batch, data.shape[-1] if data.ndim == 3 else 1))
    for k in range(batch):
        start = i_t[k] - warmup
        chunk = data[i_grid[k], start : start + total]
        if data.ndim == 3:
            out[:, k, :] = chunk
        else:
            out[:, k, 0] = chunk

    t = torch.from_numpy(out).float()
    if device is not None:
        t = t.to(device)
    return t


def _select_z(
    z_data: np.ndarray,
    c_data: np.ndarray,
    i_grid: np.ndarray,
    i_t: np.ndarray,
    seq_len: int,
    warmup: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build [T+warmup, B, nz+nc] LSTM input by concatenating z and repeated c."""
    z_sub = _select_subset(z_data, i_grid, i_t, seq_len, warmup, device)  
    nc    = c_data.shape[-1]
    batch = len(i_grid)
    total = seq_len + warmup

    c_rep = c_data[i_grid]                                     
    c_rep = np.tile(c_rep[:, np.newaxis, :], (1, total, 1))    
    c_rep = c_rep.swapaxes(0, 1)                              
    c_t   = torch.from_numpy(c_rep).float()
    if device is not None:
        c_t = c_t.to(device)
    return torch.cat([z_sub, c_t], dim=2)


def _select_month_area(
    month_tensor: torch.Tensor,
    area_tensor: torch.Tensor,
    i_grid: np.ndarray,
    i_t: np.ndarray,
    seq_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slice month [B, seq_len] and area [B] for the current batch."""
    month_batch = torch.stack([
        month_tensor[i_grid[k], i_t[k] : i_t[k] + seq_len]
        for k in range(len(i_grid))
    ]).to(device)                                             
    area_batch = area_tensor[i_grid].to(device)             
    return month_batch, area_batch

def _select_geo(
    lat_tensor,
    elev_tensor,
    doy_tensor,
    i_grid,
    i_t,
    seq_len,
    device,
):
    lat_b = lat_tensor[i_grid].to(device)
    elev_b = elev_tensor[i_grid].to(device)

    doy_b = torch.stack([
        doy_tensor[i_grid[k], i_t[k]: i_t[k] + seq_len]
        for k in range(len(i_grid))
    ]).to(device)

    return lat_b, elev_b, doy_b


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _val_metrics(
    model: nn.Module,
    x: np.ndarray,              
    z: np.ndarray,              
    y: np.ndarray,              
    c: np.ndarray,              
    month_tensor: torch.Tensor, 
    doy_tensor: torch.Tensor, 
    area_tensor: torch.Tensor,  
    lat_tensor: torch.Tensor,
    elev_tensor: torch.Tensor,
    device: torch.device,
    val_batch_size: int = 50,
    skip_days: int = 0,
    y_std: Optional[np.ndarray] = None,
    basin_ids: Optional[list] = None,
) -> dict:

    from utils.metrics import compute_metrics

    model.eval()
    preds_all = []

    B, T, _ = x.shape
    nc = c.shape[-1]

    warmup = getattr(model, "warmup", 0)

    c_rep = np.repeat(c[:, None, :], T, axis=1)   
    zc = np.concatenate([z, c_rep], axis=-1)      

    zc_t = torch.from_numpy(zc).float().permute(1, 0, 2)  

    with torch.no_grad():
        for start in range(0, B, val_batch_size):
            end = min(start + val_batch_size, B)

            x_b = torch.from_numpy(
                x[start:end].transpose(1, 0, 2)
            ).float().to(device) 

            z_b = zc_t[:, start:end, :].to(device)  

            month_b = month_tensor[start:end, warmup:T].to(device)
            doy_b   = doy_tensor[start:end, warmup:T].to(device)

            area_b  = area_tensor[start:end].to(device)
            lat_b   = lat_tensor[start:end].to(device)
            elev_b  = elev_tensor[start:end].to(device)

            out = model(
                x_b,
                z_b,
                month_b,
                area_b,
                lat=lat_b,
                elev=elev_b,
                doy=doy_b,
            ) 

            out = out.detach().cpu().numpy().transpose(1, 0, 2)  
            preds_all.append(out)

    pred_np = np.concatenate(preds_all, axis=0)  

    T_out = pred_np.shape[1]
    obs = y[:, warmup:warmup + T_out, 0]
    pred = pred_np[:, :, 0]

    if skip_days > 0:
        if skip_days >= T_out:
            raise ValueError(
                f"Validation skip_days={skip_days} leaves no data for T_out={T_out}."
            )
        obs = obs[:, skip_days:]
        pred = pred[:, skip_days:]

    valid_mask = np.isfinite(obs) & np.isfinite(pred)
    pred = np.where(valid_mask, pred, np.nan)
    obs  = np.where(valid_mask, obs, np.nan)

    metrics = compute_metrics(pred, obs)
    summary = {k: float(np.nanmedian(v)) for k, v in metrics.items()}
    ids = basin_ids if basin_ids is not None else np.arange(B)

    model.train()

    return {
        "summary": summary,
        "basin": {
            "basin_ids": ids,
            **metrics
        }
    }


def train(
    model: nn.Module,
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    loss_fn: nn.Module,
    month_tensor: torch.Tensor,
    doy_tensor: torch.Tensor,
    area_tensor: torch.Tensor,
    lat_tensor: torch.Tensor,
    elev_tensor: torch.Tensor,
    y_std: Optional[np.ndarray] = None,
    basin_ids: Optional[list] = None,
    *,
    x_va: Optional[np.ndarray] = None,
    z_va: Optional[np.ndarray] = None,
    y_va: Optional[np.ndarray] = None,
    c_va: Optional[np.ndarray] = None,
    month_tensor_va: Optional[torch.Tensor] = None,
    doy_tensor_va: Optional[torch.Tensor] = None,
    val_skip_days: int = 0,
    val_batch_size: int = 50,
    grad_clip: float = 1.0,
    start_epoch: int = 0,
    n_epochs: int = 100,
    batch_size: int = 200,
    seq_len: int = 365,
    warmup: int = 365,
    save_every: int = 10,
    save_folder: Optional[str] = None,
    optimizer_name: str = "adam",
    lr: float = 5e-4,
    lr_scheduler: str = "cosine",
    lr_min: float = 1e-6,
    gpu: int = 0,
) -> nn.Module:
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    model = model.to(device)
    loss_fn = loss_fn.to(device)

    if optimizer_name == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optim = torch.optim.Adadelta(model.parameters())

    if lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=n_epochs, eta_min=lr_min
        )
    else:
        scheduler = None

    n_basins, n_time, _ = x.shape
    effective_batch = min(batch_size, n_basins)

    coverage = min(
        0.99,
        effective_batch * seq_len /
        max(1, n_basins * max(1, n_time - warmup))
    )

    n_iter_ep = max(
        1,
        int(np.ceil(np.log(0.01) / np.log(max(1e-6, 1.0 - coverage))))
    )
    logger.info(
        "n_basins=%d  n_time=%d  batch=%d  seq_len=%d  iters/epoch=%d",
        n_basins, n_time, effective_batch, seq_len, n_iter_ep,
    )

    has_val = all(
        v is not None
        for v in [
            x_va,
            z_va,
            y_va,
            c_va,
            month_tensor_va,
            doy_tensor_va,
        ]
    )
    best_kge = -float("inf")

    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        log_path = os.path.join(save_folder, "train_log.csv")
        log_f = open(log_path, "w")
        header = "epoch,train_loss,time_s"
        if has_val:
            header += ",val_NSE,val_KGE,val_Corr,val_RMSE"
        log_f.write(header + "\n")

    from model.loss import NSELossBatch, CombinedKGENSELoss

    epoch_bar = tqdm(range(start_epoch + 1, start_epoch + n_epochs + 1), desc="Training", unit="epoch",
                     initial=start_epoch, total=start_epoch + n_epochs)
    for epoch in epoch_bar:
        model.train()
        epoch_loss = 0.0
        n_updates = 0
        n_skipped = 0
        t0 = time.time()

        iter_bar = tqdm(range(n_iter_ep), desc=f"  Ep {epoch:3d}/{start_epoch+n_epochs}", leave=False,
                        unit="iter", dynamic_ncols=True)
        for _ in iter_bar:
            i_grid, i_t = _random_indices(n_basins, n_time, effective_batch, seq_len, warmup)

            x_batch = _select_subset(x, i_grid, i_t, seq_len, warmup, device)
            z_batch = _select_z(z, c, i_grid, i_t, seq_len, warmup, device)
            y_batch = _select_subset(y, i_grid, i_t, seq_len, 0, device)
            month_b, area_b = _select_month_area(month_tensor, area_tensor, i_grid, i_t, seq_len, device)

            lat_b, elev_b, doy_b = _select_geo(
                lat_tensor, elev_tensor, doy_tensor,
                i_grid, i_t, seq_len, device,
            )

            valid_frac = torch.isfinite(y_batch).float().mean().item()
            if valid_frac < 0.3:
                n_skipped += 1
                continue

            y_pred = model(
                x_batch, z_batch, month_b, area_b,
                lat=lat_b, elev=elev_b, doy=doy_b,
            )

            if not torch.isfinite(y_pred).all():
                n_skipped += 1
                continue

            if isinstance(loss_fn, (NSELossBatch, CombinedKGENSELoss)):
                loss = loss_fn(y_pred, y_batch, i_grid)
            else:
                loss = loss_fn(y_pred, y_batch)

            if not torch.isfinite(loss):
                n_skipped += 1
                optim.zero_grad(set_to_none=True)
                continue

            optim.zero_grad(set_to_none=True)
            loss.backward()

            bad_grad = any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
            if bad_grad:
                n_skipped += 1
                optim.zero_grad(set_to_none=True)
                continue

            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            if not torch.isfinite(total_norm):
                n_skipped += 1
                optim.zero_grad(set_to_none=True)
                continue

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            optim.zero_grad(set_to_none=True)
            epoch_loss += loss.item()
            n_updates += 1
            iter_bar.set_postfix(loss=f"{loss.item():.4f}")

        if n_updates == 0:
            raise RuntimeError(
                f"No finite training batches in epoch {epoch}; skipped {n_skipped}/{n_iter_ep} iterations."
            )
        epoch_loss /= n_updates
        if n_skipped:
            logger.warning(
                "Epoch %d skipped %d/%d non-finite or mostly-missing batches.",
                epoch, n_skipped, n_iter_ep,
            )
        elapsed = time.time() - t0
        lr_now  = optim.param_groups[0]["lr"]

        val_m = {}
        if has_val:
            val_out = _val_metrics(
                model, x_va, z_va, y_va, c_va,
                month_tensor_va, doy_tensor_va, area_tensor,
                lat_tensor, elev_tensor,
                device,
                val_batch_size=val_batch_size,
                skip_days=val_skip_days,
                y_std=y_std,
                basin_ids=basin_ids,
            )
            val_m = val_out["summary"]
            tqdm.write(
                f"Epoch {epoch:3d}/{start_epoch+n_epochs}  train_loss={epoch_loss:.4f}  lr={lr_now:.2e}  {elapsed:.0f}s"
                f"  |  val  NSE={val_m['NSE']:+.3f}  KGE={val_m['KGE']:+.3f}"
                f"  Corr={val_m['Corr']:.3f}  RMSE={val_m['RMSE']:.3f}"
            )
            epoch_bar.set_postfix(
                loss=f"{epoch_loss:.4f}",
                NSE=f"{val_m['NSE']:+.3f}",
                KGE=f"{val_m['KGE']:+.3f}",
            )
        else:
            epoch_bar.set_postfix(loss=f"{epoch_loss:.4f}", lr=f"{lr_now:.2e}", s=f"{elapsed:.0f}s")

        logger.info(
            "Epoch %3d/%d  loss=%.4f  lr=%.2e  time=%.1fs%s",
            epoch, start_epoch + n_epochs, epoch_loss, lr_now, elapsed,
            f"  val_NSE={val_m['NSE']:.3f}  val_KGE={val_m['KGE']:.3f}"
            f"  val_Corr={val_m['Corr']:.3f}  val_RMSE={val_m['RMSE']:.3f}" if has_val else "",
        )

        if scheduler is not None:
            scheduler.step()

        if save_folder:
            row = f"{epoch},{epoch_loss:.6f},{elapsed:.1f}"
            if has_val:
                row += f",{val_m['NSE']:.4f},{val_m['KGE']:.4f},{val_m['Corr']:.4f},{val_m['RMSE']:.4f}"
            log_f.write(row + "\n")
            log_f.flush()
            if epoch % save_every == 0:
                ckpt = os.path.join(save_folder, f"model_Ep{epoch}.pt")
                torch.save(model, ckpt)
                logger.info("Checkpoint saved → %s", ckpt)
            if has_val and val_m.get("KGE", -float("inf")) > best_kge:
                best_kge = val_m["KGE"]
                best_path = os.path.join(save_folder, "model_best.pt")
                torch.save(model, best_path)
                tqdm.write(f"  ★ New best KGE={best_kge:.4f} → saved model_best.pt")
                logger.info("Best model saved (KGE=%.4f) → %s", best_kge, best_path)

    if save_folder:
        log_f.close()
    return model


