"""Loss functions for dMILC training."""

from typing import Union

import numpy as np
import torch
import torch.nn as nn


class NSELossBatch(nn.Module):
    """Normalized squared-error loss (per-basin normalization).

    Equivalent to 1 - NSE when averaged over all valid timesteps.
    Numerically stable via per-basin std pre-computed on training data.

    Parameters
    ----------
    std_array : array-like [n_basins]  — per-basin streamflow std (mm/day)
    eps       : float                   — floor on denominator
    """

    def __init__(self, std_array: Union[np.ndarray, torch.Tensor], eps: float = 0.1):
        super().__init__()
        std = torch.as_tensor(std_array, dtype=torch.float32).view(-1)
        self.register_buffer("std", std)
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        basin_idx: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        if pred.dim() == 3:
            pred   = pred[..., 0]
            target = target[..., 0]

        T, B = pred.shape
        std_b  = self.std[basin_idx]
        std_tb = std_b.unsqueeze(0).expand(T, B)

        mask = torch.isfinite(pred) & torch.isfinite(target)
        if not mask.any():
            return torch.nan_to_num(pred).sum() * 0.0

        pred_safe   = torch.where(mask, pred, torch.zeros_like(pred))
        target_safe = torch.where(mask, target, torch.zeros_like(target))

        loss = ((pred_safe - target_safe) ** 2 / (std_tb + self.eps) ** 2)[mask].mean()
        return loss




class RmseLossComb(nn.Module):
    """Combined linear-RMSE + log-sqrt-RMSE loss.

    loss = (1 - alpha) * RMSE + alpha * log_sqrt_RMSE

    Parameters
    ----------
    alpha : float — weight on log-sqrt term (0 = pure RMSE, 1 = pure log-sqrt)
    beta  : float — stability offset inside log-sqrt
    """

    def __init__(self, alpha: float = 0.25, beta: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 3:
            pred   = pred[..., 0]
            target = target[..., 0]

        mask = torch.isfinite(pred) & torch.isfinite(target)
        if not mask.any():
            return torch.nan_to_num(pred).sum() * 0.0
        p  = pred[mask]
        t  = target[mask]
        loss1 = torch.sqrt(((p - t) ** 2).mean())

        p_log = torch.log10(torch.sqrt(pred.clamp(min=0) + self.beta) + 0.1)
        t_log = torch.log10(torch.sqrt(target.clamp(min=0) + self.beta) + 0.1)
        mask2 = torch.isfinite(p_log) & torch.isfinite(t_log)
        if not mask2.any():
            return loss1
        loss2 = torch.sqrt(((p_log[mask2] - t_log[mask2]) ** 2).mean())

        return (1.0 - self.alpha) * loss1 + self.alpha * loss2


class KGELoss(nn.Module):
    """Vectorised per-basin KGE loss: mean(1 − KGE) over batch.

    KGE  = 1 − √[(r−1)² + (α−1)² + (β−1)²]
    Loss = 1 − KGE  (minimising drives KGE → 1) = √[(r−1)² + (α−1)² + (β−1)²

    Fully vectorised over the batch dimension — no Python loops.

    Parameters
    ----------
    eps : float — floor on std and mean denominators for numerical stability
    """

    def __init__(self, eps: float = 0.1):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        basin_idx=None,          
    ) -> torch.Tensor:
        if pred.dim() == 3:
            pred   = pred[..., 0]
            target = target[..., 0]

        mask = torch.isfinite(pred) & torch.isfinite(target)     
        n_obs_raw = mask.sum(dim=0)                               
        valid = n_obs_raw >= 10                                   

        n_obs = n_obs_raw.clamp_min(1).float()                   

        p_safe = torch.where(mask, pred, torch.zeros_like(pred))
        t_safe = torch.where(mask, target, torch.zeros_like(target))

        mean_p = p_safe.sum(dim=0) / n_obs
        mean_t = t_safe.sum(dim=0) / n_obs

        dp = torch.where(mask, pred - mean_p[None, :], torch.zeros_like(pred))
        dt = torch.where(mask, target - mean_t[None, :], torch.zeros_like(target))

        std_p = torch.sqrt((dp.square().sum(dim=0) / n_obs).clamp_min(1e-8))
        std_t = torch.sqrt((dt.square().sum(dim=0) / n_obs).clamp_min(self.eps ** 2))

        cov = (dp * dt).sum(dim=0) / n_obs
        r = (cov / (std_p * std_t)).clamp(-1.0, 1.0)

        alpha = std_p / std_t
        beta = mean_p / mean_t.clamp_min(self.eps)

        kge_loss = torch.sqrt(
            (r - 1).square()
            + (alpha - 1).square()
            + (beta - 1).square()
        )

        return kge_loss[valid].mean() if valid.any() else pred.sum() * 0.0




class CombinedKGENSELoss(nn.Module):
    """Weighted combination of KGE and NSE losses.

    loss = (1-w)*KGE_loss + w*NSE_loss

    NSE provides stable gradients early in training; KGE drives the optimum.
    Recommended: start with nse_weight=0.5, reduce toward 0 if desired.
    """

    def __init__(self, std_array, eps: float = 0.1, nse_weight: float = 0.5):
        super().__init__()
        self.kge = KGELoss(eps=eps)
        self.nse = NSELossBatch(std_array, eps=eps)
        self.w = nse_weight

    def forward(self, pred, target, basin_idx=None):
        if basin_idx is None:
            raise ValueError(
                "basin_idx is required when using CombinedKGENSELoss."
            )
        kge_l = self.kge(pred, target)
        nse_l = self.nse(pred, target, basin_idx)
        return (1.0 - self.w) * kge_l + self.w * nse_l


def build_loss(
    loss_name: str,
    train_y: np.ndarray = None,
    eps: float = 0.1,
    alpha: float = 0.25,
    nse_weight: float = 0.0,
) -> nn.Module:
    """Factory that builds the configured loss function.

    Parameters
    ----------
    loss_name  : "kge_batch" | "nse_kge" | "nse_batch" | "rmse_comb"
    train_y    : [B, T, 1] training streamflow — required for nse_batch / nse_kge
    eps, alpha : hyperparameters
    nse_weight : for "kge_batch", blend in this fraction of NSE loss (0=pure KGE)
    """
    if loss_name == "nse_kge":
        if train_y is None:
            raise ValueError("train_y is required for nse_kge loss.")
        std = np.nanstd(train_y, axis=1)
        return CombinedKGENSELoss(std, eps=eps, nse_weight=nse_weight)
    
    if loss_name == "kge_batch":
        if nse_weight > 0 and train_y is not None:
            std = np.nanstd(train_y, axis=1)
            return CombinedKGENSELoss(std, eps=eps, nse_weight=nse_weight)
        return KGELoss(eps=eps)
        
    if loss_name == "nse_batch":
        if train_y is None:
            raise ValueError("train_y is required for nse_batch loss.")
        std = np.nanstd(train_y, axis=1)  
        return NSELossBatch(std, eps=eps)
    if loss_name == "rmse_comb":
        return RmseLossComb(alpha=alpha)
    raise ValueError(f"Unknown loss '{loss_name}'. Choose 'nse_kge', 'kge_batch', 'nse_batch', or 'rmse_comb'.")

