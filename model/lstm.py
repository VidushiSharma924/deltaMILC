"""LSTM encoder and dPL wrapper for MILC parameter generation.

Classes
-------
CudnnLstm          — single-layer cuDNN LSTM with weight dropout
CudnnLstmModel     — Linear-in → CudnnLstm → Linear-out
MultiInv_MILCModel — dPL top-level: LSTM generates MILC parameters
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from utils.radiation import compute_daylength

from .milc import MILCMul

def debug_tensor(name, x):
    if not isinstance(x, torch.Tensor):
        print(f"{name}: not a tensor")
        return

    print(
        f"{name:15s} "
        f"shape={tuple(x.shape)} "
        f"min={x.min().item():10.4f} "
        f"max={x.max().item():10.4f} "
        f"mean={x.mean().item():10.4f} "
        f"std={x.std().item():10.4f} "
        f"nan={torch.isnan(x).sum().item()} "
        f"inf={torch.isinf(x).sum().item()}"
    )
# ---------------------------------------------------------------------------
# Dropout helpers
# ---------------------------------------------------------------------------

def _create_mask(tensor: torch.Tensor, dr: float) -> torch.Tensor:
    return tensor.new_empty(tensor.shape).bernoulli_(1.0 - dr).div_(1.0 - dr).detach_()


class _DropMask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, train):
        ctx.save_for_backward(mask)
        ctx.train = train
        return x.mul(mask) if train else x

    @staticmethod
    def backward(ctx, grad):
        (mask,) = ctx.saved_tensors
        return (grad * mask if ctx.train else grad), None, None


# ---------------------------------------------------------------------------
# CudnnLstm
# ---------------------------------------------------------------------------

class CudnnLstm(nn.Module):
    """Single-layer cuDNN LSTM with weight-level variational dropout."""

    def __init__(self, *, input_size: int, hidden_size: int, dropout: float = 0.5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.w_ih = Parameter(torch.empty(hidden_size * 4, input_size))
        self.w_hh = Parameter(torch.empty(hidden_size * 4, hidden_size))
        self.b_ih = Parameter(torch.empty(hidden_size * 4))
        self.b_hh = Parameter(torch.empty(hidden_size * 4))
        self._all_weights = [["w_ih", "w_hh", "b_ih", "b_hh"]]

        self._mask_ih: torch.Tensor
        self._mask_hh: torch.Tensor
        self.reset_parameters()
        self.reset_mask()

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            p.data.uniform_(-std, std)

    def reset_mask(self) -> None:
        self._mask_ih = _create_mask(self.w_ih, self.dropout)
        self._mask_hh = _create_mask(self.w_hh, self.dropout)

    # Keep cuDNN state dict compatible
    def __setstate__(self, d):
        super().__setstate__(d)
        self.__dict__.setdefault("_data_ptrs", [])
        if "all_weights" in d:
            self._all_weights = d["all_weights"]
        if isinstance(self._all_weights[0][0], str):
            return
        self._all_weights = [["w_ih", "w_hh", "b_ih", "b_hh"]]

    @property
    def all_weights(self):
        return [[getattr(self, w) for w in group] for group in self._all_weights]

    def forward(self, x: torch.Tensor, hx=None, cx=None) -> tuple:
        do_drop = self.dropout > 0.0 and self.training
        batch = x.size(1)

        if hx is None:
            hx = x.new_zeros(1, batch, self.hidden_size)
        if cx is None:
            cx = x.new_zeros(1, batch, self.hidden_size)

        if x.is_cuda:
            # --- GPU path: use cuDNN kernel with weight dropout ---
            if do_drop:
                self.reset_mask()
                weights = [
                    _DropMask.apply(self.w_ih, self._mask_ih, True),
                    _DropMask.apply(self.w_hh, self._mask_hh, True),
                    self.b_ih,
                    self.b_hh,
                ]
            else:
                weights = [self.w_ih, self.w_hh, self.b_ih, self.b_hh]

            if torch.__version__ >= "1.8":
                out, hy, cy, _, _ = torch._cudnn_rnn(
                    x, weights, 4, None, hx, cx, 2,
                    self.hidden_size, 0, 1, False, 0, self.training, False, (), None,
                )
            else:
                out, hy, cy, _, _ = torch._cudnn_rnn(
                    x, weights, 4, None, hx, cx, 2,
                    self.hidden_size, 1, False, 0, self.training, False, (), None,
                )
            return out, (hy, cy)
        else:
            # --- CPU fallback: use torch._VF.lstm (no weight dropout on CPU) ---
            out, hy, cy = torch._VF.lstm(   
                x, (hx, cx),
                [self.w_ih, self.w_hh, self.b_ih, self.b_hh],
                True,    
                1,      
                0.0,     
                self.training,
                False,   
                False,  
            )
            return out, (hy, cy)


# ---------------------------------------------------------------------------
# CudnnLstmModel
# ---------------------------------------------------------------------------

class CudnnLstmModel(nn.Module):
    """Linear-in → CudnnLstm → Linear-out parameter generator."""

    def __init__(self, *, nx: int, ny: int, hidden_size: int, dropout: float = 0.5):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.hidden_size = hidden_size
        self.linear_in = nn.Linear(nx, hidden_size)
        self.lstm = CudnnLstm(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout)
        self.linear_out = nn.Linear(hidden_size, ny)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [T, B, nx]  →  out: [T, B, ny]"""
        h = F.relu(self.linear_in(x))
        h, _ = self.lstm(h)
        return self.linear_out(h)


# ---------------------------------------------------------------------------
# MultiInv_MILCModel  (dPL top-level)
# ---------------------------------------------------------------------------

def get_var(x, forcing_map, name, permute=False):
    """
    Safely extract forcing variable by name.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor [T, B, F]

    forcing_map : dict
        Maps variable names -> column indices

    name : str
        Variable name

    permute : bool
        Whether to convert [T, B] -> [B, T]

    Returns
    -------
    torch.Tensor
    """

    if name not in forcing_map:
        raise KeyError(
            f"Variable '{name}' not found.\n"
            f"Available variables:\n{list(forcing_map.keys())}"
        )

    out = x[..., forcing_map[name]]

    if permute:
        out = out.permute(1, 0)

    return out

class MultiInv_MILCModel(nn.Module):
    """Differentiable parameter learning model: LSTM → MILC physics.

    The LSTM reads normalized forcings + static attributes, produces physical
    parameters at its last time step, which are fed into MILCMul.

    Parameters
    ----------
    n_inv      : int  — LSTM input size = n_forcing_norm + n_attr
    n_features : int  — number of MILC physical parameters per component
    n_mul      : int  — number of parallel HRU components (Nmul)
    hidden_size: int  — LSTM hidden units
    lstm_dr    : float — LSTM weight dropout
    warmup     : int  — physics spin-up timesteps (bufftime)
    comp_weights : bool — use learned weighted average instead of simple mean
    precip_corr  : bool — learn precipitation correction factor
    """

    def __init__(
        self,
        *,
        n_inv: int,
        n_features: int,
        n_mul: int,
        hidden_size: int,
        lstm_dr: float = 0.2,
        warmup: int = 0,
        comp_weights: bool = False,
        precip_corr: bool = False,
        dynamic_params: bool = False,
        static_pm_params: bool = False,
        forcing_map=None,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_mul = n_mul
        self.warmup = warmup
        self.forcing_map = forcing_map
        


        self.dynamic_params = bool(dynamic_params)
        self.static_pm_params = bool(static_pm_params)
        n_physics = n_features * n_mul
        n_weights = n_mul if comp_weights else 0
        n_pcorr   = 1 if precip_corr else 0
        n_total   = n_physics +  n_weights + n_pcorr  
        self.pm_head = PMParamHead(n_total)    

        self.lstm = CudnnLstmModel(
            nx=n_inv, ny=n_total, hidden_size=hidden_size, dropout=lstm_dr
        )
        self.milc = MILCMul()

        self._n_physics = n_physics
        self._n_weights = n_weights
        self._pcorr_range = [0.5, 1.5] if precip_corr else None


    def forward(self, x, z, month, area, lat, elev, doy=None):
        from utils.penman_monteith import penman_monteith_pet
        from utils.radiation import compute_daylength

        # ── 1. LSTM ───────────────────────────────────────────────
        gen         = self.lstm(z)
        params_mean = gen.mean(dim=0)
        

        # ── 2. Extract forcings ───────────────────────────────────
        fm = self.forcing_map
        if fm is None:
            raise ValueError("forcing_map is None.")

        prcp    = get_var(x, fm, "prcp(mm/day)")                   
        tmax    = get_var(x, fm, "tmax(C)",       permute=True)    
        tmin    = get_var(x, fm, "tmin(C)",       permute=True)    
        srad    = get_var(x, fm, "srad_sw(W/m2)", permute=True)    
        wind    = get_var(x, fm, "wind(m/s)",     permute=True)    
        rel_hum = get_var(x, fm, "rel_hum(%)",    permute=True)    

        # ── 4. Daylength [B, T_total] ─────────────────────────────
        T_total      = tmax.shape[1]              
        warmup_steps = T_total - month.shape[1]   

        # ── 3. PM params ──────────────────────────────────────────
        if self.static_pm_params:
            pm_input = params_mean.unsqueeze(0).expand(T_total, -1, -1)
        else:
            pm_input = gen
        rs, albedo, hc = self.pm_head(pm_input)

        if warmup_steps > 0:
            # Pad the front by repeating the first month value
            month_warmup = month[:, 0:1].expand(-1, warmup_steps)  
            month_full   = torch.cat([month_warmup, month], dim=1) 
        else:
            month_full = month                                     

        if doy is None:
            raise ValueError("doy is required")

        if warmup_steps > 0:
            doy_warmup = doy[:, :1].expand(-1, warmup_steps)
            doy_full = torch.cat([doy_warmup, doy], dim=1)
        else:
            doy_full = doy

        doy_approx = doy_full.float()


        dayl = compute_daylength(lat, doy_approx)
      

        assert dayl.shape == tmax.shape, \
            f"Shape mismatch: dayl={dayl.shape}, tmax={tmax.shape}"

        # ── 5. PET ────────────────────────────────────────────────
        elev_full = elev.unsqueeze(-1).expand(-1, T_total)  
        lat_full  = lat.unsqueeze(-1).expand(-1, T_total)   

        pet = penman_monteith_pet(
            tmax=tmax,
            tmin=tmin,
            wind10=wind,
            srad_sw=srad,
            rel_hum=rel_hum,
            dayl=dayl,
            elev=elev_full,
            rs=rs.permute(1, 0),
            albedo=albedo.permute(1, 0),
            hc=hc.permute(1, 0),     
            lat=lat_full,
            doy=doy_approx,
        )

        pet = torch.clamp(pet, min=0.0, max=20.0)

        # ── 6. x_phys [T_total, B, 3] ────────────────────────────
        tmean  = (tmax + tmin) / 2.0
        x_phys = torch.stack(
            [prcp, tmean.permute(1, 0), pet.permute(1, 0)],
            dim=-1,
        )

        if self.dynamic_params:
            T_phys   = x.shape[0] - self.warmup
            gen_phys = gen[self.warmup:]

            assert gen.shape[0] == x.shape[0], (
                f"gen.shape[0]={gen.shape[0]} != x.shape[0]={x.shape[0]} — "
                "LSTM output length must match x's full length (including warmup)."
            )
            assert gen_phys.shape[0] == T_phys, (
                f"gen_phys.shape[0]={gen_phys.shape[0]} != T_phys={T_phys}"
            )
            assert month.shape[1] == T_phys, (
                f"month.shape[1]={month.shape[1]} != T_phys={T_phys} — "
                "month must already be sliced to the post-warmup active period."
            )
            assert x_phys.shape[:2] == (x.shape[0], x.shape[1]), (
                f"x_phys.shape[:2]={tuple(x_phys.shape[:2])} != "
                f"x.shape[:2]={tuple(x.shape[:2])}"
            )

            p0_dyn   = torch.sigmoid(gen_phys[:, :, :self._n_physics])
            phys     = p0_dyn.view(T_phys, -1, self.n_features, self.n_mul)
        else:
            p0   = params_mean[:, :self._n_physics]
            phys = torch.sigmoid(p0).view(-1, self.n_features, self.n_mul)


        if self._n_weights > 0:
            w0  = params_mean[:, self._n_physics: 
                                self._n_physics +  self._n_weights] 
            wts = F.softmax(w0, dim=-1)
        else:
            wts = None

        if self._pcorr_range is not None:
            cp0       = params_mean[:, -1:]
            pcorr_wts = torch.sigmoid(cp0)
        else:
            pcorr_wts = None

        # ── 8. MILC routing ───────────────────────────────────────
        T_phys     = x.shape[0] - self.warmup
        month_phys = month[:, :T_phys].permute(1, 0).long()

        out = self.milc(
            x_phys,
            parameters=phys,
            month=month_phys,
            area=area,
            mu=self.n_mul,
            muwts=wts,
            bufftime=self.warmup,
            corrwts=pcorr_wts,
            pcorr=self._pcorr_range,
        )

        return out

class PMParamHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 3)   

    def forward(self, h):
        out    = self.fc(h)
        rs     = torch.sigmoid(out[..., 0]) * 200 + 50    
        albedo = torch.sigmoid(out[..., 1]) * 0.3  + 0.1  
        hc     = torch.sigmoid(out[..., 2]) * 2.45 + 0.05
        return rs, albedo,hc
