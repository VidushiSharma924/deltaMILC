"""MILC differentiable hydrological physics model.

Implements the soil-water balance equations and geomorphological IUH routing
for multi-component (Nmul) parallel simulations.

References
----------
Original MILC physics implemented by Yalan Song / MHPI group, PSU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# IUH lookup table (dimensionless, from GIUH theory)
# ---------------------------------------------------------------------------

_IUH_DATA = torch.tensor([
    [0.000, 0.00000], [0.027, 0.04021], [0.054, 0.08452], [0.081, 0.13264],
    [0.108, 0.18390], [0.135, 0.23781], [0.162, 0.29343], [0.189, 0.35029],
    [0.216, 0.40748], [0.243, 0.46434], [0.270, 0.52031], [0.297, 0.57474],
    [0.324, 0.62708], [0.351, 0.67691], [0.378, 0.72380], [0.405, 0.76742],
    [0.432, 0.80759], [0.459, 0.84404], [0.486, 0.87666], [0.514, 0.90537],
    [0.541, 0.93018], [0.568, 0.95106], [0.595, 0.96810], [0.622, 0.98138],
    [0.649, 0.99102], [0.676, 0.99717], [0.703, 1.00000], [0.730, 0.99969],
    [0.757, 0.99643], [0.784, 0.99043], [0.811, 0.98188], [0.838, 0.97100],
    [0.865, 0.95800], [0.892, 0.94308], [0.919, 0.92644], [0.946, 0.90827],
    [0.973, 0.88878], [1.000, 0.86814], [1.027, 0.84651], [1.054, 0.82407],
    [1.081, 0.80097], [1.108, 0.77735], [1.135, 0.75334], [1.162, 0.72908],
    [1.189, 0.70467], [1.216, 0.68022], [1.243, 0.65583], [1.270, 0.63158],
    [1.297, 0.60755], [1.324, 0.58380], [1.351, 0.56041], [1.378, 0.53742],
    [1.405, 0.51488], [1.432, 0.49283], [1.459, 0.47131], [1.486, 0.45034],
    [1.514, 0.42995], [1.541, 0.41015], [1.568, 0.39097], [1.595, 0.37240],
    [1.622, 0.35446], [1.649, 0.33715], [1.676, 0.32047], [1.703, 0.30442],
    [1.730, 0.28899], [1.757, 0.27417], [1.784, 0.25996], [1.811, 0.24634],
    [1.838, 0.23331], [1.865, 0.22084], [1.892, 0.20893], [1.919, 0.19757],
    [1.946, 0.18673], [1.973, 0.17640], [2.000, 0.16656], [2.027, 0.15720],
    [2.054, 0.14831], [2.081, 0.13985], [2.108, 0.13182], [2.135, 0.12421],
    [2.162, 0.11698], [2.189, 0.11014], [2.216, 0.10366], [2.243, 0.09752],
    [2.270, 0.09171], [2.297, 0.08622], [2.324, 0.08104], [2.351, 0.07613],
    [2.378, 0.07151], [2.405, 0.06714], [2.432, 0.06302], [2.459, 0.05914],
    [2.486, 0.05548], [2.514, 0.05203], [2.541, 0.04878], [2.568, 0.04572],
    [2.595, 0.04285], [2.622, 0.04014], [2.649, 0.03760], [2.676, 0.03521],
    [2.703, 0.03296], [2.730, 0.03085], [2.757, 0.02887], [2.784, 0.02700],
    [2.811, 0.02526], [2.838, 0.02362], [2.865, 0.02208], [2.892, 0.02064],
    [2.919, 0.01929], [2.946, 0.01803], [2.973, 0.01684], [3.000, 0.01573],
    [3.027, 0.01469], [3.054, 0.01372], [3.081, 0.01281], [3.108, 0.01195],
    [3.135, 0.01116], [3.162, 0.01041], [3.189, 0.00971],
], dtype=torch.float32)

# Monthly Pruitt ET coefficients (Jan..Dec)
_PRUITT_COEF = torch.tensor([
    0.21, 0.22, 0.23, 0.28, 0.30, 0.31,
    0.30, 0.29, 0.27, 0.25, 0.22, 0.20,
], dtype=torch.float32)

# MILC parameter scaling ranges: [min, max]
_PARAM_SCALES = [
    [0.02550, 25.0],    # gamma 
    [0.00254, 30.0],    # m     
    [2.84000, 5000.0],  # ks    
    [0.01000, 1.0],     # nu    
    [0.10500, 15.0],    # alpha 
    [0.10000, 1.0],     # w0    
    [1.84000, 10000.0], # wmax 
    [0.10100, 2.0],     # kc    
]


# ---------------------------------------------------------------------------
# Utility: differentiable 1-D linear interpolation
# ---------------------------------------------------------------------------

def _torch_interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Differentiable linear interpolation (equivalent to np.interp)."""
    x_clamped = x.clamp(xp[0], xp[-1])
    idx = torch.searchsorted(xp.contiguous(), x_clamped.contiguous())
    idx = idx.clamp(1, len(xp) - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = fp[idx - 1], fp[idx]
    slope = (y1 - y0) / (x1 - x0 + 1e-12)
    return y0 + slope * (x_clamped - x0)


# ---------------------------------------------------------------------------
# Batched IUH kernel builder
# ---------------------------------------------------------------------------

def _iuh_comp_batch(
    gamma_flat: torch.Tensor,
    area_flat: torch.Tensor,
    dt: float,
    delta_T: float,
    iuh_data: torch.Tensor,
) -> torch.Tensor:
    """Build geomorphological IUH kernels for a batch of (gamma, area) pairs.

    Returns
    -------
    kernels : [B_flat, L]  — normalized IUH kernel per (basin × component)
    """
    device = gamma_flat.device
    iuh_data = iuh_data.to(device)

    lag = (gamma_flat * 1.19 * area_flat ** 0.33) / delta_T  
    hp  = 0.8 / lag                                             

    t_base   = iuh_data[:, 0]
    iuh_base = iuh_data[:, 1]

    tmax = t_base[-1].item()
    L    = min(max(1, int(tmax * lag.max().item() / dt) + 1), 2000)

    t_steps = torch.arange(0, L, device=device, dtype=gamma_flat.dtype) * dt 
    t_norm  = t_steps.unsqueeze(0) / lag.unsqueeze(1)                          

    idx = torch.searchsorted(t_base.contiguous(), t_norm.reshape(-1).contiguous())
    idx = idx.clamp(1, len(t_base) - 1)

    t0 = t_base[idx - 1]
    t1 = t_base[idx]
    y0 = iuh_base[idx - 1]
    y1 = iuh_base[idx]

    frac = ((t_norm.reshape(-1) - t0) / (t1 - t0 + 1e-8)).clamp(0.0, 1.0)
    iuh_interp = (y0 + frac * (y1 - y0)).reshape(gamma_flat.shape[0], L)

    kernels = iuh_interp * hp.unsqueeze(1)
    kernels = kernels * (t_norm < tmax).float()
    return kernels

def _iuh_nash_batch(
    gamma_flat: torch.Tensor,
    area_flat: torch.Tensor,
    dt: float,
    delta_T: float,
) -> torch.Tensor:
    """Build Nash (gamma=1, K=0.5*lag) IUH kernels for baseflow routing.

    Returns
    -------
    kernels : [B_flat, L]
    """
    device = gamma_flat.device
    K = torch.clamp(
        (0.5 * gamma_flat * 1.19 * area_flat ** 0.33) / delta_T,
        min=1e-3,
    )  

    L = min(
        max(1, int((10.0 * K.max().detach().item()) / dt) + 1),
        2000,
    )
    t = torch.arange(L, device=device, dtype=K.dtype) * dt  


    kernels = torch.exp(-t[None, :] / K[:, None]) / K[:, None]  
    return kernels

def _convolve_batched(signals: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
    """Convolve each of B signals with its own kernel using grouped conv1d.

    Parameters
    ----------
    signals : [B, T]
    kernels : [B, L]

    Returns
    -------
    [B, T]
    """
    B, T = signals.shape
    L    = kernels.shape[1]
    pad  = L - 1
    out  = F.conv1d(
        signals.unsqueeze(0),           
        kernels.flip(1).unsqueeze(1),   
        padding=pad,
        groups=B,
    )
    return out.squeeze(0)[:, :T]        


# ---------------------------------------------------------------------------
# MILCMul  — the core physics model
# ---------------------------------------------------------------------------

class MILCMul(nn.Module):
    """Multi-component MILC physics model (differentiable, GPU-native).

    Physics summary per timestep
    ----------------------------
    1. Infiltration:   infil = P × (1 - sat^alpha),  sat = W/wmax
    2. Runoff:         runoff = P - infil  (excess)
    3. ET:             et = PET × sat
    4. Baseflow:       bf = (1 - nu) × ks × sat^m
    5. Percolation:    perc = nu × ks × sat^m
    6. Routing via geomorphological IUH convolution (GPU-batched)

    Parameter ordering (8 params, indices 0–7)
    ------------------------------------------
    0: gamma  [0.0255 – 25]    IUH shape / lag factor
    1: m      [0.00254 – 30]   recession exponent
    2: ks     [2.84 – 5000]    saturated hydraulic conductivity (mm/day)
    3: nu     [0.01 – 1]       baseflow fraction of total drainage
    4: alpha  [0.105 – 15]     infiltration nonlinearity exponent
    5: w0     [0.1 – 1]        initial relative soil saturation
    6: wmax   [1.84 – 10000]   maximum soil water content (mm)
    7: kc     [0.101 – 2]      Pruitt crop coefficient (used if PET not in x)
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("iuh_data", _IUH_DATA.clone())
        self.register_buffer("pruitt_coef", _PRUITT_COEF.clone())

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        parameters: torch.Tensor,
        month: torch.Tensor,
        area: torch.Tensor,
        mu: int,
        muwts=None,
        dt: float = 1.0,
        bufftime: int = 0,
        corrwts=None,
        pcorr=None,
        outstate: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : [T_total, B, ≥2]   raw forcings (prcp, tmean [, pet])
        parameters : [B, n_features, mu] or [T_phys, B, n_features, mu]
                     scaled [0,1] parameters from LSTM (static or dynamic)
        month      : [T_phys, B]          integer months (1–12) for active period
        # area       : [B]                  basin area (km²)
        area       : [B] or [B, 1]        basin area (km²)
        mu         : int                  number of components
        muwts      : [B, mu] or None      component weights (None = simple mean)
        # rtwts      : [B, 2] or None       routing parameters (unused, reserved)
        bufftime   : int                  warmup timesteps at front of x
        corrwts    : [B, 1] or None       learned precip correction weights
        pcorr      : [lo, hi] or None     precip correction range
        outstate   : bool                 if True, return final states instead of fluxes

        Returns
        -------
        Qall : [T_phys, B, 4]  — (Qs, qd_mean, qb_mean, et_mean)  [mm/day]
        """
        device = x.device
        is_dynamic = parameters.dim() == 4   

        p_ref = parameters.mean(0) if is_dynamic else parameters   

        def _sc(p, i):
            lo, hi = _PARAM_SCALES[i]
            return lo + p * (hi - lo)

        B = x.shape[1]
        if bufftime > 0:
            warmup_out = self.forward(
                x[:bufftime],
                p_ref, month[:bufftime], area,
                mu=mu, muwts=muwts,
                bufftime=0, corrwts=corrwts, pcorr=pcorr, outstate=True,
            )
            W, _ = warmup_out
            W = W.detach()
        else:
            w0_r    = _sc(p_ref[:, 5, :], 5)
            wmax_ref = _sc(p_ref[:, 6, :], 6)
            W = w0_r * wmax_ref  
        gamma = torch.clamp(_sc(p_ref[:, 0, :], 0), min=0.1, max=5.0)
        if torch.isnan(gamma).any():
            gamma = torch.nan_to_num(gamma, nan=0.1)

        if not is_dynamic:
            m_s     = _sc(parameters[:, 1, :], 1)
            ks_s    = _sc(parameters[:, 2, :], 2)
            nu_s    = _sc(parameters[:, 3, :], 3)
            alpha_s = _sc(parameters[:, 4, :], 4)
            wmax_s  = _sc(parameters[:, 6, :], 6)
            kc_s    = _sc(parameters[:, 7, :], 7)

        P_raw = x[bufftime:, :, 0]           
        T_air = x[bufftime:, :, 1]           
        Nstep, Ngrid = P_raw.shape

        if pcorr is not None:
            corr_factor = pcorr[0] + corrwts[:, 0] * (pcorr[1] - pcorr[0])
            P_raw = corr_factor.unsqueeze(0) * P_raw

        Pm   = P_raw.unsqueeze(2).expand(-1, -1, mu)   

        if x.shape[-1] >= 3:
            ETpm = x[bufftime:, :, 2].unsqueeze(2).expand(-1, -1, mu)
        else:
            kc_ref = _sc(p_ref[:, 7, :], 7)
            coef = self.pruitt_coef[month - 1].float()
            pet  = coef * torch.clamp(T_air, min=0.0) * kc_ref.mean(dim=1, keepdim=True).unsqueeze(0)
            ETpm = pet.unsqueeze(2).expand(-1, -1, mu)

        runoff_ts   = torch.zeros(Nstep, Ngrid, mu, device=device)
        baseflow_ts = torch.zeros_like(runoff_ts)
        et_ts       = torch.zeros_like(runoff_ts)

   
        for t in range(Nstep):
            if is_dynamic:
                p_t     = parameters[t]                  
                m_t     = _sc(p_t[:, 1, :], 1)
                ks_t    = _sc(p_t[:, 2, :], 2)
                nu_t    = _sc(p_t[:, 3, :], 3)
                alpha_t = _sc(p_t[:, 4, :], 4)
                wmax_t  = _sc(p_t[:, 6, :], 6)
            else:
                m_t, ks_t, nu_t, alpha_t, wmax_t = m_s, ks_s, nu_s, alpha_s, wmax_s

            P_t   = Pm[t]
            PET_t = ETpm[t]

            sat_pre = torch.clamp(W / wmax_t, min=1e-7, max=1.0)
            infil   = P_t * (1.0 - torch.pow(sat_pre, alpha_t))
            ovfl    = torch.clamp(W + infil - wmax_t, min=0.0)
            infil   = infil - ovfl
            runoff  = torch.clamp(P_t - infil, min=0.0)
            W = W + infil

            sat_post_infil = torch.clamp(W / wmax_t, min=1e-7, max=1.0)
            et_t = torch.clamp(PET_t * sat_post_infil, min=0.0)
            W = W - et_t

            sat_post_et = torch.clamp(W / wmax_t, min=1e-7, max=1.0)
            drainage = ks_t * sat_post_et ** m_t
            bf_t   = (1.0 - nu_t) * drainage
            perc_t = nu_t * drainage
            W = W - bf_t - perc_t
            W = torch.clamp(W, min=1e-5)

            runoff_ts[t]   = runoff
            baseflow_ts[t] = bf_t
            et_ts[t]       = et_t

        if outstate:
            return W, (ks_s if not is_dynamic else _sc(parameters[-1, :, 2, :], 2))


        B_flat       = Ngrid * mu
        gamma_flat   = gamma.reshape(-1)                              
        area_1d = area.reshape(-1)
        if area_1d.numel() != Ngrid:
            raise ValueError(
                f"Expected {Ngrid} basin areas, got shape {tuple(area.shape)}"
            )
        area_flat = (
            area_1d[:, None]
            .expand(Ngrid, mu)
            .reshape(-1)
        )

        delta_T = 24.0  

        IUH1 = _iuh_comp_batch(gamma_flat, area_flat, dt, delta_T, self.iuh_data) * dt
        IUH1 = IUH1 / IUH1.sum(dim=1, keepdim=True).clamp(min=1e-6)

        IUH2 = _iuh_nash_batch(
            gamma_flat,
            area_flat,
            dt,
            delta_T,
        ) * dt
        IUH2 = IUH2 / IUH2.sum(dim=1, keepdim=True).clamp(min=1e-6)

        qs_flat = runoff_ts.permute(1, 2, 0).reshape(B_flat, -1)
        bf_flat = baseflow_ts.permute(1, 2, 0).reshape(B_flat, -1)

        qd_flat = _convolve_batched(qs_flat, IUH1)
        qb_flat = _convolve_batched(bf_flat, IUH2)

        qd = qd_flat.view(Ngrid, mu, -1).permute(2, 0, 1)
        qb = qb_flat.view(Ngrid, mu, -1).permute(2, 0, 1)
        qt = qd + qb

        if muwts is None:
            Qs = qt.mean(dim=-1)            # [T, B]
        else:
            Qs = (qt * muwts.unsqueeze(0)).sum(dim=-1)

        Qall = torch.stack(
            [Qs, qd.mean(-1), qb.mean(-1), et_ts.mean(-1)],
            dim=-1,
        )
        return Qall

