#!/usr/bin/env python3
"""dMILC — main entry point.

Usage
-----
Train:
    python run.py --config config.yml --mode train

Evaluate:
    python run.py --config config.yml --mode eval

Train + Evaluate:
    python run.py --config config.yml --mode both
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from data.dataset import build_doy_tensor
from model.milc import _PARAM_SCALES



# ---------------------------------------------------------------------------
# Path setup:
# ---------------------------------------------------------------------------
_DMILC_DIR = Path(__file__).resolve().parent
if str(_DMILC_DIR) not in sys.path:
    sys.path.insert(0, str(_DMILC_DIR))


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
    )


logger = logging.getLogger("dmilc.run")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg

def resolve_output_dir(cfg: dict) -> str:
    exp_root = cfg["output"]["root"]
    run_name = cfg["output"].get("run_name") or _auto_run_name(cfg)
    out_dir = os.path.join(exp_root, run_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir



def _auto_run_name(cfg: dict) -> str:
    t = cfg["train"]
    return (
        f"{cfg['model']['name']}"
        f"_Ep{t['epochs']}_BS{t['batch_size']}"
        f"_RHO{t['seq_len']}_HS{t['hidden_size'] if 'hidden_size' in t else cfg['model']['hidden_size']}"
        f"_Buff{t['warmup']}_seed{t['seed']}"
    )


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def build_data(cfg: dict, mode: str = "both"):

    import json, os
    import numpy as np

    from data.dataset import (
        CAMELSDataset,
        load_or_build_cache,
        basin_norm,
        build_area_month_tensors,
    )
    from utils.scaler import HydroScaler

    dcfg = cfg["data"]

    forcing_vars = [
        "prcp(mm/day)",
        "tmax(C)",
        "tmin(C)",
        "tavg(C)",
        "srad_sw(W/m2)",
        "srad_lw(W/m2)",
        "wind(m/s)",
        "rel_hum(%)",
    ]
    forcing_map = {v: i for i, v in enumerate(forcing_vars)}

    root_db      = dcfg["root_db"]
    forcing_root = dcfg["forcing_root"]
    attr_vars    = list(dcfg["attr_vars"])

    with open(dcfg["subset_file"]) as f:
        basin_ids = json.load(f)
    basin_ids = sorted(basin_ids)

    dataset = CAMELSDataset(
        root_db=root_db,
        forcing_root=forcing_root,
        flow_path=dcfg["flow_path"],
        attr_path=dcfg["attr_path"],
        pet_subdir=dcfg.get("pet_subdir", "pet_harg"),
        pet_range=dcfg.get("pet_range", None),
    )


    val_cache = os.path.join(root_db, dcfg["val_cache"]) if dcfg.get("val_cache") else None
    
    if mode == "eval":
        val_period = dcfg.get("test_period") or dcfg.get("val_period")
        val_cache  = os.path.join(root_db, dcfg.get("test_cache") or dcfg.get("val_cache"))

        ckpt_path = cfg.get("eval", {}).get("checkpoint")
        if not ckpt_path:
            out_dir = resolve_output_dir(cfg)
            ckpt_path = os.path.join(out_dir, "model_best.pt")
            
        train_out_dir = os.path.dirname(ckpt_path)
        stat_path = os.path.join(train_out_dir, "statDict.json")



        logger.info("Loading validation data …")
        x_va, y_va, c_va, mask_va, valid_ids = load_or_build_cache(
            dataset, val_cache, basin_ids, val_period,
            attr_vars, forcing_vars, 
            rebuild=dcfg.get("rebuild_cache", False),
        )
        

        ckpt_path = cfg["eval"]["checkpoint"]
        train_out_dir = os.path.dirname(ckpt_path)
    
        stat_path = os.path.join(train_out_dir, "statDict.json")
        if not os.path.exists(stat_path):
            raise FileNotFoundError(
                f"statDict.json not found at {stat_path}. "
                "Check eval.checkpoint points to the correct training run."
            )
       
        ystd_path = os.path.join(train_out_dir, "y_std.npy")
        if not os.path.exists(ystd_path):
            raise FileNotFoundError(
                f"y_std.npy not found at {ystd_path}. "
                "Check eval.checkpoint points to the correct training run."
            )


        model_name = cfg["model"]["name"]
        log_norm   = ["prcp"] if model_name == "dMILC1.0" else []
        
        import json

        scaler = HydroScaler(
            attr_vars=attr_vars,
            series_vars=forcing_vars,
            log_norm_cols=log_norm,
        )
        
        with open(stat_path, "r") as f:
            scaler.stat_dict = json.load(f)
        
        valid_attrs = scaler.attr_vars

        valid_idx = [
            i for i, name in enumerate(attr_vars)
            if name in valid_attrs
        ]
        c_va = c_va[:, valid_idx]

        y_std = np.load(ystd_path).astype(np.float32)      
        if y_std.ndim == 2:
            y_std = y_std[:, :, np.newaxis]
        if y_std.shape[0] != y_va.shape[0]:
            raise ValueError(
                f"y_std basin count {y_std.shape[0]} does not match "
                f"validation basin count {y_va.shape[0]} — "
                f"eval.checkpoint may not match the current subset_file."
            )
        y_std = np.maximum(y_std, 0.1)

        print("new y_std shape:", y_std.shape)
        print("valid_ids:", len(valid_ids))
        print("y_std shape:", y_std.shape)

        y_va  = y_va / y_std


        z_va_norm = scaler.transform(x_va.copy(), forcing_vars)
        c_va_norm = scaler.transform(c_va, valid_attrs)
        x_va[np.isnan(x_va)]         = 0.0
        z_va_norm[np.isnan(z_va_norm)] = 0.0
        c_va_norm[np.isnan(c_va_norm)] = 0.0


        area_col   = [i for i, n in enumerate(valid_attrs) if n == "cwc_area"]
        if not area_col:
            raise ValueError("'cwc_area' not found in valid_attrs")
        basin_area = c_va[:, area_col[0]:area_col[0]+1]

        logger.info("Building area and month tensors …")
        area_tensor, month_tensor_va, lat_tensor, elev_tensor, doy_tensor_va = build_area_month_tensors(
            valid_ids, forcing_root, dcfg["attr_path"], t_range=val_period,
        )

        return dict(
            x_tr=None, z_tr=None, y_tr=None, c_tr=None,
            x_va=x_va, z_va=z_va_norm, y_va=y_va, c_va=c_va_norm,
            area_tensor=area_tensor,
            month_tensor=month_tensor_va,      
            doy_tensor=doy_tensor_va,
            month_tensor_va=month_tensor_va,    
            doy_tensor_va=doy_tensor_va,
            lat_tensor=lat_tensor,
            elev_tensor=elev_tensor,
            scaler=scaler,
            basin_ids=valid_ids,
            valid_attrs=valid_attrs,
            y_std=y_std,
            basin_area_km2=basin_area[:, 0],
            forcing_map=forcing_map,
        )


    # ── TRAIN (or BOTH) PATH ────────────────────────────────────────────────
    train_cache  = os.path.join(root_db, dcfg["train_cache"])
    train_period = dcfg["train_period"]
    val_cache    = os.path.join(root_db, dcfg["val_cache"])
    val_period   = dcfg["val_period"]

    logger.info("Loading training data …")
    x_tr, y_tr, c_tr, mask_tr, valid_ids = load_or_build_cache(
        dataset, train_cache, basin_ids, train_period,
        attr_vars, forcing_vars,
        rebuild=dcfg.get("rebuild_cache", False),
    )

    logger.info("Loading validation data …")
    x_va, y_va, c_va, mask_va, valid_ids = load_or_build_cache(
        dataset, val_cache, basin_ids, val_period,
        attr_vars, forcing_vars,
        rebuild=dcfg.get("rebuild_cache", False),
    )

    valid_idx, valid_attrs = [], []
    for i, name in enumerate(attr_vars):
        if np.any(~np.isnan(c_tr[:, i])):
            valid_idx.append(i)
            valid_attrs.append(name)
        else:
            logger.warning("Dropping all-NaN attribute: %s", name)
    c_tr = c_tr[:, valid_idx]
    c_va = c_va[:, valid_idx]

    area_col = np.where(np.array(valid_attrs) == "cwc_area")[0]
    if len(area_col) == 0:
        raise ValueError("'cwc_area' not found in attribute list.")
    basin_area = c_tr[:, area_col[0]:area_col[0]+1]

    print("=== STREAMFLOW SANITY CHECK ===")
    print(f"y_tr (mm/day) — mean: {np.nanmean(y_tr):.3f}, max: {np.nanmax(y_tr):.1f}")


    y_std = np.nanstd(y_tr[:, :, 0], axis=1, keepdims=True)
    y_std = np.maximum(y_std, 0.1)[:, :, np.newaxis]          
    y_tr  = y_tr / y_std
    y_va  = y_va / y_std


    out_dir = resolve_output_dir(cfg)
    np.save(os.path.join(out_dir, "y_std.npy"), y_std)
    logger.info("y_std saved → %s", os.path.join(out_dir, "y_std.npy"))

    model_name = cfg["model"]["name"]
    log_norm   = ["prcp"] if model_name == "dMILC1.0" else []

    scaler = HydroScaler(attr_vars=valid_attrs, series_vars=forcing_vars, log_norm_cols=log_norm)
    c_tr_norm, z_tr_norm = scaler.fit_transform(c_tr, x_tr.copy())
    c_tr_norm[np.isnan(c_tr_norm)] = 0.0

    z_va_norm = scaler.transform(x_va.copy(), forcing_vars)
    c_va_norm = scaler.transform(c_va, valid_attrs)

    x_tr[np.isnan(x_tr)]           = 0.0
    x_va[np.isnan(x_va)]           = 0.0
    z_tr_norm[np.isnan(z_tr_norm)] = 0.0
    z_va_norm[np.isnan(z_va_norm)] = 0.0
    c_va_norm[np.isnan(c_va_norm)] = 0.0


    logger.info("Building area and month tensors …")
    area_tensor, month_tensor_tr, lat_tensor, elev_tensor, doy_tensor_tr = build_area_month_tensors(
        valid_ids, forcing_root, dcfg["attr_path"], t_range=train_period,
    )
    _, month_tensor_va, _, _, doy_tensor_va = build_area_month_tensors(
        valid_ids, forcing_root, dcfg["attr_path"], t_range=val_period,
    )

    return dict(
        x_tr=x_tr, z_tr=z_tr_norm, y_tr=y_tr, c_tr=c_tr_norm,
        x_va=x_va, z_va=z_va_norm, y_va=y_va, c_va=c_va_norm,
        area_tensor=area_tensor,
        month_tensor=month_tensor_tr,       
        doy_tensor=doy_tensor_tr,           
        month_tensor_va=month_tensor_va,    
        doy_tensor_va=doy_tensor_va,       
        lat_tensor=lat_tensor,
        elev_tensor=elev_tensor,
        scaler=scaler,
        basin_ids=valid_ids,
        valid_attrs=valid_attrs,
        y_std=y_std,
        basin_area_km2=basin_area[:, 0],
        forcing_map=forcing_map,
    )




# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(cfg: dict, n_inv: int, forcing_map: dict) -> torch.nn.Module:
    from model.lstm import MultiInv_MILCModel

    mcfg = cfg["model"]
    return MultiInv_MILCModel(
        n_inv=n_inv,
        n_features=mcfg["n_physics_params"],
        n_mul=mcfg["n_components"],
        hidden_size=mcfg["hidden_size"],
        lstm_dr=mcfg["lstm_dropout"],
        warmup=cfg["train"]["warmup"],
        comp_weights=mcfg.get("component_weights", False),
        precip_corr=mcfg.get("precip_correction", False),
        dynamic_params=mcfg.get("dynamic_params", False),
        static_pm_params=mcfg.get("static_pm_params", False),
        forcing_map=forcing_map
    )


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def _find_resume_checkpoint(out_dir: str):
    """Return (model, start_epoch) from the latest checkpoint in out_dir, or (None, 0)."""
    import glob, re
    ckpts = sorted(glob.glob(os.path.join(out_dir, "model_Ep*.pt")))
    latest_ep, latest_path = 0, None
    for ck in ckpts:
        m = re.search(r"model_Ep(\d+)\.pt", ck)
        if m:
            ep = int(m.group(1))
            if ep > latest_ep:
                latest_ep, latest_path = ep, ck
    if latest_path is None:
        return None, 0
    logger.info("Resuming from checkpoint: %s (epoch %d)", latest_path, latest_ep)
    model = torch.load(latest_path, map_location="cpu", weights_only=False)
    return model, latest_ep


def run_train(cfg: dict, data: dict, out_dir: str) -> torch.nn.Module:
    from model.loss import build_loss
    from trainer import train

    tcfg = cfg["train"]
    lcfg = cfg["loss"]

    loss_fn = build_loss(
        lcfg["name"],
        train_y=data["y_tr"],
        eps=lcfg.get("eps", 0.1),
        alpha=lcfg.get("rmse_alpha", 0.25),
        nse_weight=lcfg.get("nse_weight", 0.0),
    )
    print("y_std sample:", data["y_std"][:5, 0, 0])

    import numpy as np
    std_check = np.nanstd(data["y_tr"][:, :, 0], axis=1)

    n_inv = data["x_tr"].shape[-1] + data["c_tr"].shape[-1]


    resume = tcfg.get("resume", False)
    start_epoch = 0
    if resume:
        model, start_epoch = _find_resume_checkpoint(out_dir)
        if model is None:
            model = build_model(cfg, n_inv, forcing_map=data["forcing_map"])
            logger.info("No checkpoint found — starting from scratch.")
    else:
        model = build_model(cfg, n_inv, forcing_map=data["forcing_map"])

    logger.info("Model: %s  (n_inv=%d  n_params=%s)",
                cfg["model"]["name"], n_inv,
                sum(p.numel() for p in model.parameters()))


    import json as _json
    stat_path = os.path.join(out_dir, "statDict.json")
    with open(stat_path, "w") as f:
        _json.dump(data["scaler"].stat_dict, f, indent=2)
    logger.info("Scaler stats saved → %s", stat_path)


    cfg_path = os.path.join(out_dir, "config_used.yml")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    ecfg = cfg["eval"]
    val_skip = ecfg.get("skip_warmup_years", 3) * 365

    remaining_epochs = tcfg["epochs"] - start_epoch
    if remaining_epochs <= 0:
        logger.info("Already trained for %d epochs. Nothing to do.", start_epoch)
        return model
    logger.info("Training from epoch %d to %d.", start_epoch + 1, tcfg["epochs"])

    model = train(
        model,
        x=data["x_tr"],
        z=data["z_tr"],
        y=data["y_tr"],
        c=data["c_tr"],
        basin_ids=data["basin_ids"],
        y_std=data["y_std"], 
        loss_fn=loss_fn,
        month_tensor=data["month_tensor"],
        doy_tensor=data["doy_tensor"],
        area_tensor=data["area_tensor"],
        lat_tensor=data["lat_tensor"],
        elev_tensor=data["elev_tensor"],
        x_va=data["x_va"],
        z_va=data["z_va"],
        y_va=data["y_va"],
        c_va=data["c_va"],
        month_tensor_va=data["month_tensor_va"],
        doy_tensor_va=data["doy_tensor_va"],
        val_skip_days=val_skip,
        val_batch_size=ecfg.get("batch_size", 50),
        grad_clip=tcfg.get("grad_clip", 1.0),
        start_epoch=start_epoch,
        n_epochs=remaining_epochs,
        batch_size=tcfg["batch_size"],
        seq_len=tcfg["seq_len"],
        warmup=tcfg["warmup"],
        save_every=tcfg["save_every"],
        save_folder=out_dir,
        optimizer_name=tcfg.get("optimizer", "adam"),
        lr=tcfg.get("lr", 5e-4),
        lr_scheduler=tcfg.get("lr_scheduler", "cosine"),
        lr_min=tcfg.get("lr_min", 1e-6),
        gpu=tcfg.get("gpu", 0),
    )

    print(f"[DEBUG] month_tensor shape: {data['month_tensor'].shape}, x_tr T: {data['x_tr'].shape[1]}")
    print(f"[DEBUG] month_tensor_va shape: {data['month_tensor_va'].shape}, x_va T: {data['x_va'].shape[1]}")
    return model


# # ---------------------------------------------------------------------------
# # Evaluate
# # ---------------------------------------------------------------------------

def run_eval(cfg: dict, data: dict, out_dir: str, model=None) -> None:
    from evaluate import evaluate, load_checkpoint

    ecfg = cfg["eval"]
    gpu  = cfg["train"].get("gpu", 0)

    eval_cfg = cfg["eval"]
    batch_size = eval_cfg.get("batch_size", 10)
    print("CONFIG eval.batch_size:", batch_size)
    print("PASSED batch_size:", batch_size)

    if model is None:
        ckpt_path = ecfg.get("checkpoint") or os.path.join(out_dir, "model_best.pt")
        print("Loading checkpoint:", ckpt_path)
        model = load_checkpoint(ckpt_path, gpu)

    warmup_days = ecfg.get("skip_warmup_years", 3) * 365
    eval_dir = os.path.join(out_dir, "evaluation")

    import pandas as pd
    x = data["x_va"]


    import pandas as pd
    x = data["x_va"]

    dcfg = cfg["data"]
    period = dcfg.get("test_period") or dcfg.get("val_period")
    start_date = pd.to_datetime(str(period[0]), format="%Y%m%d")

    prediction_dates = pd.date_range(
        start=start_date,
        periods=x.shape[1],
        freq="D"
    )

    print("LSTM input features:", model.lstm.linear_in.in_features)

    import numpy as np

    pm_records = []

    def _capture_pm(module, input, output):
        rs, albedo, hc = output

        def to_TB(t, T_ref):
            a = t.detach().cpu().numpy()
            if a.ndim == 2 and a.shape[0] != T_ref and a.shape[1] == T_ref:
                a = a.T
            return a

        rs_np = rs.detach().cpu().numpy()
        if rs_np.ndim != 2:
            raise ValueError(f"Expected rs [T,B], got {rs_np.shape}")
        T_ref = rs_np.shape[0]
        pm_records.append({
            "rs":     rs_np,
            "albedo": to_TB(albedo, T_ref),
            "hc":     to_TB(hc, T_ref),
        })

    handle = model.pm_head.register_forward_hook(_capture_pm)

    milc_records = []
    orig_milc_forward = model.milc.forward

    def _wrapped_milc_forward(*args, **kwargs):
        parameters = kwargs.get("parameters", None)
        muwts = kwargs.get("muwts", None)
        corrwts = kwargs.get("corrwts", None)

        if parameters is not None:
            print(
                "phys:", parameters.shape,
                "muwts:", muwts.shape if muwts is not None else None,
                "corrwts:", corrwts.shape if corrwts is not None else None,
            )
            milc_records.append({
                "phys":    parameters.detach().cpu().numpy(),
                "muwts":   muwts.detach().cpu().numpy() if muwts is not None else None,
                "corrwts": corrwts.detach().cpu().numpy() if corrwts is not None else None,
            })

        return orig_milc_forward(*args, **kwargs)

    model.milc.forward = _wrapped_milc_forward

    evaluate(
        model,
        x=data["x_va"],
        z=data["z_va"],
        y=data["y_va"],
        c=data["c_va"],
        month_tensor=data["month_tensor_va"],
        area_tensor=data["area_tensor"],
        lat_tensor=data["lat_tensor"],
        elev_tensor=data["elev_tensor"],
        doy_tensor=data["doy_tensor_va"],
        save_folder=eval_dir,
        skip_warmup_days=warmup_days,
        batch_size=batch_size,
        gpu=gpu,
        y_std=data["y_std"],
        basin_ids=data["basin_ids"],
        basin_area_km2=data["basin_area_km2"],
        prediction_dates=prediction_dates,
    )

    handle.remove()
    model.milc.forward = orig_milc_forward
    print("milc_records collected:", len(milc_records))
    if len(milc_records) == 0:
        raise RuntimeError(
            "model.milc.forward was never called during evaluate() — "
            "check the attribute path / call signature"
        )

    for i, r in enumerate(pm_records):
        print(i, "rs", r["rs"].shape, "albedo", r["albedo"].shape, "hc", r["hc"].shape)

    rs_all     = np.concatenate([r["rs"]     for r in pm_records], axis=1)
    albedo_all = np.concatenate([r["albedo"] for r in pm_records], axis=1)
    hc_all     = np.concatenate([r["hc"]     for r in pm_records], axis=1)

    print("rs_all shape:", rs_all.shape, " expected B =", x.shape[0])

    pm_out_dir = os.path.join(out_dir, "pm_diagnostics")
    os.makedirs(pm_out_dir, exist_ok=True)

    np.save(os.path.join(pm_out_dir, "rs.npy"), rs_all)
    np.save(os.path.join(pm_out_dir, "albedo.npy"), albedo_all)
    np.save(os.path.join(pm_out_dir, "hc.npy"), hc_all)
    print("Saved PM diagnostics to", pm_out_dir)



    phys_all = np.concatenate([r["phys"] for r in milc_records], axis=1)   
    phys_all = np.moveaxis(phys_all, 1, 0)                                  

    dynamic_params = cfg["model"].get("dynamic_params", False)


    if dynamic_params:
        phys_all_physical = phys_all.copy()
        for i, (lo, hi) in enumerate(_PARAM_SCALES):
            phys_all_physical[:, :, i, :] = lo + phys_all_physical[:, :, i, :] * (hi - lo)
        np.save(os.path.join(pm_out_dir, "milc_phys_timevarying.npy"), phys_all_physical)
        phys_static_physical = phys_all_physical.mean(axis=1)  
    else:
        spread = np.abs(phys_all - phys_all[:, :1, :, :]).max()
        print(f"phys max deviation across time (should be ~0 if static): {spread:.3e}")
        phys_static = phys_all[:, 0, :, :]
        phys_static_physical = phys_static.copy()
        for i, (lo, hi) in enumerate(_PARAM_SCALES):
            phys_static_physical[:, i, :] = lo + phys_static_physical[:, i, :] * (hi - lo)

    np.save(os.path.join(pm_out_dir, "milc_phys.npy"), phys_static_physical)


    param_names = [    "gamma",
    "m",
    "ks",
    "nu",
    "alpha",
    "w0",
    "wmax",
    "kc",
    ]

    assert len(param_names) == phys_static_physical.shape[1], (
    f"param_names has {len(param_names)} entries but phys has {phys_static_physical.shape[1]} feature slots"
    )



    B, n_feat, n_mul = phys_static_physical.shape
    phys_2d = phys_static_physical.reshape(B, n_feat * n_mul)
    phys_cols = [f"{param_names[f]}_mul{m}" for f in range(n_feat) for m in range(n_mul)]

    pd.DataFrame(phys_2d, index=data["basin_ids"], columns=phys_cols).to_csv(
        os.path.join(pm_out_dir, "milc_phys.csv"), index_label="basin_id"
    )

    if milc_records[0]["muwts"] is not None:
        muwts_all = np.concatenate([r["muwts"] for r in milc_records], axis=1)
        muwts_all = np.moveaxis(muwts_all, 1, 0)
        np.save(os.path.join(out_dir, "milc_muwts.npy"), muwts_all)
        print("muwts_all shape:", muwts_all.shape)

    if milc_records[0]["corrwts"] is not None:
        corrwts_all = np.concatenate([r["corrwts"] for r in milc_records], axis=1)
        corrwts_all = np.moveaxis(corrwts_all, 1, 0)
        np.save(os.path.join(out_dir, "milc_corrwts.npy"), corrwts_all)
        print("corrwts_all shape:", corrwts_all.shape)

    pm_dates = pd.date_range(start=start_date, periods=rs_all.shape[0], freq="D")
    pm_basin_ids = data["basin_ids"]

    print("n basins in rs_all:", rs_all.shape[1], " n basin_ids:", len(pm_basin_ids))

    for name, arr in [("rs", rs_all), ("albedo", albedo_all), ("hc", hc_all)]:
        df = pd.DataFrame(
            arr.T,                                 
            index=pm_basin_ids,
            columns=pm_dates.strftime("%Y-%m-%d"),
        )
        csv_path = os.path.join(pm_out_dir, f"pm_{name}.csv")
        df.to_csv(csv_path, index_label="basin_id")
        print(f"Saved {name} CSV → {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="dMILC differentiable hydrological model")
    parser.add_argument("--config", default="config.yml", help="Path to YAML config file")
    parser.add_argument("--mode", choices=["train", "eval", "both"], default="both",
                        help="Run mode: train, eval, or both (default: both)")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG/INFO/WARNING)")
    args = parser.parse_args()

    _setup_logging(args.log_level)

    cfg = load_config(args.config)
    set_seeds(cfg["train"]["seed"])

    gpu = cfg["train"].get("gpu", 0)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        logger.info("CUDA device %d: %s", gpu, torch.cuda.get_device_name(gpu))
    else:
        logger.warning("CUDA not available — running on CPU.")

    out_dir = resolve_output_dir(cfg)
    logger.info("Output directory: %s", out_dir)

    logger.info("Loading / building data …")
    data = build_data(cfg, mode=args.mode)

    model = None
    if args.mode in ("train", "both"):
        logger.info("=== TRAINING ===")
        model = run_train(cfg, data, out_dir)

    if args.mode in ("eval", "both"):
        logger.info("=== EVALUATION ===")
        eval_ckpt = cfg.get("eval", {}).get("checkpoint") or os.path.join(out_dir, "model_best.pt")
        if os.path.exists(eval_ckpt):
            logger.info("Loading checkpoint from %s for evaluation...", eval_ckpt)
            model = torch.load(eval_ckpt, map_location=f"cuda:{gpu}" if torch.cuda.is_available() else "cpu", weights_only=False)
        elif model is None:
            raise FileNotFoundError(f"No checkpoint found at {eval_ckpt} to run evaluation.")
            
        run_eval(cfg, data, out_dir, model=model)


    logger.info("Done.")


if __name__ == "__main__":
    main()

