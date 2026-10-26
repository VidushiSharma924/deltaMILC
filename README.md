# δMILC-DP 1.0 — Differentiable MILC Hydrological Model for India (CAMELS-IND)

δMILC-DP couples an LSTM parameter generator with a differentiable implementation of the
**MILC** (Modello Idrologico Semi-Distribuito in Continuo) soil-water-balance model and a
**Penman–Monteith** PET module. The whole chain — LSTM → PET → soil water balance → IUH routing —
is written in PyTorch and trained end-to-end on daily streamflow from the
[CAMELS-IND](https://doi.org/10.5194/essd-17-461-2025) catchments.

```
normalized forcings + static attributes ──► LSTM ──┬─► MILC parameters (8 × 16 components, per day)
                                                   └─► PM parameters (rs, albedo, crop height)
raw forcings (tmax, tmin, srad, wind, RH) ──► Penman–Monteith PET
prcp, tmean, PET ──► MILC soil water balance ──► GIUH / linear-reservoir routing ──► Qs, qd, qb, ET
```

---

## Contents

- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Data](#data)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Model description](#model-description)
- [Outputs](#outputs)
- [Results of the included run](#results-of-the-included-run)
- [Known limitations](#known-limitations)
- [References](#references)

---

## Repository layout

```
MILC-DP_1.0/
├── run.py                       # Entry point: train / eval / both (training basins)
├── run_inference.py             # Entry point: eval on new (held-out) basins
├── trainer.py                   # Mini-batch training loop, per-epoch validation, checkpointing
├── evaluate.py                  # Batched inference, metrics, CSV export
├── config.yml                   # Config for training (Train.txt basins)
├── config_inference_new_basins_SUBSET_54.yml   # Config for inference (Test.txt basins)
├── Steps_to_run_the_model.ipynb # Shell commands used to run the model
├── dmilc_environment.yml        # Conda environment (Linux, CUDA 11.8)
├── pip_requirements.txt         # pip packages of that environment
│
├── model/
│   ├── lstm.py                  # CudnnLstm, CudnnLstmModel, PMParamHead, MultiInv_MILCModel
│   ├── milc.py                  # MILCMul physics, parameter ranges, IUH kernels, routing
│   └── loss.py                  # KGE, NSE, combined KGE+NSE, RMSE+log losses
│
├── data/
│   ├── dataset.py               # CAMELS-IND loader, pickle cache, area/month/DOY tensors
│   └── dataset_inference.py     # Variant used by run_inference.py
│
├── utils/
│   ├── penman_monteith.py       # FAO-56-style Penman–Monteith PET with variable crop height
│   ├── radiation.py             # Extraterrestrial radiation and day length
│   ├── scaler.py                # HydroScaler (mean/std normalization)
│   └── metrics.py               # NSE, KGE, Corr, RMSE, Bias, ubRMSE, FLV, FHV, ...
│
├── CAMELS_IND/                  # Dataset + basin lists (see "Data")
└── experiments/                 # Trained run and evaluation outputs
```

`CAMELS_IND/CAMELS_IND_Catchments_Streamflow_No_Nan/hydroDL/` is a bundled copy of the MHPI
`hydroDL` library. It is not imported by the δMILC-DP code.

---

## Installation

The reference environment is Linux + CUDA 11.8 (PyTorch 2.2.2, Python 3.10).

```bash
conda env create -f dmilc_environment.yml
conda activate dmilc_environment
```

`dmilc_environment.yml` pins Linux-only build packages (`_libgcc_mutex`, `alsa-lib`, …). On Windows or
macOS, create the environment by hand instead:

```bash
conda create -n dmilc python=3.10
conda activate dmilc
# GPU build (CUDA 11.8); use the CPU wheel from pytorch.org if you have no GPU
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.26.4 pandas scipy pyyaml tqdm
```

Only `torch`, `numpy`, `pandas`, `scipy`, `pyyaml` and `tqdm` are needed to train and evaluate.
The geospatial packages in the environment files (`geopandas`, `rasterio`, …) are not used by the model code.

A GPU is strongly recommended. The code runs on CPU, but there the LSTM has no weight dropout
(see `CudnnLstm.forward`).

---

## Data

The model reads the CAMELS-IND files under `CAMELS_IND/CAMELS_IND_Catchments_Streamflow_No_Nan/`:

| Path | Content |
|------|---------|
| `catchment_mean_forcings/<gauge>.csv` | Daily forcings, 1980-01-01 – 2020-12-31 (`prcp`, `tmax`, `tmin`, `tavg`, `srad_sw`, `srad_lw`, `wind`, `rel_hum`, …) |
| `streamflow_timeseries/final_filled.csv` | Daily discharge (m³/s) with columns `Date, Basin_ID, Q_obs, Q_lstm, mask, Q_filled` |
| `attributes_txt/camels_ind_attributes.csv` | Static catchment attributes (`gauge_id` + 210 columns) |
| `CAMELS_IND_Data_Description.pdf` | Variable definitions and sources |

Basin splits are JSON lists of 5-digit gauge IDs:

| File | Basins | Used by |
|------|-------:|---------|
| `CAMELS_IND/Train.txt` | 121 | `config.yml` (training + temporal validation) |
| `CAMELS_IND/Test.txt` | 50 | `config_inference_new_basins_SUBSET_54.yml` (spatial hold-out) |

The two lists do not overlap.

**Streamflow handling.** Discharge is read from `Q_filled` and converted from m³/s to mm/day using
`cwc_area`. Days where `mask == 0` (gap-filled, not observed) are set to NaN, so only observed days
enter the loss and the metrics.

**Caching.** On first use the extracted arrays are pickled to `training_file`, `validation_file` or
`test_file` under `root_db`. The cache is not keyed on basins or dates. Keep `rebuild_cache: true`
(the default in both configs) whenever you change the basin list, the periods or the attributes.

CAMELS-IND is distributed for academic use. Please cite the data paper (see [References](#references))
and read `Disclaimer.txt`.

---

## Quick start

Run all commands from the `MILC-DP_1.0/` folder that contains `run.py`. The repository paths in the
configs are relative to that folder.

### 1. Train and evaluate on the training basins

```bash
python run.py --config config.yml --mode both      # train, then evaluate model_best.pt
python run.py --config config.yml --mode train     # train only
python run.py --config config.yml --mode eval      # evaluate an existing checkpoint
```

Long runs in the background (as in `Steps_to_run_the_model.ipynb`):

```bash
nohup python run.py --config config.yml --mode both > DP_train.log 2>&1 &
```

> **Before training, check `config.yml`.** It is currently set to `epochs: 1` and
> `eval.checkpoint: experiments/MILC-DP_1.0_India_Train1/model_best.pt`. To reproduce the included run,
> set `epochs: 100`, `nse_weight: 0.5`, and make `eval.checkpoint` point to
> `experiments/<output.run_name>/model_best.pt`. Also check `train.gpu`, which is set to `1`
> (the second GPU).

### 2. Evaluate on new (held-out) basins

```bash
python run_inference.py --config config_inference_new_basins_SUBSET_54.yml --mode eval
```

This loads the checkpoint in `eval.checkpoint` (default
`experiments/MILC-DP_1.0_India_Train/model_best.pt`). It also needs `statDict.json` and `y_std.npy`
from the same folder, and it runs over `test_period`. `run_inference.py` supports only `--mode eval`.

`--log-level DEBUG|INFO|WARNING` is accepted by both scripts.

---

## Configuration

Both entry points read one YAML file. These are the keys that matter:

### `data`

| Key | Meaning |
|-----|---------|
| `root_db` | CAMELS-IND root; cache files are written here |
| `forcing_root` | Folder with `<gauge>.csv` forcing files |
| `flow_path` | `final_filled.csv` (used when the forcing CSV has no `Q` column) |
| `attr_path` | Static attribute CSV |
| `subset_file` | JSON basin list (`Train.txt` / `Test.txt`) |
| `train_period`, `val_period`, `test_period` | `[YYYYMMDD, YYYYMMDD]`; the start is inclusive and the end is **exclusive** |
| `train_cache`, `val_cache`, `test_cache` | Cache file names under `root_db` |
| `attr_vars` | Static attributes fed to the LSTM (110 in the shipped configs). `cwc_area` is required |
| `rebuild_cache` | Re-extract from the raw CSVs instead of loading the pickle |

`data.forcing_vars` is **not read**: the forcing list is hard-coded in `build_data()` as
`prcp, tmax, tmin, tavg, srad_sw, srad_lw, wind, rel_hum` (8 variables).

### `model`

| Key | Shipped value | Meaning |
|-----|---------------|---------|
| `name` | `δMILC-DP_1.0` | Run label. Precipitation is log-normalized only when this equals `dMILC1.0` |
| `n_components` | 16 | Parallel MILC components (Nmul) per basin |
| `hidden_size` | 256 | LSTM hidden size |
| `lstm_dropout` | 0.4 (train) | LSTM weight dropout (GPU only) |
| `n_physics_params` | 8 | MILC parameters per component |
| `dynamic_params` | `true` | Per-day MILC parameters instead of one set per sequence |
| `static_pm_params` | `true` | One rs/albedo/hc set per sequence instead of per-day values |
| `component_weights` | `false` | Learned softmax weights over components (otherwise a simple mean) |
| `precip_correction` | `false` | Learned precipitation multiplier in [0.5, 1.5] |

### `train`

| Key | Meaning |
|-----|---------|
| `epochs`, `lr`, `lr_scheduler` (`cosine`/`none`), `lr_min`, `optimizer` (`adam`/`adadelta`) | Optimization |
| `batch_size` | Basins per mini-batch (capped at the number of basins; 10000 ⇒ all basins every step) |
| `seq_len` | Length of each training window (days) |
| `warmup` | Physics spin-up days prepended to each window |
| `grad_clip` | Global gradient-norm clip (0 disables it) |
| `save_every` | Save `model_Ep<N>.pt` every N epochs |
| `resume` | Continue from the newest `model_Ep*.pt` in the output folder |
| `gpu`, `seed` | CUDA device index and random seed |

The number of iterations per epoch is derived automatically, so that each day is sampled with ≈99%
probability per epoch.

### `loss`

| `name` | Loss |
|--------|------|
| `nse_kge` | `(1 − nse_weight)·(1 − KGE) + nse_weight·NSE-loss` |
| `kge_batch` | `1 − KGE` per basin, averaged |
| `nse_batch` | Squared error normalized by per-basin training std |
| `rmse_comb` | `(1 − α)·RMSE + α·RMSE(log10(√Q + 0.1))` with `α = rmse_alpha` |

`eps` floors the std/mean denominators.

### `output` and `eval`

| Key | Meaning |
|-----|---------|
| `output.root`, `output.run_name` | Output folder. `run.py` uses `<root>/<run_name>` relative to the working directory. `run_inference.py` uses `<root_db>/<root>/<run_name>`. Leave `run_name` empty for an auto-generated name |
| `eval.checkpoint` | Model to evaluate; its folder must also contain `statDict.json` and `y_std.npy` |
| `eval.batch_size` | Basins per inference batch |
| `eval.skip_warmup_years` | Years removed from the start of the scored period (after the model `warmup`) |

`eval.epoch` and `eval.warmup` are present in the configs but not used.

---

## Model description

### 1. LSTM parameter generator (`model/lstm.py`)

- Input `z`: `[T, B, 8 + n_attr]` = normalized forcings + static attributes repeated over time
  (118 features with the shipped configs).
- `Linear → ReLU → CudnnLstm → Linear`, with output size `n_physics·n_mul (+ n_mul) (+ 1)` = 128 by default.
- `CudnnLstm` calls the cuDNN kernel directly and applies variational **weight** dropout to `W_ih` and `W_hh`.
- **MILC parameters**: with `dynamic_params: true`, `sigmoid(gen[warmup:])` gives a per-day
  `[T, B, 8, 16]` tensor. Otherwise the time-mean of the LSTM output is used.
- **PM parameters** (`PMParamHead`): a linear layer on the LSTM output maps to
  surface resistance `rs ∈ [50, 250] s m⁻¹`, `albedo ∈ [0.1, 0.4]` and crop height `hc ∈ [0.05, 2.5] m`.

### 2. Penman–Monteith PET (`utils/penman_monteith.py`)

This follows FAO-56, but with a variable crop height and learned `rs` and albedo:

- 10 m wind is reduced to 2 m with the FAO log-profile.
- Net shortwave radiation is `(1 − albedo)·Rs`, and net longwave uses the FAO Rs/Rso cloud factor.
- The aerodynamic resistance uses `d = 2/3·hc`, `z0m = 0.123·hc` and `z0h = 0.1·z0m`, with `hc` clipped to 2.5 m.
- Extraterrestrial radiation and day length come from latitude and day of year (`utils/radiation.py`).
- The result is clipped to `[0, 20] mm day⁻¹`.

### 3. MILC soil water balance (`model/milc.py`)

The model runs 16 parallel components per basin, each with its own parameters. Each day:

```
sat    = W / wmax
infil  = P · (1 − sat^alpha)            (limited so that W ≤ wmax)
runoff = P − infil                      → surface runoff (qd before routing)
ET     = PET · sat                      (after infiltration)
D      = ks · sat^m                     drainage
qb     = (1 − nu) · D                   → baseflow
perc   = nu · D                         → deep percolation (leaves the system)
W      = max(W + infil − ET − qb − perc, 1e-5)
```

| # | Parameter | Range | Role |
|---|-----------|-------|------|
| 0 | `gamma` | 0.0255 – 25 (clamped to 0.1 – 5 in routing) | GIUH lag coefficient |
| 1 | `m` | 0.00254 – 30 | Drainage exponent |
| 2 | `ks` | 2.84 – 5000 mm d⁻¹ | Saturated conductivity |
| 3 | `nu` | 0.01 – 1 | Percolation fraction of drainage |
| 4 | `alpha` | 0.105 – 15 | Infiltration exponent |
| 5 | `w0` | 0.1 – 1 | Initial relative saturation |
| 6 | `wmax` | 1.84 – 10000 mm | Soil storage capacity |
| 7 | `kc` | 0.101 – 2 | Crop coefficient (only used by the Pruitt fallback when no PET is supplied) |

**Spin-up.** The first `warmup` days of each window are simulated with time-averaged parameters to get
the initial storage `W`. That state is then detached from the graph.

### 4. Routing

- Surface runoff is convolved with a geomorphological IUH. The lag is
  `lag = gamma · 1.19 · A^0.33 / 24` days, with `A` in km², and the shape comes from a dimensionless lookup table.
- Baseflow is convolved with a single linear reservoir (`K = 0.5·lag`).
- Both convolutions are batched with grouped `conv1d` over all basins × components. The component
  outputs are averaged (or weighted, if `component_weights: true`).

Model outputs per day are `Qs` (total streamflow), `qd` (routed surface flow), `qb` (routed baseflow)
and `ET`.

### 5. Target scaling

Observed streamflow is divided by each basin's training-period std (`y_std.npy`, floored at 0.1)
before training. At evaluation, `Qs`, `qd` and `qb` are multiplied back by the same factor. `ET` is
not rescaled. See [Known limitations](#known-limitations).

---

## Outputs

Training (`run.py --mode train|both`) writes to `experiments/<run_name>/`:

```
config_used.yml        # copy of the config
statDict.json          # normalization statistics (needed for eval/inference)
y_std.npy              # per-basin target scale (needed for eval)
train_log.csv          # epoch, train_loss, time_s, val_NSE, val_KGE, val_Corr, val_RMSE (medians)
model_Ep<N>.pt         # full pickled model every save_every epochs (~6 MB each; 100 epochs ≈ 590 MB)
model_best.pt          # best median validation KGE
```

Evaluation writes:

```
evaluation/
├── Qs.csv  qd.csv  qb.csv  ET.csv   # rows = basin_id, columns = dates, mm/day
└── metrics.csv                      # per-basin NSE, KGE, Bias, RMSE, ubRMSE, Corr, FLV, FHV, lowRMSE, highRMSE
pm_diagnostics/                      # run.py only
├── pm_rs.csv  pm_albedo.csv  pm_hc.csv   (+ .npy)   # Penman–Monteith parameters per basin and day
├── milc_phys.csv / .npy                         # time-mean MILC parameters (physical units), basin × param_component
└── milc_phys_timevarying.npy                    # per-day MILC parameters (dynamic_params only)
```

Checkpoints are whole pickled `nn.Module` objects. Load them from the repository root (so that
`model.*` is importable) with:

```python
model = torch.load("experiments/MILC-DP_1.0_India_Train/model_best.pt", weights_only=False)
```

---

## Results of the included run

`experiments/MILC-DP_1.0_India_Train` was trained for 100 epochs with the settings in its
`config_used.yml`: 121 basins, `nse_kge` with `nse_weight: 0.5`, `lstm_dropout: 0.4`, `seq_len` and
`warmup` = 365. The best validation KGE was at epoch 67.

These are medians over basins:

| Evaluation | Basins | Scored period | NSE | KGE | Corr | FHV (%) | FLV (%) |
|------------|-------:|---------------|----:|----:|-----:|--------:|--------:|
| Temporal validation (`run.py`) | 121 | ≈1990-10 – 1992-09 | 0.659 | 0.666 | 0.863 | −17.9 | −33.0 |
| Spatial hold-out (`run_inference.py`) | 50 | 1985-01 – 2020-12 | 0.442 | 0.519 | 0.716 | −3.3 | −7.5 |

The temporal-validation numbers come from the same period that was used to pick `model_best.pt`, so
they are optimistic. Also read the hold-out row together with the first two
[known limitations](#known-limitations).

---

## Known limitations

These come from a code review of this version. They should be addressed before results are published.

1. **The physics model does not output physical streamflow.** The model is trained against `Q / y_std`
   but runs on precipitation and PET in mm/day. It must therefore produce runoff at the scale of
   `Q / σ_basin`, which breaks the water balance. `qd`, `qb` and `Qs` are only in mm/day after they are
   multiplied by `y_std`. `ET` is never rescaled, so it is not consistent with `Qs`.
2. **The hold-out evaluation uses the test observations.** `run_inference.py` computes `y_std` from the
   *test* basins' observed streamflow and uses it to rescale the predictions. This would not be possible
   for truly ungauged basins.
3. **Drainage is not limited by the available storage.** `ks · sat^m` can exceed `W` (`ks` can reach
   5000 mm/day). `W` is then clipped to 1e-5, but `qb` is not reduced, so water is created.
4. **`gamma` is clamped to [0.1, 5]** while its scaling range goes up to 25. About 80% of the sigmoid
   range therefore gives no gradient. The learned values in `pm_diagnostics/milc_phys.csv` are mostly 8–20.
5. **Shortwave radiation units.** The PET code converts `srad_sw` as a daylight-hours mean
   (`× daylength`). CAMELS-IND `srad_sw` is an IMDAA daily-mean flux, which would need `× 86400 s`.
   Please check this. If it is a 24 h mean, Rs and therefore PET are underestimated.
6. **Model selection and reporting use the same period.** `--mode both` evaluates on `val_period`,
   which was also used to choose `model_best.pt`. After the 1-year warmup and 3 skipped years, only
   about 2 years are scored.
7. **Resuming resets the training state.** Resuming restarts Adam and the cosine schedule and resets
   the best-KGE tracker, because checkpoints store only the model.
8. **Config and robustness issues.**
   - `run.py --mode eval` reads `cfg["eval"]["checkpoint"]` directly, so it fails if the key is null.
   - In eval mode the attribute list comes from the config, not from the training run.
   - `y_std` is matched to basins by count only, not by ID.
   - `run_inference.py` builds month/DOY tensors without the test period and hard-codes the start
     date `1981-01-01`.

---

## References

- **CAMELS-IND:** Mangukiya, N. K., Kumar, K. B., Dey, P., et al. (2025). CAMELS-IND: hydrometeorological
  time series and catchment attributes for 228 catchments in Peninsular India. *Earth System Science Data*, 17, 461–491.
- **MILC model:** Brocca, L., Melone, F., & Moramarco, T. (2011). Distributed rainfall-runoff modelling for
  flood frequency estimation and flood forecasting. *Hydrological Processes*, 25(18), 2801–2813.
- **Differentiable parameter learning:** Tsai, W.-P., Feng, D., Pan, M., et al. (2021). From calibration to
  parameter learning: harnessing the scaling effects of big data in geoscientific modeling.
  *Nature Communications*, 12, 5988.
- **Penman–Monteith / FAO-56:** Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). *Crop
  evapotranspiration — Guidelines for computing crop water requirements.* FAO Irrigation and Drainage Paper 56.
- **GIUH:** Rodríguez-Iturbe, I., & Valdés, J. B. (1979). The geomorphologic structure of hydrologic response.
  *Water Resources Research*, 15(6), 1409–1420.
- **KGE:** Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared
  error and NSE performance criteria. *Journal of Hydrology*, 377(1–2), 80–91.
