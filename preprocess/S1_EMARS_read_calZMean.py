#!/usr/bin/env python
# ==============================================================================
# SCRIPT original from: EMARS_TW_prep_NH.py
# PURPOSE: Advanced Pre-process EMARS data (Global).
#          1. Calculates true 4D Pressure using EMARS hybrid coefficients.
#          2. Interpolates to fixed levels using calculated 4D Pressure.
#          3. MASKS underground data.
#          4. Calculates Zonal Means and Diagnostics.
#
# ==============================================================================

import sys

import numpy as np
import os
import xarray as xr
import gc
import warnings
import glob

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=xr.SerializationWarning)

# --- Physical Constants (Mars) ---
G        = 3.711
R_GAS    = 188.92
CP       = 846
KAPPA    = R_GAS / CP
P0       = 100000.0
R_EARTH  = 3.3895e6

# --- CONFIGURATION ---
data_dir   = '/depot/wanglei/data/EMARS (real) Data/'
output_dir = '/scratch/bell/hu1029/Mars-BAM_interm/EMARS_preprocessed_dailyVar'
os.makedirs(output_dir, exist_ok=True)

# --- Target Pressure Grid (Fixed Pressure for Climatology) ---
P_LEVELS_PA = np.logspace(np.log10(750.0), np.log10(0.05), 60)
P_LEVELS_PA = P_LEVELS_PA[::-1]  # TOA → surface
print(f"Target pressure levels: {P_LEVELS_PA[0]:.2f} Pa to {P_LEVELS_PA[-1]:.4f} Pa")


# HELPER FUNCTIONS (this version is slightly different from OpenMARS version, but no major influences)

def vectorized_vertical_interp(data_src, p_src, p_targets):
    """Linear interpolation in log-pressure space."""
    n_lev, n_lat, n_lon = data_src.shape
    n_targets = len(p_targets)

    # Ensure source pressure increases for interpolation
    p_mean_profile = np.nanmean(p_src, axis=(1, 2))
    if p_mean_profile[0] > p_mean_profile[-1]:
        p_src    = np.flip(p_src,    axis=0)
        data_src = np.flip(data_src, axis=0)

    target_sort_indices = np.argsort(p_targets)
    p_targets_sorted    = p_targets[target_sort_indices]
    ln_p_src            = np.log(p_src            + 1e-30)
    ln_p_targets        = np.log(p_targets_sorted + 1e-30)
    output_sorted       = np.full((n_targets, n_lat, n_lon), np.nan, dtype=np.float32)

    for k in range(n_lev - 1):
        ln_p_k   = ln_p_src[k]
        ln_p_kp1 = ln_p_src[k + 1]
        val_k    = data_src[k]
        val_kp1  = data_src[k + 1]
        for t_idx, ln_tgt in enumerate(ln_p_targets):
            mask = (
                (ln_p_k <= ln_tgt) & (ln_p_kp1 >= ln_tgt)
                & np.isfinite(val_k) & np.isfinite(val_kp1)
            )
            if np.any(mask):
                denom = np.maximum(ln_p_kp1[mask] - ln_p_k[mask], 1e-10)
                alpha = (ln_tgt - ln_p_k[mask]) / denom
                output_sorted[t_idx, mask] = val_k[mask] + alpha * (val_kp1[mask] - val_k[mask])

    output = np.zeros_like(output_sorted)
    output[target_sort_indices] = output_sorted
    return output


# def high_pass_filter(data, cutoff=30):
#     """
#     High-pass Butterworth filter along axis=0 (time).

#     SAFETY GUARD: scipy.signal.filtfilt requires n_samples > padlen
#     (approximately 27 for a 4th-order filter with cutoff=30).  When
#     n_time is too small the C library corrupts the heap and the process
#     aborts with 'munmap_chunk(): invalid pointer'.  We detect this case
#     and return zeros instead of crashing.
#     """
#     b, a   = signal.butter(4, (1.0 / cutoff) / 0.5, btype='high')
#     padlen = 3 * max(len(a), len(b))          # scipy default padlen
#     n_time = data.shape[0]

#     mask     = np.isnan(data)
#     filtered = np.zeros_like(data, dtype=np.float64)

#     if n_time <= padlen:
#         # Too few timesteps — cannot filter safely; return zeros (NaN-masked below)
#         print(f"        high_pass_filter: n_time={n_time} ≤ padlen={padlen}. "
#               f"Skipping filter; ekeTransientZonal will be zero for this file.")
#     else:
#         filtered = signal.filtfilt(b, a, np.nan_to_num(data), axis=0)

#     filtered[mask] = np.nan
#     return filtered


# ==============================================================================
# MAIN
# ==============================================================================

def process_one_file(filepath):

    base_name = os.path.basename(filepath)
    out_name    = base_name.replace('back_mean', 'preprocessed_global').replace('emars_v1.0_', 'EMARS_')
    output_path = os.path.join(output_dir, out_name)

    # check if all expected single-variable output files exist
    expected_vars = [
    "t", "u", "v",
    "tZonal", "uZonal", "vZonal",
    "ekeZonal",
    "hozMomFluxZonal",
    "hozHeatFluxZonal",
    "mass_stream_func",
    "baroclinicityZonal",
    ]
    expected_outputs = [
        output_path.replace(".nc", f"_{v}.nc")
        for v in expected_vars
    ]
    if all(os.path.exists(p) for p in expected_outputs):
        return f"Skipped: {out_name}"


    try:
        with xr.open_dataset(filepath) as ds:
            ds_global = ds.sel(lat=slice(-90, 90))

            # --- HYBRID PRESSURE CALCULATION ---
            # EMARS coefficients: ak (Pa) and bk (dimensionless sigma)
            ak    = ds_global['ak'].values          # (phalf,)
            bk    = ds_global['bk'].values          # (phalf,)
            ps_pa = ds_global['ps'].values          # (time, lat, lon)

            # Pressure at half-levels:  p_ik = Psfc * bk + ak
            p_interface = (
                ps_pa[:, None, :, :] * bk[None, :, None, None]
                + ak[None, :, None, None]
            )

            # Pressure at full levels (log-linear midpoint):
            # p = Δp / ln(p_k+1 / p_k)
            p_4d = (
                (p_interface[:, 1:, :, :] - p_interface[:, :-1, :, :])
                / np.log(p_interface[:, 1:, :, :] / p_interface[:, :-1, :, :])
            )

            t_in = ds_global['t'].values
            u_in = ds_global['u'].values
            v_in = ds_global['v'].values
            n_time, n_lev, n_lat, n_lon = t_in.shape

            t_out = np.zeros((n_time, len(P_LEVELS_PA), n_lat, n_lon), dtype=np.float32)
            u_out = np.zeros_like(t_out)
            v_out = np.zeros_like(t_out)

            for t_idx in range(n_time):
                if t_idx % 200 == 0:
                    print(f"       Progress: {t_idx}/{n_time}")
                t_out[t_idx] = vectorized_vertical_interp(t_in[t_idx], p_4d[t_idx], P_LEVELS_PA)
                u_out[t_idx] = vectorized_vertical_interp(u_in[t_idx], p_4d[t_idx], P_LEVELS_PA)
                v_out[t_idx] = vectorized_vertical_interp(v_in[t_idx], p_4d[t_idx], P_LEVELS_PA)

                # Mask data below the surface
                underground = P_LEVELS_PA[:, None, None] >= ps_pa[t_idx][None, :, :]
                t_out[t_idx][underground] = np.nan
                u_out[t_idx][underground] = np.nan
                v_out[t_idx][underground] = np.nan

            # --- Build xarray DataArrays (zonal fields only — no lon dim saved) ---
            coords_3d = {
                'time':  ds_global.time,
                'level': P_LEVELS_PA,
                'lat':   ds_global.lat,
                'lon':   ds_global.lon,
            }
            T = xr.DataArray(t_out, coords=coords_3d, dims=('time', 'level', 'lat', 'lon'))
            U = xr.DataArray(u_out, coords=coords_3d, dims=('time', 'level', 'lat', 'lon'))
            V = xr.DataArray(v_out, coords=coords_3d, dims=('time', 'level', 'lat', 'lon'))

            # Zonal means
            tZ = T.mean('lon')
            uZ = U.mean('lon')
            vZ = V.mean('lon')

            # Eddy components
            u_p = U - uZ
            v_p = V - vZ
            t_p = T - tZ

            # Diagnostics
            ekeZ  = (0.5 * (u_p**2 + v_p**2)).mean('lon')
            hMFZ  = (u_p * v_p).mean('lon')
            hHFZ  = (v_p * t_p).mean('lon')

            # # Transient EKE (high-pass filtered) — safe against short files
            # eke_trans_raw = 0.5 * (
            #     high_pass_filter(u_out)**2 + high_pass_filter(v_out)**2
            # )
            # ekeTransZ = np.nanmean(eke_trans_raw, axis=3)   # zonal mean

            # Mass stream function
            dp  = xr.DataArray(
                np.gradient(P_LEVELS_PA), dims=['level'], coords={'level': P_LEVELS_PA}
            )
            msf = (
                (2 * np.pi * R_EARTH / G)
                * np.cos(np.deg2rad(vZ.lat))
                * (vZ * dp).cumsum('level')
            )

            # Baroclinicity
            thetaZ = tZ * (P0 / tZ.level) ** KAPPA
            N2     = (
                (G / thetaZ)
                * thetaZ.differentiate('level')
                * (tZ.level / (-R_GAS * tZ / G))
            )
            N      = np.sqrt(np.abs(N2))
            dy     = R_EARTH * np.deg2rad(np.gradient(ds_global.lat))
            baro   = (
                0.31
                * (G / xr.where(N <= 0, 1e-10, N))
                * (thetaZ.differentiate('lat') / dy / thetaZ)
            )

            # Scalar time variables (MY, Ls, etc.). Keep only the following:
            KEEP_TIME_VARS = [
                "emars_sol",
                "MY",
                "Ls",
                "mars_soy",
                "macda_sol",
            ]
            time_vars = {
                v: ds_global[v]
                for v in KEEP_TIME_VARS
                if v in ds_global and ds_global[v].dims == ('time',)
            }

            res = xr.Dataset({
                't': T.astype('float32'),
                'u': U.astype('float32'),
                'v': V.astype('float32'),
                'tZonal':             tZ.astype('float32'),
                'uZonal':             uZ.astype('float32'),
                'vZonal':             vZ.astype('float32'),
                'ekeZonal':           ekeZ.astype('float32'),
                'hozMomFluxZonal':    hMFZ.astype('float32'),
                'hozHeatFluxZonal':   hHFZ.astype('float32'),
                # 'ekeTransientZonal':  (('time', 'level', 'lat'), ekeTransZ.astype('float32')),
                'mass_stream_func':   msf.astype('float32'),
                'baroclinicityZonal': baro.astype('float32'),
            })

            full_res = xr.merge([res, time_vars])

            daily = full_res.groupby("emars_sol").mean("time", skipna=True)
            daily = daily.rename({"emars_sol": "time"})
            daily["emars_sol"] = ("time", daily["time"].values.astype(np.float32))

            # save single variable files
            for varname in res.data_vars:
                one = daily[[varname] + [v for v in KEEP_TIME_VARS if v in daily]]

                out_var_path = output_path.replace(".nc", f"_{varname}.nc")
                tmp_var_path = out_var_path + ".tmp"

                if os.path.exists(tmp_var_path):
                    os.remove(tmp_var_path)
                encoding = {
                    v: {"zlib": True, "complevel": 4}
                    for v in one.data_vars
                }

                one.to_netcdf(tmp_var_path, encoding=encoding)
                os.rename(tmp_var_path, out_var_path)


            del t_out, u_out, v_out, T, U, V, t_in, u_in, v_in
            del p_4d, p_interface #, eke_trans_raw
            gc.collect()

            return f"Saved: {out_name}"

    except Exception as e:
        # Remove any partial .tmp so the next run retries cleanly
        for tmp in glob.glob(output_path.replace(".nc", "_*.nc.tmp")):
            os.remove(tmp)
        return f"Failed: {base_name}: {repr(e)}"   



# Parallel processing of EMARS files using ProcessPoolExecutor
if __name__ == "__main__":

    file_list = sorted(
        glob.glob(os.path.join(data_dir, "emars_v1.0_back_mean_*.nc"))
    )

    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if task_id >= len(file_list):
        print(f"Task {task_id}: no file assigned")
        sys.exit(0)

    filepath = file_list[task_id]

    print(f"Task {task_id}")
    print(f"Processing: {os.path.basename(filepath)}")

    result = process_one_file(filepath)

    print(result, flush=True)
