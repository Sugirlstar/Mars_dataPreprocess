#!/usr/bin/env python
# ==============================================================================
# SCRIPT original from: OpenMARS_prep_full.py
# PURPOSE: Advanced Pre-process EMARS data (Global).
#          1. Calculates true 4D Pressure using EMARS hybrid coefficients.
#          2. Interpolates to fixed levels using calculated 4D Pressure.
#          3. MASKS underground data.
#          4. Calculates Zonal Means and Diagnostics.
#
# ==============================================================================



"""OpenMARS Northern Hemisphere (NH) 4D Masked Pre-processing Pipeline
===================================================================
Dataset: OpenMARS (Mars Years 28-36)
Domain: Global
Target Grid: 30 log-spaced pressure levels spanning 750 Pa to 0.05 Pa

Objective:
----------
This script reads raw OpenMARS NH files on hybrid-sigma levels, reconstructs
pressure on a fixed 4D grid, interpolates ``temp``, ``u``, and ``v`` to a
shared pressure coordinate, masks subsurface values, and writes per-file
derived NetCDF products for downstream diagnostics.

Scientific Context:
-------------------
The original file-level documentation frames this script as the 4D masked
preprocessing stage for the OpenMARS BAM and storm-track workflow. In the code,
the derived products actually written to disk are:
1. Zonal-mean fields ``tZonal`` and ``uZonal``.
2. Zonal-mean eddy diagnostics ``ekeZonal``, ``hozMomFluxZonal``, and
   ``hozHeatFluxZonal``.
3. Longitude-resolved 4D products ``hozMomFlux_4D``, ``hozHeatFlux_4D``,
   ``u``, ``v``, and ``temp``.

Inputs:
-------
- Raw files discovered from ``/depot/wanglei/data/OpenMARS Data/OpenMARS MY28-36/*.nc``
- Source variables used directly in code: ``ps``, ``lev``, ``temp``, ``u``,
  ``v``, plus ``time``, ``lat``, and ``lon`` coordinates

Outputs:
--------
- One derived NetCDF per input file in
  ``/depot/wanglei/data/OpenMARS_derived_full_fields/``
- Output filenames constructed as
  ``OpenMARS_preprocessed_full_NH_`` + lowercased input basename after
  removing the ``OpenMARS_`` prefix
- Saved variables: ``tZonal``, ``uZonal``, ``ekeZonal``,
  ``hozMomFluxZonal``, ``hozHeatFluxZonal``, ``hozMomFlux_4D``,
  ``hozHeatFlux_4D``, ``u``, ``v``, ``temp``, plus time-only variables copied
  from the source file

Methodology:
------------
1. Input discovery:
   Gather all ``*.nc`` files from ``data_dir``
2. NH extraction:
   Open each dataset and subset latitude with ``ds.sel(lat=slice(90, 0))``.
3. Surface-pressure normalization:
   Inspect the mean of ``ps`` and rescale by 100 when the values appear to be
   inconsistent with the Pa-based target grid.
4. Pressure reconstruction:
   Reconstruct 4D physical pressure as ``lev_sigma * ps`` for each time,
   level, latitude, and longitude point.
5. Vertical interpolation:
   Interpolate ``temp``, ``u``, and ``v`` to the target pressure levels using
   ``vectorized_vertical_interp`` and ``ln(p)`` interpolation.
6. Underground masking:
   For each time step, apply a mask where ``P_target >= ps(surface)`` and set
   those interpolated values to ``NaN``.
7. Derived diagnostics:
   Compute zonal means, perturbations relative to zonal means, zonal-mean eddy
   kinetic energy, zonal-mean eddy momentum and heat fluxes, and 4D products
   ``u * v`` and ``v * temp``.
8. Serialization:
   Merge derived variables with time-only source variables and write a
   compressed NetCDF using ``engine='netcdf4'`` and per-variable zlib
   compression.
9. Overwrite behavior:
   If an output file already exists, delete it before rewriting it.

Mars Physical Constants:
------------------------
- Gravity (``G``): 3.711 m/s^2
- Specific Gas Constant (``R_GAS``): 188.92 J/kg/K
- Specific Heat (``CP``): 846 J/kg/K
- ``KAPPA = R_GAS / CP``
- Reference Pressure (``P0``): 100000 Pa
- Planetary Radius (``R_EARTH`` variable; Mars radius value): 3.3895e6 m

Audit note:
-----------
Downstream scripts in this repository appear to expect a more specific output
filename pattern (``openmars_my*.nc``) than the filename construction used
here, so filename compatibility should be checked separately from this
preprocessing step.
"""
import sys

import numpy as np
import os
import xarray as xr
import gc
import warnings
import glob
import re
import os

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=xr.SerializationWarning)

# --- Physical Constants (Mars) ---
G = 3.711           
R_GAS = 188.92       
CP = 846            
KAPPA = R_GAS / CP  
P0 = 100000.0       
R_EARTH = 3.3895e6  

# --- CONFIGURATION ---
data_dir = '/depot/wanglei/data/OpenMARS Data/OpenMARS MY28-36/'
output_dir = '/scratch/bell/hu1029/Mars-BAM_interm/OpenMars_preprocessed_dailyVar' 
os.makedirs(output_dir, exist_ok=True)

# --- Target Pressure Grid ---
# 30 levels from 750 Pa to 0.05 Pa, ordered High -> Low (Surface -> TOA)
P_LEVELS_PA = np.logspace(np.log10(750.0), np.log10(0.05), 30)
P_LEVELS_PA = P_LEVELS_PA[::-1]

print(f"Target pressure levels: {P_LEVELS_PA[0]:.2f} Pa to {P_LEVELS_PA[-1]:.4f} Pa")

# --- Helper: Vectorized Vertical Interpolation ---
def vectorized_vertical_interp(data_src, p_src, p_targets):

    """Interpolates 3D (Lev, Lat, Lon) data to 1D target pressure levels."""
    n_lev, n_lat, n_lon = data_src.shape
    n_targets = len(p_targets)
    
    # Sort source pressure LOW -> HIGH for interpolation
    p_mean_profile = np.nanmean(p_src, axis=(1, 2))
    if p_mean_profile[0] > p_mean_profile[-1]:
        p_src = np.flip(p_src, axis=0)
        data_src = np.flip(data_src, axis=0)
    
    # Sort target pressures LOW -> HIGH
    target_sort_indices = np.argsort(p_targets)
    p_targets_sorted = p_targets[target_sort_indices]
    
    ln_p_src = np.log(p_src + 1e-30)
    ln_p_targets = np.log(p_targets_sorted + 1e-30)
    
    output_sorted = np.full((n_targets, n_lat, n_lon), np.nan, dtype=np.float32)
    
    for k in range(n_lev - 1):
        ln_p_k = ln_p_src[k, :, :]
        ln_p_kp1 = ln_p_src[k+1, :, :]
        val_k = data_src[k, :, :]
        val_kp1 = data_src[k+1, :, :]
        
        for t_idx, ln_tgt in enumerate(ln_p_targets):
            mask = (ln_p_k <= ln_tgt) & (ln_p_kp1 >= ln_tgt)
            mask = mask & np.isfinite(val_k) & np.isfinite(val_kp1)
            
            if np.any(mask):
                denom = ln_p_kp1[mask] - ln_p_k[mask]
                denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
                alpha = (ln_tgt - ln_p_k[mask]) / denom
                output_sorted[t_idx, mask] = val_k[mask] + alpha * (val_kp1[mask] - val_k[mask])
    
    output = np.zeros_like(output_sorted)
    # Undo sorting to match input P_LEVELS_PA order
    output[target_sort_indices, :, :] = output_sorted
    return output

# ==============================================================================
# Main Execution
# ==============================================================================

def process_one_file(filepath):

    base_name = os.path.basename(filepath)

    m = re.match(
        r"openmars_my(\d+)_ls(\d+)_my(\d+)_ls(\d+)\.nc",
        base_name,
        re.IGNORECASE,
    )

    if m is None:
        raise ValueError(f"Unexpected OpenMARS filename: {base_name}")

    my1, ls1, my2, ls2 = map(int, m.groups())

    out_name = (
        f"OPENMARS_preprocessed_global_"
        f"MY{my1:02d}_Ls{ls1:03d}_"
        f"MY{my2:02d}_Ls{ls2:03d}.nc"
    )
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


    print(f"Processing {base_name}")

    try:
        with xr.open_dataset(filepath) as ds_global:

            # 1. Load Data (global)
            # add a dimension, i.e., integer_sol, representing the integer part of the sol (Martian day) for getting daily mean
            ds_global = ds_global.assign_coords(
                integer_sol=(
                    "time",
                    np.ceil(ds_global["time"].values - 1e-6).astype(np.int32) - 1
                )
            )
            
            # Check PS units
            ps_raw = ds_global['ps'].values
            if np.nanmean(ps_raw) > 10000: ds_global['ps'] /= 100.0
            elif np.nanmean(ps_raw) < 100: ds_global['ps'] *= 100.0
            
            # Calculate 4D Pressure
            lev_sigma = ds_global['lev'].values 
            ps_3d = ds_global['ps'].values      
            p_4d = lev_sigma[None, :, None, None] * ps_3d[:, None, :, :]

            t_in = ds_global['temp'].values
            u_in = ds_global['u'].values
            v_in = ds_global['v'].values
            
            n_time, n_orig_lev, n_lat, n_lon = t_in.shape

            # 2. Vectorized Interpolation & Masking
            t_out = np.zeros((n_time, len(P_LEVELS_PA), n_lat, n_lon), dtype=np.float32)
            u_out = np.zeros((n_time, len(P_LEVELS_PA), n_lat, n_lon), dtype=np.float32)
            v_out = np.zeros((n_time, len(P_LEVELS_PA), n_lat, n_lon), dtype=np.float32)

            for t_idx in range(n_time):
                p_vol = p_4d[t_idx]
                
                # Interpolate
                t_interp = vectorized_vertical_interp(t_in[t_idx], p_vol, P_LEVELS_PA)
                u_interp = vectorized_vertical_interp(u_in[t_idx], p_vol, P_LEVELS_PA)
                v_interp = vectorized_vertical_interp(v_in[t_idx], p_vol, P_LEVELS_PA)
                
                # --- UNDERGROUND MASKING ---
                # Create mask where Target Pressure >= Surface Pressure
                # P_LEVELS_PA is (n_levels,)
                # ps_3d[t_idx] is (n_lat, n_lon)
                # Resulting mask is (n_levels, n_lat, n_lon)
                underground_mask = P_LEVELS_PA[:, None, None] >= ps_3d[t_idx][None, :, :]
                
                # Apply NaN
                t_interp[underground_mask] = np.nan
                u_interp[underground_mask] = np.nan
                v_interp[underground_mask] = np.nan
                
                # Store
                t_out[t_idx] = t_interp
                u_out[t_idx] = u_interp
                v_out[t_idx] = v_interp

            # 3. Create Xarray Objects
            plev_coord = xr.DataArray(P_LEVELS_PA, dims=['level'], coords={'level': P_LEVELS_PA}, attrs={'units': 'Pa'})
            coords = {'time': ds_global.time, 'level': plev_coord, 'lat': ds_global.lat, 'lon': ds_global.lon}
            
            t = xr.DataArray(t_out, coords=coords, dims=('time', 'level', 'lat', 'lon'))
            u = xr.DataArray(u_out, coords=coords, dims=('time', 'level', 'lat', 'lon'))
            v = xr.DataArray(v_out, coords=coords, dims=('time', 'level', 'lat', 'lon'))

            # 4. Calculate Zonal Means & Fluxes
            # Note: mean(dim='lon') handles NaNs automatically by ignoring them
            tZonal = t.mean(dim='lon').astype('float32')
            uZonal = u.mean(dim='lon').astype('float32')
            vZonal = v.mean(dim='lon').astype('float32')
            
            u_prime = u - uZonal
            v_prime = v - vZonal 
            t_prime = t - tZonal
            
            ekeZonal = (0.5 * (u_prime**2 + v_prime**2)).mean(dim='lon').astype('float32')
            hozHeatFluxZonal = (v_prime * t_prime).mean(dim='lon').astype('float32')
            hozMomFluxZonal = (u_prime * v_prime).mean(dim='lon').astype('float32')
            
            # hozMomFlux_4D = (u * v).astype('float32')
            # hozHeatFlux_4D = (v * t).astype('float32')

            # Mass stream function
            dp  = xr.DataArray(
                np.gradient(P_LEVELS_PA), dims=['level'], coords={'level': P_LEVELS_PA}
            )
            msf = (
                (2 * np.pi * R_EARTH / G)
                * np.cos(np.deg2rad(vZonal.lat))
                * (vZonal * dp).cumsum('level')
            )

            # Baroclinicity
            thetaZ = tZonal * (P0 / tZonal.level) ** KAPPA
            N2     = (
                (G / thetaZ)
                * thetaZ.differentiate('level')
                * (tZonal.level / (-R_GAS * tZonal / G))
            )
            N      = np.sqrt(np.abs(N2))
            dy     = R_EARTH * np.deg2rad(np.gradient(ds_global.lat))
            baro   = (
                0.31
                * (G / xr.where(N <= 0, 1e-10, N))
                * (thetaZ.differentiate('lat') / dy / thetaZ)
            )

            # 5. Save
            KEEP_TIME_VARS = ['Ls', 'MY', 'integer_sol']
            time_vars = {
                var: ds_global[var] 
                for var in KEEP_TIME_VARS
                if var in ds_global and ds_global[var].dims == ('time',)
            }
            time_vars["integer_sol"] = ds_global["integer_sol"] # add integer_sol explicitly

            derived_ds = xr.Dataset({
                'u': u.astype('float32'), 
                'v': v.astype('float32'),
                't': t.astype('float32'),
                'tZonal': tZonal.astype('float32'), 
                'uZonal': uZonal.astype('float32'),
                'vZonal': vZonal.astype('float32'),
                'ekeZonal': ekeZonal.astype('float32'),

                'hozMomFluxZonal': hozMomFluxZonal.astype('float32'), 
                'hozHeatFluxZonal': hozHeatFluxZonal.astype('float32'),

                'mass_stream_func':   msf.astype('float32'),
                'baroclinicityZonal': baro.astype('float32'),
            })

            derived_ds = derived_ds.assign_coords(lat=ds_global.lat, lon=ds_global.lon, level=plev_coord)
            results = xr.merge([derived_ds, time_vars])
            
            # arithmetic daily mean
            daily = results.groupby("integer_sol").mean("time", skipna=True)
            daily = daily.rename({"integer_sol": "time"})
            daily["integer_sol"] = ("time", daily["time"].values.astype("int32"))

            # Ls: circular daily mean
            ls_rad = np.deg2rad(results["Ls"])
            sin_ls = np.sin(ls_rad).groupby(results["integer_sol"]).mean("time", skipna=True)
            cos_ls = np.cos(ls_rad).groupby(results["integer_sol"]).mean("time", skipna=True)
            ls_daily = (np.rad2deg(np.arctan2(sin_ls, cos_ls)) + 360) % 360
            ls_daily = ls_daily.rename({"integer_sol": "time"})
            daily["Ls"] = ("time", ls_daily.values.astype("float32"))

            # MY: majority vote within each sol
            MY_daily = results["MY"].groupby(results["integer_sol"]).mean("time", skipna=True)
            MY_daily = MY_daily.rename({"integer_sol": "time"})
            daily["MY"] = ("time", np.rint(MY_daily.values).astype("int32"))

            # save single variable files
            for varname in derived_ds.data_vars:
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

        print("Processing complete.")
        return f"Saved {os.path.basename(filepath)}"

    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")
        return f"Failed {os.path.basename(filepath)}"



# Parallel processing of EMARS files using ProcessPoolExecutor
if __name__ == "__main__":

    file_list = sorted(
        glob.glob(os.path.join(data_dir, "openmars_my*.nc"))
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


