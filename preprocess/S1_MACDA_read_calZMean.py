#!/usr/bin/env python
# ==============================================================================
# SCRIPT original from: MACDA_prep.py
# PURPOSE: Advanced Pre-process EMARS data (Global).
#          1. Calculates true 4D Pressure using EMARS hybrid coefficients.
#          2. Interpolates to fixed levels using calculated 4D Pressure.
#          3. MASKS underground data.
#          4. Calculates Zonal Means and Diagnostics.
#
# ==============================================================================

"""
MACDA v2 Northern Hemisphere (NH) Atmospheric Prep Pipeline
===========================================================
Dataset: Mars Analysis Correction Data Assimilation (MACDA) v2.0
Domain: Northern Hemisphere ($0^\circ N - 90^\circ N$)
Target Grid: 30 Log-spaced Pressure Levels ($750$ Pa to $0.05$ Pa)

Objective:
----------
This script serves as the primary ingestion engine for standardizing MACDA v2 
output. It transforms model-level sigma ($\sigma$) coordinates into a fixed 
pressure-coordinate framework, isolates transient eddy signals via high-pass 
filtering, and derives the fundamental energetic diagnostics required for 
Martian Annular Mode (SAM/BAM) analysis.

Scientific Context:
-------------------
MACDA v2 provides a continuous record of the Martian atmosphere by assimilating 
Thermal Emission Spectrometer (TES) data from the Mars Global Surveyor (MGS). 
This script extracts:
1. Zonal Mean State ($[u], [v], [T]$): The primary Martian winter circulation.
2. Eddy Kinetic Energy ($EKE$): Quantifying the vigor of synoptic-scale waves:
   $$EKE = \frac{1}{2}([u'^2] + [v'^2])$$
3. Eddy Fluxes ($[v'T'], [u'v']$): Identifying the transport of heat and 
   momentum sustaining the baroclinic cycle.
4. Mass Stream Function ($\psi$): Visualizing the meridional overturning 
   circulation via vertical integration of the meridional wind:
   $$\psi = \frac{2\pi a \cos \phi}{g} \int_{0}^{p} [v] dp$$



Technical Implementation:
-------------------------
1. Vectorized Vertical Interpolation: Uses `xr.apply_ufunc` to perform 
   $ln(p)$ interpolation from model sigma levels to a fixed log-pressure grid. 
   This ensures accuracy in the thin, high-altitude Martian middle atmosphere.
2. Topographic Masking: Implements a strict "Underground Mask" where 
   $P_{target} \ge P_{surface}$. This prevents topographic artifacts from 
   contaminating near-surface heat flux signals.
3. Temporal Filtering: Employs a 4th-order high-pass Butterworth filter to 
   isolate transients (synoptic weather) from the seasonal and diurnal cycles.
4. Memory Management: Utilizes Dask parallelism and explicit garbage collection 
   (`gc.collect()`) to handle the memory-intensive 4D transformation of MACDA 
   history files.

Mars Physical Constants ($CO_2$ Atmosphere):
--------------------------------------------
- Gravity ($g$): $3.711 \text{ m/s}^2$
- Gas Constant ($R_{gas}$): $188.92 \text{ J/kg K}$
- Specific Heat ($C_p$): $846 \text{ J/kg K}$
- Planetary Radius ($a$): $3,389.5 \text{ km}$
"""
import sys

import numpy as np
import os
import xarray as xr
import gc
import glob


# --- Physical Constants (Mars) ---
G        = 3.711
R_GAS    = 188.92
CP       = 846
KAPPA    = R_GAS / CP
P0       = 100000.0
R_EARTH  = 3.3895e6
R_MARS = 3.3895e6

# --- CONFIGURATION ---
data_dir = '/depot/wanglei/data/MACDA_v2/'
output_dir = '/scratch/bell/hu1029/Mars-BAM_interm/MACDA_preprocessed_dailyVar/'
os.makedirs(output_dir, exist_ok=True)

# Target Pressure Grid
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

# def apply_highpass(data, fs=12.0):
#     b, a = signal.butter(4, 1/(2*fs), btype='highpass')
#     def _filter_1d(x):
#         if np.all(x == 0) or np.any(np.isnan(x)): return x
#         return signal.filtfilt(b, a, x, axis=-1)
#     return xr.apply_ufunc(_filter_1d, data, input_core_dims=[['time']],
#                          output_core_dims=[['time']], vectorize=True)

def process_one_file(filepath):

    base_name = os.path.basename(filepath)

    if base_name.startswith("mgs-tes"):
        era = "mgs-tes"
    elif base_name.startswith("mro-mcs"):
        era = "mro-mcs"
    else:
        raise ValueError(f"Unknown MACDA era: {base_name}")

    suffix = (
        base_name
        .replace("mgs-tes-reanalysis_mars_", "")
        .replace("mro-mcs-reanalysis_mars_", "")
        .replace("_v2-0", "")
    ) # keep the MY and SOY

    out_name = f"MACDA_{era}_processed_global_{suffix}"
    output_path = os.path.join(output_dir, out_name)

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
    
    print(f"Now Processing {base_name}")


    with xr.open_dataset(filepath, decode_times=False) as ds_global:

        # add a dimension, i.e., integer_sol, representing the integer part of the sol (Martian day) for getting daily mean
        ds_global = ds_global.assign_coords(
            integer_sol=(
                "time",
                np.ceil(ds_global["time"].values - 1e-6).astype(np.int32) - 1
            )
        )

        # Check PS units
        ps_raw = ds_global['psurf'].values
        if np.nanmean(ps_raw) > 10000: ds_global['psurf'] /= 100.0
        elif np.nanmean(ps_raw) < 100: ds_global['psurf'] *= 100.0
        
        # Calculate 4D Pressure
        lev_sigma = ds_global['lev'].values 
        ps_3d = ds_global['psurf'].values      
        p_4d = lev_sigma[None, :, None, None] * ps_3d[:, None, :, :]

        t_in = ds_global['temp'].values
        u_in = ds_global['uwind'].values
        v_in = ds_global['vwind'].values

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
        KEEP_TIME_VARS = ['Ls', 'MY_Ls', 'integer_sol']
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
        MY_daily = results["MY_Ls"].groupby(results["integer_sol"]).mean("time", skipna=True)
        MY_daily = MY_daily.rename({"integer_sol": "time"})
        daily["MY_Ls"] = ("time", np.rint(MY_daily.values).astype("int32"))

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


# Parallel processing of EMARS files using ProcessPoolExecutor
if __name__ == "__main__":

    file_list = sorted(
        glob.glob(os.path.join(data_dir, "mgs-tes-reanalysis_mars_*.nc"))
        +
        glob.glob(os.path.join(data_dir, "mro-mcs-reanalysis_mars_*.nc"))
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
