# This is to calculate the daily anomalies and climatology. 
# The inputs are the merged daily data (outputs of ./preprocess/S2_dataset_merge.py).
# The outputs are .nc files of daily anomalies and climatology.

import sys
import os, sys
import numpy as np
import scipy as sp
import xarray as xr
import itertools
import argparse
import glob

# region PARAMETERS
OUTPUT_DIR = "/scratch/bell/hu1029/Mars-BAM_interm"
# HEMI = "NH"

eraNames = ["mgs-tes", "mro-mcs", "combinedwithGap"]
vars_to_process = [
    "tZonal", "uZonal", "vZonal",
    "ekeZonal",
    "hozMomFluxZonal",
    "hozHeatFluxZonal",
    "mass_stream_func",
    "baroclinicityZonal",
] # t, u, v are 4-dimensional (time, level, lat, lon), not processed here

datasets = {
    "EMARS": {
        "input_dir": "/scratch/bell/hu1029/Mars-BAM_interm/EMARS_preprocessed_dailyVar/Merged",
        "solname": "emars_sol",
        "MYname": "MY",
        "GDS_ranges": [[25, 170, 300],  [28, 260, 325], [34, 144, 360]]
        
    },
    "MACDA": {
        "input_dir": "/scratch/bell/hu1029/Mars-BAM_interm/MACDA_preprocessed_dailyVar/Merged",
        "solname": "integer_sol",
        "MYname": "MY_Ls",
        "GDS_ranges": [[25, 180, 250],  [28, 260, 310]]
    },
    "OPENMARS": {
        "input_dir": "/scratch/bell/hu1029/Mars-BAM_interm/OpenMars_preprocessed_dailyVar/Merged",
        "solname": "integer_sol",
        "MYname": "MY",
        "GDS_ranges": [[25, 170, 300],  [28, 260, 325]]
    },
}

# endregion

# region FUNCTIONS
# pick the method to calculate anomalies
def get_mars_climatology_anomalies(data, Ls_array):
    """Calculates anomalies by removing the mean for each integer degree of Ls."""
    print("   -> Computing Climatology (Ls binning) and subtracting...")

    Ls_array = np.asarray(Ls_array, dtype=float)
    valid_ls = np.isfinite(Ls_array)          # NaN Ls steps arise when make_daily

    # Build integer bin array; use -1 as a sentinel for invalid (NaN Ls) steps
    Ls_int = np.full(len(Ls_array), -1, dtype=int)
    Ls_int[valid_ls] = (np.round(Ls_array[valid_ls]) % 360).astype(int)

    climatology = np.full((360,) + data.shape[1:], np.nan)
    for d in range(360):
        indices = np.where(Ls_int == d)[0]
        if len(indices) > 0:
            climatology[d] = np.nanmean(data[indices], axis=0)

    anomalies = np.full_like(data, np.nan)
    for t in range(len(Ls_array)):
        if Ls_int[t] < 0:      
            continue
        anomalies[t] = data[t] - climatology[Ls_int[t]]

    return anomalies, climatology

def fft_anom(varIn, cutoff, ops, order=5,pad=True):
    nyq = 0.5 * ops
    normal_cutoff = cutoff / nyq
    if isinstance(cutoff, float):
        sos = sp.signal.butter(order, normal_cutoff, btype='lowpass', analog=False, output='sos')
    else:
        sos = sp.signal.butter(order, normal_cutoff, btype='bandpass', analog=False, output='sos')

    if varIn.ndim>1:
        opDim=np.argmax(np.shape(varIn))
        varIn=np.rollaxis(varIn,opDim)
        
        if pad:
            varBar = sp.signal.sosfiltfilt(sos, varIn,axis=0,padlen=np.max(np.int64(1/cutoff)*3*ops))
        else:
            varBar = sp.signal.sosfiltfilt(sos, varIn,axis=0)
            
        for _ in range(varIn.ndim-opDim+1):
            varBar=np.rollaxis(varBar,opDim)
            varIn=np.rollaxis(varIn,opDim)
    else:
        if pad:
            varBar = sp.signal.sosfiltfilt(sos, varIn,axis=0,padlen=np.max(np.int64(1/cutoff)*3*ops))
        else:
            varBar = sp.signal.sosfiltfilt(sos, varIn,axis=0)
        
    return varIn-varBar,varBar

# Apply circular Gaussian smoothing to the climatology along the DOY/Ls axis.
# The kernel width is controlled by 'conv' (approximately conv+1 points,
# with a Gaussian standard deviation of conv/5).
def calc_Anom(varIn, period=None, doy=None, numYr=None, ops=None, idxClim=None, conv=None, z_max=None, median=False):
    """
    Compute climatology and anomalies. Handles both regularly-spaced data
    (via period) and irregularly-spaced data with an explicit day-of-year
    array (via doy). Time is assumed to be axis 0 on entry (decorator handles this).

    Parameters
    ----------
    varIn   : ndarray
        Input data, time on axis 0.
    period  : int, optional
        Steps per year for regularly-spaced data (e.g. 12 for monthly, 669
        for Mars daily, 365 for Earth daily, 59900//41 for sub-daily Earth data). 
        If provided, doy is constructed automatically.
    doy     : array-like, optional
        Explicit day-of-year array (1-based) for irregularly-spaced data.
        One of period or doy must be provided.
    numYr   : int, optional
        Number of years. Only used with period; inferred if not provided.
    ops     : int, optional
        If provided, also return data averaged to daily resolution by
        averaging every ops steps. Only used with period.
    idxClim : array-like, optional
        Indices into the time axis to use when building the climatology
        (e.g. to exclude dust storm years). All time steps used if None.
    conv    : int, optional
        Gaussian smoothing half-window applied to climatology across the
        doy axis (wrapping) to reduce susceptible to outliers. Skipped if None.
    z_max   : float, optional
        Z-score threshold for outlier removal when building climatology.
        Set to 0 or None to skip. Default 5.
    median  : bool, optional
        If True, use median instead of mean for climatology. Default False.

    Returns
    -------
    varAnom : ndarray
        Anomalies, same shape as varIn (trimmed to numYr*period if period used).
    climVar : ndarray
        Climatology, shape (n_doy, ...).
    errVar  : ndarray
        Standard error of the climatology, shape (n_doy, ...).
    varDaily : ndarray or None
        Daily-averaged data. Only returned if ops is provided.
    varAnomDaily : ndarray or None
        Daily-averaged anomalies. Only returned if ops is provided.
    """

    # internal function ---------------------------------------
    def rollavg_1Dconvolve(varin,n,wrap=False,square=False):
        'np.convolve, with edge handling for timeseries'
        'varin is the vector to smooth, n is the number of points over which to do so'
        'use wrap=True if the domain is circular (like over a latitude circle)'
        'use square=True a boxcar filter is used instead of guassian'
        'returns the rolling convolve and the deviation from said convolve'
        
        sx=n/5
        x=np.linspace(1,n,n+1)
        Xyc=np.ceil(n/2)
        h= np.exp(-((x - Xyc)**2. / (2. * sx**2.)))
        h=h/np.sum(h)
        if square:h=np.ones(n,dtype='float')
        
        nDim=np.ndim(varin)
        
        if nDim==1:
            if wrap:
                varinA = np.append(varin,varin[:int(np.floor(n/2))],axis=0)
                varinA = np.append(varin[-int(np.ceil(n/2)):],varinA,axis=0)
                #conv with 'valid' only returns full overlap
                conv=np.convolve(varinA,h, 'valid')/np.convolve(np.ones(len(varinA)),h, 'valid') 
            else:
                # conv with 'same' returns same length as a
                conv=np.convolve(varin,h, 'same')/np.convolve(np.ones(len(varin)),h, 'same') 

        else:
            opDim=np.argmax(np.shape(varin))
            varin=np.rollaxis(varin,opDim)
            shp=np.shape(varin)
            varin=np.reshape(varin,(shp[0],-1))
            conv=np.zeros_like(varin)*np.nan
            kern=np.convolve(np.ones(np.shape(varin)[0]),h, 'same')
            
            for i in range(0,np.shape(varin)[1]):
                #mask = (np.isfinite(varin[:,i,j]))
                #if (~mask).all():  #if there are no points in the regression, skip
                #    continue
                conv[:,i]=np.convolve(varin[:,i],h, 'same')/kern

            varin=np.reshape(varin,(shp))
            conv=np.reshape(conv,(shp))
        
            for _ in range(varin.ndim-opDim+1):
                conv=np.rollaxis(conv,opDim)
                varin=np.rollaxis(varin,opDim)

        return conv,varin-conv
    # --------------------------------------------------------

    varIn = np.asarray(varIn, dtype=float)
    spatial_shape = varIn.shape[1:]   # () for 1D, (lat,) for 2D, (lat,lon) for 3D

    # --- Build DOY array from period if not provided explicitly ---
    if doy is None and period is not None:
        nT    = varIn.shape[0]
        numYr = numYr or (nT // period)
        nUse  = numYr * period
        varIn = varIn[:nUse]                          # trim to exact multiple
        doy   = np.tile(np.arange(1, period + 1), numYr)   # 1-based, repeating
    elif doy is not None:
        doy  = np.asarray(doy, dtype=int)
        nUse = len(doy)
        varIn = varIn[:nUse]
    else:
        raise ValueError("One of period or doy must be provided.")

    # --- Climatology index: default to all time steps ---
    if idxClim is None:
        idxClim = np.arange(nUse)
    idxClim = np.asarray(idxClim, dtype=int)

    doyClim  = doy[idxClim]                          # doy values used for climatology
    n_doy    = int(np.nanmax(doyClim))               # number of doy bins

    # --- Allocate outputs ---
    climVar = np.full((n_doy,)    + spatial_shape, np.nan)
    errVar  = np.full((n_doy,)    + spatial_shape, np.nan)
    varAnom = np.full(varIn.shape,                  np.nan)

    # --- Build climatology from only values indicated by idxClim ---
    for ii in range(np.min(np.unique(doyClim)) - 1, n_doy):
        idx = np.flatnonzero(doyClim == ii + 1)     # time indices for this doy
        if len(idx) == 0:
            continue
        slc = varIn[idxClim][idx]                   # shape (n_matches, ...) or (n_matches,)

        # if z_max:
        #     slc = zscoreVar(slc, axis=0, z_max=z_max) # didn't find the zscoreVar function in the notebook, so I commented it out for now. It seems to be a function that removes outliers based on z-score.

        climVar[ii] = np.nanmedian(slc, axis=0) if median else np.nanmean(slc, axis=0)
        errVar[ii]  = sp.stats.sem(slc, axis=0, nan_policy='omit')

    # --- Optional smoothing of climatology ---
    if conv is not None:
        if varIn.ndim == 1:
            climVar = rollavg_1Dconvolve(climVar, conv, wrap=True, square=False)[0]
            errVar  = rollavg_1Dconvolve(errVar,  conv, wrap=True, square=False)[0]
        else:
            # Tile, smooth along doy axis for each spatial/level point, untile
            # Uses moveaxis so it works for any number of spatial dimensions
            tLen    = n_doy
            climVar = np.concatenate([climVar]*3, axis=0)   # tile along doy
            errVar  = np.concatenate([errVar]*3,  axis=0)
            for idx in np.ndindex(spatial_shape):
                climVar[(slice(None),)+idx] = rollavg_1Dconvolve(
                    climVar[(slice(None),)+idx], conv, wrap=False, square=False)[0]
                errVar[(slice(None),)+idx]  = rollavg_1Dconvolve(
                    errVar[(slice(None),)+idx],  conv, wrap=False, square=False)[0]
            climVar = climVar[tLen:-tLen]             # untile
            errVar  = errVar[tLen:-tLen]

    # --- Compute anomalies ---
    for ii in range(np.min(np.unique(doyClim)) - 1, n_doy):
        idx = np.flatnonzero(doy == ii + 1)
        if len(idx) == 0:
            continue
        varAnom[idx] = varIn[idx] - climVar[ii]

    # --- Catch leap days (doy not in climatology) ---
    # Find time steps where anomaly is still NaN despite valid input data
    nan_mask = np.isnan(varAnom)
    if varIn.ndim > 1:
        nan_mask = nan_mask.any(axis=tuple(range(1, varIn.ndim)))
    leap_idx = np.flatnonzero(nan_mask & np.isfinite(varIn).any(
        axis=tuple(range(1, varIn.ndim))) if varIn.ndim > 1 
        else nan_mask & np.isfinite(varIn))
    for ii in leap_idx:
        d = doy[ii] - 1                              # 0-based index into climVar
        varAnom[ii] = varIn[ii] - np.nanmean(
            climVar[max(0, d-1):d+2], axis=0)        # average neighboring doys

    # # --- Optional daily averaging (period mode only) ---
    # if ops is not None and period is not None:
    #     varDaily     = make_daily(varIn[:nUse], ops=ops, pad=False)[0]
    #     varAnomDaily = make_daily(varAnom[:nUse], ops=ops, pad=False)[0]
    #     #varDaily     = varIn[:nUse].reshape((nUse // ops, ops) + spatial_shape).mean(axis=1).squeeze()
    #     return varAnom, climVar, errVar, varDaily, varAnomDaily

    return varAnom, climVar, errVar


# region MAIN PROCESS

if __name__ == "__main__":

    # 1. build input arguments ------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    args = parser.parse_args()

    # All task combinations
    taskCombine = list(itertools.product(
        datasets.keys(),
        eraNames,
        vars_to_process,
    ))

    if args.task_id >= len(taskCombine):
        print(f"Task {args.task_id}: no work assigned. Exiting.")
        sys.exit(0)

    dataset_name, era_name, var_name = taskCombine[args.task_id]
    dataset = datasets[dataset_name]

    print("=" * 80)
    print(f"Task ID      : {args.task_id}/{len(taskCombine)-1}")
    print(f"Dataset      : {dataset_name}")
    print(f"Era          : {era_name}")
    print(f"Variable     : {var_name}")
    print("=" * 80)

    input_dir = dataset["input_dir"]
    solname   = dataset["solname"]
    MYname    = dataset["MYname"]
    GDS = dataset["GDS_ranges"]  # List of GDS ranges for the dataset

    # 2. build input/output file paths ------------------------------------------
    input_file = glob.glob(os.path.join(input_dir,f"{dataset_name}_{era_name}_merged_daily_global_{var_name}_*.nc"))[0]

    output_dir = os.path.join(OUTPUT_DIR, f"{dataset_name}_analysis_ClimAnom")
    os.makedirs(output_dir, exist_ok=True)

    anom_output = os.path.join(output_dir, f"{dataset_name}_{era_name}_{var_name}_anom.nc")
    clim_output = os.path.join(output_dir, f"{dataset_name}_{era_name}_{var_name}_clim.nc")

    # if os.path.exists(clim_output):
    #     print(f"Output already exists. Skipping: {clim_output}", flush=True)
    #     sys.exit(0)

    # 3. read data ------------------------------------------
    ds = xr.open_dataset(input_file, decode_times=False)
    data = ds[var_name].values
    ds_lat = ds['lat'].values 
    # don't change the lat order, to keep the _anom align with the orignial Zonal data
    # # check if lat is in ascending order, if not, reverse (just to make sure the order is consistent from this point)
    # if ds_lat[0] > ds_lat[-1]:
    #     ds_lat = ds_lat[::-1]
    #     data = data[:, :, ::-1]
    # if lon exists, read it

    if 'lon' in ds.coords:
        ds_lon = ds['lon'].values

    # don't pick the hemisphere at this step, to keep it consistent with the original Zonal data
    # # pick the target hemisphere
    # if HEMI == "NH":
    #     lat_mask = ds_lat >= 0
    # elif HEMI == "SH":
    #     lat_mask = ds_lat <= 0
    # data = data[:, :, lat_mask]    
    # lat_hemi = ds_lat[lat_mask]

    Ls = ds['Ls'].values # mars' position in its orbit, in degrees, float
    sol = ds[solname].values # not SOY, but a continuous mannually asigned time stamp (e.g., 2000), integer
    my = ds[MYname].values # Mars year, integer

    # 4. mask out the global dust storm periods ------------------------------------------
    for r in GDS:
        mask_idx = np.where((Ls >= r[1]) & (Ls <= r[2]) & (my == r[0]))[0]
        if len(mask_idx) > 0:
            data[mask_idx] = np.nan

    # 5. calculate climatology and anomaly ------------------------------------------------------------------
    
    var_anom, var_clim = get_mars_climatology_anomalies(data, Ls)
    # var_anom.shape   = (time, level, lat)
    # var_clim.shape = (360, level, lat)

    # if using fft_anom:
    #     anom.shape   = (time, level, lat)
    #     clim.shape = (time, level, lat)
    # if using calc_Anom:
    #     varAnom.shape = (time, level, lat)
    #     climVar.shape = (n_doy, level, lat)
    #     errVar.shape  = (n_doy, level, lat)

    # 6. save output ------------------------------------------------------------------
    anom_encoding = {f"{var_name}_anom": {"zlib": True, "complevel": 4}}
    clim_encoding = {f"{var_name}_clim": {"zlib": True, "complevel": 4}}
    
    out_anom = xr.Dataset()
    out_anom[f"{var_name}_anom"] = xr.DataArray(
        var_anom.astype(np.float32),
        dims=("time", "level", "lat"),
        coords={
            "time": ds.time.values,
            "level": ds.level.values,
            "lat": ds_lat,
            "Ls": ("time", Ls),
            solname: ("time", sol),
            MYname: ("time", my),
        },
    )
    tmp_output = anom_output + ".tmp"
    out_anom.to_netcdf(tmp_output, encoding=anom_encoding)
    os.replace(tmp_output, anom_output)

    out_clim = xr.Dataset()
    out_clim[f"{var_name}_clim"] = xr.DataArray(
        var_clim.astype(np.float32),
        dims=("Ls_bin", "level", "lat"),
        coords={
            "Ls_bin": np.arange(360),
            "level": ds.level.values,
            "lat": ds_lat,
        },
    )
    tmp_output = clim_output + ".tmp"
    out_clim.to_netcdf(tmp_output, encoding=clim_encoding)
    os.replace(tmp_output, clim_output)

    print('Done.')

# endregion
