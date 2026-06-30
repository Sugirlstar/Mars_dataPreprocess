
Author: Yanjun Hu
Last updated: June 26, 2026
Python version: 3.12.11
Environment: So far, the workflow relies primarily on widely used scientific Python packages (`numpy`, `scipy`, `xarray`, `matplotlib`), so no dedicated environment is provided at this point.

---
# Issues (Keep Updated)

> [!todo]+ To be solved:
> 1. The original workflow combines the TES and MCS eras into a single continuous time series. These periods should first be treated as separate samples, and their statistical properties compared before deciding whether they can be merged into a single record.
> 2. Our MACDA v2 data now is only till MY31, but it is available to MY35. Should we update it to MY35) ? Data Source: https://catalogue.ceda.ac.uk/uuid/cd037a9ea387438fabf4d674dbe53088/
> 3. The MACDA dataset contains gaps due to missing files in the official archive  (https://data.ceda.ac.uk/badc/mgs/data/macda/v2-0). The missing periods appear to be available in the corresponding `ody-themis` set and could be used to fill the gaps. A statistical consistency check should be performed before merging these data with the surrounding record. If substantial differences are detected, it may be preferable to leave the gaps unfilled.

> [!success]- Solved
> 4. Data source? Found:
> 	- MACDA v2 now is till MY31, need to update (from MY 24 through MY 35) ? Data Source: https://catalogue.ceda.ac.uk/uuid/cd037a9ea387438fabf4d674dbe53088/
> 	- OpenMARS: The workflow downloads OpenMARS from Figshare article 24573205 rather than directly from the original Holmes et al. (2020) repository described in the publication. The downloaded files appear to extend beyond MY32 (up to MY36), indicating that a later extended release may have been used. The provenance and version history of this extension should be verified. Data source found: https://figshare.com/articles/dataset/OpenMARS_continuous_MY28-35_standard_database/24573205
> 	- EMARS: Data Souce found - https://www.datacommons.psu.edu/download/meteorology/greybush/emars-1p0/a_landing_page.html, and paper that can be cited: https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/gdj3.
> 5. EMARS v1.0 has anal_mean, back_mean. Currently using back_mean. Which one should be used? ==back_mean==
> 6. 60 levels for EMARS, 30 levels for OpenMARS, (750 to 0.05), why? ==Make them all 30==
> 7. OpenMARS only has NH while EMARS has global, why? (==NH for now==)
> 8. **Major issue 1**: The data-merging procedure relies on filename sorting, but the Ls values in the OpenMARS filenames are not zero-padded to three digits. As a result, lexicographic sorting does not follow the correct chronological order and will lead to incorrectly merged time series. (only for OpenMars) ==Solved in the new workflow==
> 9. **Major issue 2**: EMARS data is hourly, but in the code it's considered as 6-hourly. ==Solved in the new workflow==
> 10. OpenMars and MACDA: 2-hourly, but it starts from 2AM, ends at 24. How to deal with this? ==Just take 2-24 as a day==
> 11. The MACDA preprocessing code performs interpolation linearly in pressure coordinates, whereas the EMARS and OpenMARS preprocessing codes perform interpolation linearly in log-pressure coordinates. ==has changed to align with the EMARS and OpenMARS==. 

---
# Dataset Info

## Raw data directory: 

1. **OpenMARS** (/depot/wanglei/data/OpenMARS Data/OpenMARS MY28-36) (MY27-35)
	- Contains: .nc files, be like: `openmars_my31_ls98_my31_ls111.nc` (~400Mb each)
 2. **MACDA v2** (/depot/wanglei/data/MACDA_v2/) (MY24-31)
	- Contains: .nc files, be like: `mgs-tes-reanalysis_mars_MY24SOY211_MY24SOY241_v2-0.nc` (SOY means sol of year; ~800Mb each)
 3. **EMARS** (/depot/wanglei/data/EMARS (real) Data/) (MY24-33)
	- Contains: four groups of .nc files, including "_anal_mean_", "_back_mean_", "_back_memb" and "anal_sprd_". "back_mean" is used. be like: `emars_v1.0_back_mean_MY29_Ls150-180.nc` (~5GB each)
## **Summary** (==Need check==)

| Dataset  | Time Coverage             | Horizontal Resolution          | Vertical Levels                             | Vertical Coordinate                                    | Temporal Resolution |
| -------- | ------------------------- | ------------------------------ | ------------------------------------------- | ------------------------------------------------------ | ------------------- |
| OpenMARS | MY27–MY35                 | 2° × 2°<br>(lat decrease)      | ~50 levels (actually 35 levels in data)     | Sigma (σ = p/ps)                                       | 2 hourly            |
| MACDA v2 | MY24–MY37 (we use: -MY31) | 5° × 5°<br>(lat decrease)      | ~25 levels (?)                              | Sigma (σ = p/ps)                                       | 2 hourly            |
| EMARS    | MY24–MY33                 | 5° × 6°<br>(latitude increase) | 28 levels (actually 28 full levels in data) | Hybrid sigma-pressure (distributed on pressure levels) | hourly              |

> [!note]+ Note
>Mars reanalysis datasets span two major observational eras. These two instruments differ substantially in observing geometry, vertical coverage, and retrieval methodology:
>1. TES era (MY24–MY27), based on observations from the Thermal Emission Spectrometer (TES) aboard Mars Global Surveyor (MGS). 
>2. MCS era (MY28 onward), based on observations from the Mars Climate Sounder (MCS) aboard Mars Reconnaissance Orbiter (MRO). 
> As a result, the TES and MCS portions of the reanalysis record are not directly equivalent. Changes in the underlying observing system can introduce artificial discontinuities into climatological and variability analyses. Therefore, when studying long-term atmospheric variability, it is often useful to analyze the two eras separately and verify that the results are robust across both periods. 
> 
> For the three dataset: 
> - EMARS: In the official document, states that it explicitly separates TES and MCS eras (TES: MY24 Ls103 – MY27 Ls102; MCS: MY28 Ls112 – MY33 Ls105). A gap exists between the two eras in the dataset.
> - MACDA: Data product explicitly separates `mgs-tes` and `mro-mcs` in the raw datasets. (TES: MY24SOY211 – MY27SOY185. MCS: MY28SOY237 – end of record.)
> -  OpenMARS: Effectively an MCS-era reanalysis product. Entire record is treated as `mro-mcs` here. Earliest files extend slightly into late MY27 (e.g., MY27 Ls358 – MY28 Ls013).


---
# Code (./YH_Mars_3Datasets)

Below is a detailed description of the code repository. 
The workflow is mainly organized into two directories: `./preprocess` and `./analysis`. It is recommended to create a separate `./plotting` directory to store plotting notebooks/scripts for figure-generation.
The `S1_`, `S2_`, etc. prefixes in the script names indicate the recommended execution order. Scripts sharing the same stage number do not depend on one another and can be run independently or in parallel.
A seperate `./upstream` folder contains original code shared by Dr. Battalio (not included in the public version of this repository).

> [!Note] Note
> 1. All references to `MACDA` in this repository correspond to MACDA v2. 
> 2. All intermediate files are stored under `/scratch/bell/hu1029/Mars-BAM_interm`. To work on your side, simply replace `/scratch/bell/hu1029` with your own directory throughout the scripts.
> 3. During development, some output filenames were revised after the corresponding jobs had already been completed. In a few cases, the generated files were manually renamed and the scripts were subsequently updated to reflect the new naming convention, but the jobs were not rerun. As a result, historical log files or job outputs may reference filenames that differ slightly from the files currently present in the repository. If the scripts are executed again from the current version of the code, the generated filenames should be consistent with the files included here.
## Data preprocess (./preprocess)

Sub-daily data from all three datasets are preprocessed (and zonal-mean fields calculated) in this step and converted into a uniform daily format.
### 1. `S1_{dataset}_read_calZMean.py` (submitted via `jobsubmit.slurm`)

- When calculating daily mean: The previous function simply groups every {ops=12} consecutive time steps into one daily mean, assuming that the input data are already correctly ordered and continuous in time. Here we calculate daily mean based on the data grouped by sol.
- Output var list: `u, v, t, uZonal, vZonal, tZonal, ekeZonal, hozMomFluxZonal, hozHeatFluxZonal, mass_stream_func, baroclinicityZonal`. The outputs are the same among all three datasets, as well as the naming convention: `./Mars-BAM_interm/{dataset}_preprocessed_dailyVar/{DATASET}_preprocessed_global_{timeidentifier}_{var}.nc`.
- EMARS differs from MACDA v2.0 and OpenMARS in two key ways:
	1) EMARS is provided on hybrid sigma–pressure levels, whereas MACDA v2.0 and OpenMARS are provided on terrain-following sigma levels (pressure normalized by the local surface pressure). But they are all transferred to 30 fixed pressure levels from 750 Pa to 0.05 Pa via log-pressure interpolation.
	2) The raw file organization is different. Hourly EMARS files are divided by solar longitude (Ls, 0–360° within each Mars Year, and sol is provided as integer), making the daily averaging straightforward by grouping records with the same sol. In contrast, 2-hrouly MACDA v2.0 and OpenMARS files are organized by sol (decimals). For these datasets, the sol values are first converted to an integer day identifier using `ceil(sol) - 1`. Records are then grouped by this identifier to compute daily means. The daily Ls is calculated using a circular mean, while the Mars Year is assigned based on the majority vote among records within each group.
- The preprocessing jobs are submitted as a SLURM job array, allowing all files to be processed in parallel (`#SBATCH --array=0-140%12`). As a result, this step takes only about 10 minutes per dataset.
### 2. `S2_{dataset}_merge.sh`

This script is a SLURM batch job and can be submitted directly using `sbatch`
Its purpose is to merge all preprocessed EMARS files for each variable into continuous time-series products using `cdo mergetime`. CDO is installed by default on the cluster.

For each variable, the script produces three outputs:
1. **TES-era record** (`mgs-tes`, `MY24–MY27`) (OpenMARS doesn't have it)
2. **MCS-era record** (`mro-mcs`, `MY28+`)
3. **Combined record with observational gap** (`combinedwithGap`): Concatenates the TES-era and MCS-era merged products into a single dataset while preserving the observational gap between the two periods. (OpenMARS doesn't have it)

The script uses a SLURM job array to process all variables in parallel and automatically generates output filenames containing the corresponding MY/Ls time range.

The outputs' naming convention: `./Mars-BAM_interm/{DATASET}_preprocessed_dailyVar/Merged/{DATASET}_combinedwithGap_merged_daily_global_{var}_{timeidentifier}.nc`

### 3. `sanCheck.ipynb` and `test.ipynb`

- sanCheck.ipynb: To check whether there are any gaps within the dataset. Gaps identified in the MACDA (missing files in the source archive).
- test.ipynb: Simply read the .nc files and review the data structure.

## Analysis (./analysis)

### 1. `S1_anomCal_Climatology` (must be submitted via `S1_jobsubmit_anomCal_Climatology.slurm` )

This script calculates the daily climatology and daily anomalies from the merged daily zonal-mean files produced by `./preprocess/S2_{dataset}_merge.sh`.
For each dataset, era, and variable, the script reads the corresponding merged file, masks the prescribed global dust storm periods, calculates the climatology and anomalies. The 4-D variables `t`, `u`, and `v` are not processed in this step; only zonal-mean variables and derived diagnostics are included.

The outputs are saved as NetCDF files in: `/scratch/bell/hu1029/Mars-BAM_interm/{DATASET}_analysis_ClimAnom/`

Parallel computing is used to process each dataset–era–variable combination independently.

Note that multiple methods for calculating climatology and anomalies are implemented in this script, corresponding to the functions `get_mars_climatology_anomalies()`, `fft_anom()`, and `calc_Anom()`. At present, the simplest approach, `get_mars_climatology_anomalies()`, is used, in which the climatology is calculated from integer-degree Ls bins and anomalies are obtained by subtracting the corresponding climatological mean. The other two methods have not yet been tested within the current workflow.

For additional discussion of anomaly definitions in Martian reanalysis datasets, see the section _“Anomalies defined from the seasonal average”_ in Battalio and Lora (2021) (https://www.nature.com/articles/s41550-021-01447-4) .

### 2. `S1_plot_WinterClim.ipynb`

Plot the annual mean and winter mean of uZonal and ekeZonal to do the sanity check. The MACDA annual mean uZonal is consistent with Fig 1b contours in Battalio and Lora (2021) (https://www.nature.com/articles/s41550-021-01447-4). The winter time mean of uZonal also resembles Fig1 in this paper: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023JE008137

### 3. `S2_test_EOFanalysis.ipynb` (==Need check==)

This script is currently a prototype used to test the EOF-based identification of the Martian annular modes. At present, it is applied only to the MACDA `mgs-tes` period and is intended as a sanity check against Figures 1 and 2 of Battalio and Lora (2021) before extending the workflow to all datasets and eras.

The script reads zonal-mean zonal wind (`uZonal`) and eddy kinetic energy (`ekeZonal`) anomalies, removes global dust storm periods, selects the winter season, applies latitude–pressure weighting, detrends the data, and performs EOF analysis using the `eofs` package.

The primary objective is to reproduce:
- EOF1 of zonal-mean zonal wind anomalies (`uZonal_anom`), corresponding to the Northern Annular Mode (NAM).
- EOF1 of eddy kinetic energy anomalies (`ekeZonal_anom`), corresponding to the Baroclinic Annular Mode (BAM).

The EOF patterns and explained variance fractions obtained so far appear broadly consistent with Battalio and Lora (2021), suggesting that the preprocessing, weighting, and EOF calculations are behaving reasonably. However, the regression analyses performed using the resulting principal components do not yet reproduce the published regression structures. The source of this discrepancy remains under investigation.

Please note that various versions of the EOF-analysis pipeline exist across the original repository and Dr. Battalio’s notebooks.
