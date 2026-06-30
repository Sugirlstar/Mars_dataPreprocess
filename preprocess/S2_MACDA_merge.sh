#!/bin/bash
#SBATCH -A ccrc
#SBATCH --partition=cpu
#SBATCH --job-name=datapreprocess
#SBATCH --output=./slurm_logs/debugresult_%j.out
#SBATCH --error=./slurm_logs/debugerror_%j.err
#SBATCH --array=0-11%11
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=06:00:00

set -euo pipefail

module load gcc/11.1.0
module load openmpi/4.1.6
module load cdo

IN_DIR="/scratch/bell/hu1029/Mars-BAM_interm/MACDA_preprocessed_dailyVar"
OUT_DIR="${IN_DIR}/Merged"
mkdir -p "$OUT_DIR"

VARS=(
  t
  u
  v
  tZonal
  uZonal
  vZonal
  ekeZonal
  hozMomFluxZonal
  hozHeatFluxZonal
  mass_stream_func
  baroclinicityZonal
)

VAR="${VARS[$SLURM_ARRAY_TASK_ID]}"

echo "Merging variable: $VAR"

merge_era () {
    ERA="$1"

    mapfile -t FILES < <(
        ls "${IN_DIR}"/MACDA_"${ERA}"_processed_global_MY??SOY???_MY??SOY???_"${VAR}".nc 2>/dev/null | sort
    )

    if [ ${#FILES[@]} -eq 0 ]; then
        echo "No files found for era=${ERA}, variable=${VAR}"
        return 1
    fi

    FIRST=$(basename "${FILES[0]}")
    LAST=$(basename "${FILES[-1]}")

    START_TAG=$(echo "$FIRST" | sed -E 's/.*_global_(MY[0-9]{2}SOY[0-9]{3})_MY[0-9]{2}SOY[0-9]{3}_.*/\1/')
    END_TAG=$(echo "$LAST"  | sed -E 's/.*_global_MY[0-9]{2}SOY[0-9]{3}_(MY[0-9]{2}SOY[0-9]{3})_.*/\1/')

    OUT_FILE="${OUT_DIR}/MACDA_${ERA}_merged_daily_global_${VAR}_${START_TAG}-${END_TAG}.nc"

    echo "Era       : $ERA"
    echo "First file: $FIRST"
    echo "Last file : $LAST"
    echo "Output    : $OUT_FILE"

    cdo mergetime "${FILES[@]}" "$OUT_FILE"

    echo "Done: $OUT_FILE"
}

# 1. Merge each era separately: mgs_tes is before mro_mcs
merge_era "mgs-tes" 
merge_era "mro-mcs"

# 2. Merge the two era-merged files together
MGS_FILE=$(ls "${OUT_DIR}/MACDA_mgs-tes_merged_daily_global_${VAR}"_MY*SOY*-MY*SOY*.nc)
MRO_FILE=$(ls "${OUT_DIR}/MACDA_mro-mcs_merged_daily_global_${VAR}"_MY*SOY*-MY*SOY*.nc)
MGS_BASE=$(basename "$MGS_FILE")
MRO_BASE=$(basename "$MRO_FILE")
START_TAG_combinedera=$(echo "$MGS_BASE" | sed -E 's/.*_(MY[0-9]{2}SOY[0-9]{3})-MY[0-9]{2}SOY[0-9]{3}\.nc/\1/')
END_TAG_combinedera=$(echo "$MRO_BASE"  | sed -E 's/.*_MY[0-9]{2}SOY[0-9]{3}-(MY[0-9]{2}SOY[0-9]{3})\.nc/\1/')

OUT_FILE="${OUT_DIR}/MACDA_combinedwithGap_merged_daily_global_${VAR}_${START_TAG_combinedera}-${END_TAG_combinedera}.nc"

echo "Merging combined eras"
echo "Output: $OUT_FILE"

cdo mergetime "$MGS_FILE" "$MRO_FILE" "$OUT_FILE"

echo "Done: $OUT_FILE"
