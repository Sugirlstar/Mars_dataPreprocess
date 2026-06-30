#!/bin/bash
#SBATCH -A ccrc
#SBATCH --partition=cpu
#SBATCH --job-name=datapreprocess
#SBATCH --output=debugresult_%j.out
#SBATCH --error=debugerror_%j.err
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

IN_DIR="/scratch/bell/hu1029/Mars-BAM_interm/EMARS_preprocessed_dailyVar"
OUT_DIR="${IN_DIR}/Merged"
mkdir -p "$OUT_DIR"

# variable list to be merged
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
    PATTERN="$2"

    mapfile -t FILES < <(ls ${PATTERN} 2>/dev/null | sort)

    if [ ${#FILES[@]} -eq 0 ]; then
        echo "No files found for era=${ERA}, variable=${VAR}"
        return 1
    fi

    FIRST=$(basename "${FILES[0]}")
    LAST=$(basename "${FILES[-1]}")

    START_TAG=$(echo "$FIRST" | sed -E 's/.*_MY([0-9]{2})_Ls([0-9]{3})-.*/MY\1Ls\2/')
    END_TAG=$(echo "$LAST"  | sed -E 's/.*_MY([0-9]{2})_Ls[0-9]{3}-([0-9]{3})_.*/MY\1Ls\2/')

    OUT_FILE="${OUT_DIR}/EMARS_${ERA}_merged_daily_global_${VAR}_${START_TAG}-${END_TAG}.nc"

    echo "Era       : $ERA"
    echo "First file: $FIRST"
    echo "Last file : $LAST"
    echo "Output    : $OUT_FILE"

    cdo mergetime "${FILES[@]}" "$OUT_FILE"

    echo "Done: $OUT_FILE"
}

# 1. TES era: MY24–MY27
merge_era "mgs-tes" "${IN_DIR}/EMARS_preprocessed_global_MY2[4-7]_Ls???-???_${VAR}.nc"

# 2. MCS era: MY28 onward
merge_era "mro-mcs" "${IN_DIR}/EMARS_preprocessed_global_MY2[8-9]_Ls???-???_${VAR}.nc ${IN_DIR}/EMARS_preprocessed_global_MY3[0-9]_Ls???-???_${VAR}.nc"

# 3. Combined record with observational gap
MGS_FILE=$(ls "${OUT_DIR}/EMARS_mgs-tes_merged_daily_global_${VAR}"_MY*Ls*-MY*Ls*.nc)
MRO_FILE=$(ls "${OUT_DIR}/EMARS_mro-mcs_merged_daily_global_${VAR}"_MY*Ls*-MY*Ls*.nc)

MGS_BASE=$(basename "$MGS_FILE")
MRO_BASE=$(basename "$MRO_FILE")

START_TAG_combined=$(echo "$MGS_BASE" | sed -E 's/.*_(MY[0-9]{2}Ls[0-9]+)-MY[0-9]{2}Ls[0-9]+\.nc/\1/')
END_TAG_combined=$(echo "$MRO_BASE"  | sed -E 's/.*_MY[0-9]{2}Ls[0-9]+-(MY[0-9]{2}Ls[0-9]+)\.nc/\1/')

OUT_FILE="${OUT_DIR}/EMARS_combinedwithGap_merged_daily_global_${VAR}_${START_TAG_combined}-${END_TAG_combined}.nc"

echo "Merging combined record with gap"
echo "Input 1: $MGS_FILE"
echo "Input 2: $MRO_FILE"
echo "Output : $OUT_FILE"

cdo mergetime "$MGS_FILE" "$MRO_FILE" "$OUT_FILE"

echo "Done: $OUT_FILE"
