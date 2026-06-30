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

IN_DIR="/scratch/bell/hu1029/Mars-BAM_interm/OpenMars_preprocessed_dailyVar"
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

ERA="mro-mcs"

mapfile -t FILES < <(
    ls "${IN_DIR}"/OPENMARS_preprocessed_global_MY??_Ls???_MY??_Ls???_"${VAR}".nc 2>/dev/null | sort
)

if [ ${#FILES[@]} -eq 0 ]; then
    echo "No files found for variable: $VAR"
    exit 1
fi

echo "Sorted files:"
printf '%s\n' "${FILES[@]}"

FIRST=$(basename "${FILES[0]}")
LAST=$(basename "${FILES[-1]}")

START_TAG=$(echo "$FIRST" | sed -E 's/.*_global_(MY[0-9]{2})_Ls([0-9]{3})_MY[0-9]{2}_Ls[0-9]{3}_.*/\1Ls\2/')
END_TAG=$(echo "$LAST"  | sed -E 's/.*_global_MY[0-9]{2}_Ls[0-9]{3}_(MY[0-9]{2})_Ls([0-9]{3})_.*/\1Ls\2/')

OUT_FILE="${OUT_DIR}/OPENMARS_${ERA}_merged_daily_global_${VAR}_${START_TAG}-${END_TAG}.nc"

echo "Era       : $ERA"
echo "First file: $FIRST"
echo "Last file : $LAST"
echo "Output    : $OUT_FILE"

cdo mergetime "${FILES[@]}" "$OUT_FILE"

echo "Done: $OUT_FILE"
