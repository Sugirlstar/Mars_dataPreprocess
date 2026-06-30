#!/bin/bash
#SBATCH -A ccrc
#SBATCH --partition=cpu
#SBATCH --job-name=datapreprocess
#SBATCH --output=./slurm_logs/logdebugresult_%j.out
#SBATCH --error=./slurm_logs/debugerror_%j.err

#SBATCH --array=0-72%12

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

mkdir -p slurm_logs

module load conda

python S1_anomCal_Climatology.py --task-id ${SLURM_ARRAY_TASK_ID}

echo "###########DONE###########"


