#!/bin/bash
#SBATCH -A ccrc
#SBATCH --partition=cpu
#SBATCH --job-name=datapreprocess
#SBATCH --output=./slurm_logs/logdebugresult_%j.out
#SBATCH --error=./slurm_logs/debugerror_%j.err

#SBATCH --array=0-180%12

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00


module load conda

python S1_OpenMARS_read_calZMean.py

echo "###########DONE###########"


