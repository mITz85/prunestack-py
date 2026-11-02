#!/usr/bin/env bash
# SLURM array job: one task per dataset, 30 optimization runs each.
#
#   ls -d data/*/ > datasets.txt
#   sbatch --array=1-$(wc -l < datasets.txt) scripts/slurm_array.sh datasets.txt
#
#SBATCH --job-name=prunestack
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/prunestack_%A_%a.out
set -euo pipefail

DATASET=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${1:?usage: sbatch --array=1-N scripts/slurm_array.sh LIST}")
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
prunestack --dataset-path "$DATASET" --seed 1985 --n-runs 30 \
           --n-jobs "${SLURM_CPUS_PER_TASK:-1}" --quiet
