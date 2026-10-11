#!/usr/bin/env bash
# Allocation reuse only; no sbatch/scancel and no new GPU reservation.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ "${SLURM_JOB_ID:-}" != 447101 || "$(hostname -s)" != ilab1 ]]; then
  echo 'Must execute as a step of Joseph allocation 447101 on ilab1' >&2
  exit 2
fi
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(); assert "A4500" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)'
export DATA="${DATA:-$PWD/gpu_jobs/T-319/data/team_slices.npz}"
export DEVICE=cuda
export STEPS="${STEPS:-2000}"
bash gpu_jobs/T-319/reproduce.sh real "${OUT:-$PWD/eval/results/T-319}"
