#!/bin/bash
# Single-experiment launcher for SLURM.
# Called by SLURM_script.sh as the --script argument.
#
# Usage (via SLURM_script.sh):
#   bash slurm/SLURM_script.sh \
#       --script slurm/run_experiment.sh \
#       --params "exp.seed=42 model.hidden_dim=256" \
#       --time 04:00:00 --mem 32GB --gpu 1
#
# Usage (direct, for local testing):
#   MODEL=default DATASET=default \
#   WANDB_MODE=offline \
#   bash slurm/run_experiment.sh "exp.seed=42"
#
# Environment variables:
#   MODEL     - selects configs/model/<MODEL>.yaml  (default: "default")
#   DATASET   - selects configs/dataset/<DATASET>.yaml (default: "default")

set -e

PARAMS="${@}"

MODEL="${MODEL:-default}"
DATASET="${DATASET:-default}"
WANDB_MODE="${WANDB_MODE:-online}"

echo "MODEL=$MODEL"
echo "DATASET=$DATASET"
echo "WANDB_MODE=$WANDB_MODE"
echo "Hydra overrides: $PARAMS"

MODEL=$MODEL DATASET=$DATASET WANDB_MODE=$WANDB_MODE \
    uv run python src/main.py $PARAMS
