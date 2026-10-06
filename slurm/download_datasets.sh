#!/bin/bash
#SBATCH --job-name=download_datasets
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/download_datasets/%j.out
#SBATCH --error=logs/download_datasets/%j.err

set -euo pipefail

# Raw data lives in scratch (/lustre/tmp), not the project fileset (only ~16GB free there).
# Treat this as a read-only cache for downstream processing jobs.
DATA_DIR=/lustre/tmp/slurm/$SLURM_JOB_ID/data/raw
mkdir -p "$DATA_DIR"

# Bodegas Terras Gauda UAV RGB videos (Zenodo 7330951, ~21.7 GB)
BTG_DIR="$DATA_DIR/bodegas_terras_gauda"
mkdir -p "$BTG_DIR"
wget -c -O "$BTG_DIR/7330951.zip" "https://zenodo.org/api/records/7330951/files-archive"
unzip -o "$BTG_DIR/7330951.zip" -d "$BTG_DIR"
rm "$BTG_DIR/7330951.zip"

# VineLiDAR point clouds (Zenodo 8113105, ~2.8 GB)
VL_DIR="$DATA_DIR/vinelidar"
mkdir -p "$VL_DIR"
wget -c -O "$VL_DIR/8113105.zip" "https://zenodo.org/api/records/8113105/files-archive"
unzip -o "$VL_DIR/8113105.zip" -d "$VL_DIR"
rm "$VL_DIR/8113105.zip"

echo "Done. Raw data in $DATA_DIR"
