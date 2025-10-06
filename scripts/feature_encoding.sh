#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Check input arguments
if [ $# -lt 1 ]; then
  echo "Usage: $0 <PROJECT_DIR>"
  exit 1
fi

# Set paths
PROJECT_DIR=$1
LOG_FILE="$PROJECT_DIR/logs/preprocess/feature_encoding.log"
PROCESSED_DIR="$PROJECT_DIR/data/processed"
CONFIGS_DIR="$PROJECT_DIR/configs"

# Get the base directory of the Conda installation
CONDA_BASE=$(conda info --base)

# Source the Conda initialization script to enable 'conda' commands
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate the Conda environment
conda activate ANIA_Data

# ===================== Step 1: iFeature Encoding =====================
rm -rf $LOG_FILE
find data/processed/ -type d -name "ifeature" -exec rm -rf {} +
N_SPLITS=5
python "$PROJECT_DIR/src/main.py" \
  --stage ifeature \
  --log_path $LOG_FILE \
  --ifeature_input_dir $PROCESSED_DIR \
  --ifeature_output_dir $PROCESSED_DIR \
  --ifeature_config_path "$CONFIGS_DIR/ifeature_config.yml" \
  --ifeature_n_splits $N_SPLITS

# ===================== Step 2: CGR Encoding =====================
find data/processed/ -type d -name "cgr" -exec rm -rf {} +
RESOLUTIONS=(8 16 32)
KMER=3
for RESOLUTION in "${RESOLUTIONS[@]}"; do
    python "$PROJECT_DIR/src/main.py" \
        --stage cgr \
        --log_path "$LOG_FILE" \
        --cgr_input_dir "$PROCESSED_DIR" \
        --cgr_output_dir "$PROCESSED_DIR" \
        --cgr_aaindex_path "$CONFIGS_DIR/AAindex_properties.csv" \
        --cgr_n_splits "$N_SPLITS" \
        --cgr_resolution "$RESOLUTION" \
        --cgr_kmer_k "$KMER"
done

# Deactivate the Conda environment
conda deactivate
