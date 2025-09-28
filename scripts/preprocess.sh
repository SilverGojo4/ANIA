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
LOG_FILE="$PROJECT_DIR/logs/preprocess/preprocess.log"
INTERIM_DIR="$PROJECT_DIR/data/interim"
AGGREGATED_DIR="$INTERIM_DIR/aggregated"
CDHIT_DIR="$INTERIM_DIR/cdhit"
ZSCORE_DIR="$INTERIM_DIR/zscore"
GROUP_DIR="$INTERIM_DIR/group"
PROCESSED_DIR="$PROJECT_DIR/data/processed"

# Get the base directory of the Conda installation
CONDA_BASE=$(conda info --base)

# Source the Conda initialization script to enable 'conda' commands
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate the Conda environment
conda activate ANIA_Data

# ===================== Step 1: Data Collection =====================
rm -rf $LOG_FILE
rm -rf $INTERIM_DIR
python "$PROJECT_DIR/src/main.py" \
  --stage collect_dbaasp \
  --log_path $LOG_FILE \
  --collect_input_path "$PROJECT_DIR/data/raw/peptides-complete1220.csv" \
  --collect_output_dir $INTERIM_DIR

python "$PROJECT_DIR/src/main.py" \
  --stage collect_dbamp \
  --log_path $LOG_FILE \
  --collect_input_path "$PROJECT_DIR/data/raw/dbAMP3_pepinfo.xlsx" \
  --collect_output_dir $INTERIM_DIR

python "$PROJECT_DIR/src/main.py" \
  --stage collect_dramp \
  --log_path $LOG_FILE \
  --collect_input_path "$PROJECT_DIR/data/raw/DRAMP.xlsx" \
  --collect_output_dir $INTERIM_DIR

python "$PROJECT_DIR/src/main.py" \
  --stage merge_all_sources \
  --log_path $LOG_FILE \
  --collect_output_dir $INTERIM_DIR

# ===================== Step 2: Data Aggregation =====================
AGGREGATE_METHOD="min"
python "$PROJECT_DIR/src/main.py" \
  --stage aggregate \
  --log_path $LOG_FILE \
  --aggregate_input_dir $INTERIM_DIR \
  --aggregate_output_dir $AGGREGATED_DIR \
  --aggregate_method $AGGREGATE_METHOD

# ===================== Step 3: CD-HIT =====================
# IDENTITIES=("1.0" "0.9" "0.8" "0.7" "0.6" "0.5")
# WORD_SIZES=(5 5 5 4 3 2)
IDENTITIES=("0.9")
WORD_SIZES=(5)
CDHIT_MEMORY=32000
CDHIT_THREADS=8
for i in "${!IDENTITIES[@]}"; do
  IDENTITY=${IDENTITIES[$i]}
  WORD_SIZE=${WORD_SIZES[$i]}
  python "$PROJECT_DIR/src/main.py" \
    --stage cd_hit \
    --log_path $LOG_FILE \
    --cdhit_input_dir $AGGREGATED_DIR \
    --cdhit_output_dir $CDHIT_DIR \
    --cdhit_aggregate_method $AGGREGATE_METHOD \
    --cdhit_identity $IDENTITY \
    --cdhit_word_size $WORD_SIZE \
    --cdhit_memory $CDHIT_MEMORY \
    --cdhit_threads $CDHIT_THREADS
done

# ===================== Step 4. Z-score Outlier Filtering =====================
IDENTITY=0.9
ZSCORE_THRESHOLD=3.0
python "$PROJECT_DIR/src/main.py" \
  --stage zscore_filter \
  --log_path $LOG_FILE \
  --zscore_input_dir $CDHIT_DIR \
  --zscore_output_dir $ZSCORE_DIR \
  --zscore_aggregate_method $AGGREGATE_METHOD \
  --zscore_cdhit_identity $IDENTITY \
  --zscore_threshold $ZSCORE_THRESHOLD


# ===================== Step 5. MIC Grouping =====================
python "$PROJECT_DIR/src/main.py" \
  --stage group \
  --log_path "$LOG_FILE" \
  --group_input_dir $ZSCORE_DIR \
  --group_output_dir $GROUP_DIR \
  --group_aggregate_method $AGGREGATE_METHOD \
  --group_cdhit_identity $IDENTITY \
  --group_threshold $ZSCORE_THRESHOLD

# ===================== Step 6: Stratified Data Splitting =====================
TEST_SIZE=0.2
N_BINS=10
RANDOM_STATE=42
N_SPLITS=5
rm -rf $PROCESSED_DIR
python "$PROJECT_DIR/src/main.py" \
  --stage split \
  --log_path "$LOG_FILE" \
  --split_input_dir $GROUP_DIR \
  --split_output_dir $PROCESSED_DIR \
  --split_aggregate_method $AGGREGATE_METHOD \
  --split_cdhit_identity $IDENTITY \
  --split_threshold $ZSCORE_THRESHOLD \
  --split_test_size $TEST_SIZE \
  --split_n_bins $N_BINS \
  --split_random_state $RANDOM_STATE \
  --split_n_splits $N_SPLITS

# Deactivate the Conda environment
conda deactivate
