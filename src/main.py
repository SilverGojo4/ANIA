# pylint: disable=line-too-long, import-error, wrong-import-position, broad-exception-caught, too-many-statements
"""
ANIA - Project Main Entry Point

This script serves as the centralized execution interface for the ANIA pipeline.
"""
# ============================== Standard Library Imports ==============================
import argparse
import importlib
import logging
import os
import sys

# ============================== Project Root Path Setup ==============================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
LOGGING_PATH = os.path.join(BASE_PATH, "src/utils/Logging-Toolkit/src/python")

# Fix Python Path
if BASE_PATH not in sys.path:
    sys.path.append(BASE_PATH)
if LOGGING_PATH not in sys.path:
    sys.path.append(LOGGING_PATH)

# ============================== Project-Specific Imports ==============================
# Logging configuration and custom logger
from setup_logging import setup_logging

# ============================== Stage Configuration ==============================
SUPPORTED_STAGES = {
    "collect_dbaasp": {
        "title": "DBAASP Data Collection",
        "import_path": "src.preprocess.collect.run_collect_dbaasp",
    },
    "collect_dbamp": {
        "title": "dbAMP Data Collection",
        "import_path": "src.preprocess.collect.run_collect_dbamp",
    },
    "collect_dramp": {
        "title": "Dramp Data Collection",
        "import_path": "src.preprocess.collect.run_collect_dramp",
    },
    "merge_all_sources": {
        "title": "Merge All Sources",
        "import_path": "src.preprocess.collect.run_merge_all_sources",
    },
    "aggregate": {
        "title": "Aggregate AMP Datasets",
        "import_path": "src.preprocess.aggregate.run_aggregate_pipeline",
    },
    "cd_hit": {
        "title": "CD-HIT Redundancy Filtering",
        "import_path": "src.preprocess.cd_hit.run_cd_hit_pipeline",
    },
    "zscore_filter": {
        "title": "Z-score Outlier Filtering",
        "import_path": "src.preprocess.zscore_filter.run_zscore_pipeline",
    },
    "group": {
        "title": "MIC Value Grouping",
        "import_path": "src.preprocess.group.run_group_pipeline",
    },
    "split": {
        "title": "Stratified Train/Test Split by Log MIC Value",
        "import_path": "src.preprocess.split.run_split_pipeline",
    },
    "ifeature": {
        "title": "iFeature Feature Encoding",
        "import_path": "src.features.ifeature_encoding.run_ifeature_pipeline",
    },
    "cgr": {
        "title": "Chaos Game Representation (CGR) Encoding",
        "import_path": "src.features.cgr_encoding.run_cgr_pipeline",
    },
}


# ============================== Pipeline Dispatcher ==============================
def dispatch_stage(args: argparse.Namespace) -> None:
    """
    Dispatch execution to the appropriate pipeline stage using lazy import.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments containing stage-specific options.
    """
    # -------------------- Stage Validation --------------------
    # Ensure that the provided stage name is supported.
    stage = args.stage.lower()
    if stage not in SUPPORTED_STAGES:
        available = ", ".join(SUPPORTED_STAGES.keys())
        raise ValueError(f"Unknown stage '{stage}'. Available stages: {available}.")

    # -------------------- Log Path Validation --------------------
    # Ensure that the --log_path argument is provided.
    if not args.log_path:
        raise ValueError("Please specify '--log_path' to enable logging output.")

    # -------------------- Dynamic Stage Import --------------------
    # Perform a lazy import of the selected pipeline stage based on its import path.
    # This avoids loading all modules at startup and improves modularity.
    stage_info = SUPPORTED_STAGES[stage]
    module_path, func_name = stage_info["import_path"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    stage_func = getattr(module, func_name)

    # -------------------- Logger Initialization --------------------
    # Set up the logging environment, loading the logging config and preparing output directory.
    log_config_file = os.path.join(BASE_PATH, "configs/logging.json")
    stage_log_path = os.path.abspath(args.log_path)
    stage_log_dir = os.path.dirname(stage_log_path)
    os.makedirs(name=stage_log_dir, exist_ok=True)
    logger = setup_logging(
        input_config_file=log_config_file,
        logger_name="combo_logger",
        handler_name="file",
        output_log_path=stage_log_path,
    )

    # -------------------- Log Pipeline Metadata --------------------
    logger.info("[ 'Pipeline Initialization Summary' ]")
    logger.log_pipeline_initialization(
        project_name="ANIA",
        line_width=120,
    )
    logger.add_spacer(level=logging.INFO, lines=1)

    # -------------------- Execute Stage Function --------------------
    # Dynamically call the selected stage function, passing all stage-specific arguments.
    extra_args = vars(args)
    extra_args.pop("stage", None)
    extra_args.pop("log_path", None)
    stage_func(base_path=BASE_PATH, logger=logger, **extra_args)
    logger.add_spacer(level=logging.INFO, lines=1)


# ============================== Main Entry ==============================
def main():
    """
    Main CLI entry point for the ANIA pipeline.
    Parses CLI arguments and routes execution to the selected pipeline stages.
    """
    # -------------------- Argument Parser --------------------
    available_stage_lines = [
        f"  - {stage:<15} {info['title']}" for stage, info in SUPPORTED_STAGES.items()
    ]
    available_stages_text = "\n".join(available_stage_lines)
    example_stage = list(SUPPORTED_STAGES.keys())[0]
    example_command = (
        f"  python main.py --stage {example_stage} --log_path logs/{example_stage}.log"
    )
    parser = argparse.ArgumentParser(
        description=(
            "ANIA - An Inception-Attention Network for Predicting the Minimum Inhibitory Concentration (MIC) of Antimicrobial Peptides\n\n"
            "Available stages:\n"
            f"{available_stages_text}\n\n"
            "Example:\n"
            f"{example_command}\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # -------------------- General Options --------------------
    parser.add_argument(
        "--stage",
        type=str,
        choices=list(SUPPORTED_STAGES.keys()),
        required=True,
        help="Pipeline stage to run. Choose one of: "
        + ", ".join(SUPPORTED_STAGES.keys()),
    )
    parser.add_argument(
        "--log_path",
        type=str,
        required=True,
        help="Path for log file output.",
    )

    # -------------------- Collect Parameters --------------------
    parser.add_argument(
        "--collect_input_path",
        type=str,
        required=False,
        help=(
            "Path to the input file for the selected collection stage. "
            "If not provided, each stage will use its default raw data path "
            "(e.g., peptides-complete1220.csv for DBAASP, dbAMP3_pepinfo.xlsx for dbAMP, DRAMP.xlsx for DRAMP)."
        ),
    )
    parser.add_argument(
        "--collect_output_dir",
        type=str,
        required=False,
        help=(
            "Directory to save the processed strain-specific outputs "
            "(e.g., EC_DBAASP.csv, PA_dbAMP.csv, SA_Dramp.csv). "
            "Defaults to '<base_path>/data/interim/'."
        ),
    )

    # -------------------- Aggregate Parameters --------------------
    parser.add_argument(
        "--aggregate_input_dir",
        type=str,
        required=False,
        help="Directory containing merged strain-specific input files (e.g., *_All.csv). Defaults to '<base_path>/data/interim/'.",
    )
    parser.add_argument(
        "--aggregate_output_dir",
        type=str,
        required=False,
        help="Directory to save aggregated outputs (e.g., *_agg_min.csv). Defaults to '<base_path>/data/processed/aggregated/'.",
    )
    parser.add_argument(
        "--aggregate_method",
        type=str,
        required=False,
        default="min",
        choices=["min", "mean", "median"],
        help="Aggregation method for Log MIC Value. Default: 'min'.",
    )

    # -------------------- CD-HIT Parameters --------------------
    parser.add_argument(
        "--cdhit_input_dir",
        type=str,
        required=False,
        help=(
            "Directory containing group-level FASTA files to be filtered with CD-HIT "
            "(default: '<base_path>/data/processed/group')."
        ),
    )
    parser.add_argument(
        "--cdhit_output_dir",
        type=str,
        required=False,
        help=(
            "Directory to save CD-HIT filtered FASTA files "
            "(default: '<base_path>/data/processed/cdhit')."
        ),
    )
    parser.add_argument(
        "--cdhit_aggregate_method",
        type=str,
        required=False,
        default="min",
        choices=["min", "mean", "median"],
        help="Aggregation method for Log MIC Value. Default: 'min'.",
    )
    parser.add_argument(
        "--cdhit_identity",
        type=float,
        required=False,
        default=0.9,
        help=(
            "Sequence identity threshold for CD-HIT clustering (default: 0.9). "
            "Example: 0.9 = 90% identity."
        ),
    )
    parser.add_argument(
        "--cdhit_word_size",
        type=int,
        required=False,
        default=5,
        help="Word size (k-mer) used by CD-HIT (default: 5). Must match identity threshold.",
    )
    parser.add_argument(
        "--cdhit_memory",
        type=int,
        required=False,
        default=16000,
        help="Memory limit (MB) for CD-HIT (default: 16000 MB).",
    )
    parser.add_argument(
        "--cdhit_threads",
        type=int,
        required=False,
        default=4,
        help="Number of CPU threads to use for CD-HIT (default: 4).",
    )

    # -------------------- Z-Score Filtering Parameters --------------------
    parser.add_argument(
        "--zscore_input_dir",
        type=str,
        required=False,
        help=(
            "Directory containing CD-HIT filtered CSV files "
            "(default: '<base_path>/data/interim/cdhit')."
        ),
    )
    parser.add_argument(
        "--zscore_output_dir",
        type=str,
        required=False,
        help=(
            "Directory to save Z-score filtered CSV and FASTA outputs "
            "(default: '<base_path>/data/interim/zscore')."
        ),
    )
    parser.add_argument(
        "--zscore_aggregate_method",
        type=str,
        required=False,
        default="min",
        choices=["min", "mean", "median"],
        help="Aggregation method label to match file names (default: 'min').",
    )
    parser.add_argument(
        "--zscore_cdhit_identity",
        type=float,
        required=False,
        default=0.9,
        help="CD-HIT identity label to match file names (default: 0.9).",
    )
    parser.add_argument(
        "--zscore_threshold",
        type=float,
        required=False,
        default=3.0,
        help="Absolute Z-score threshold for filtering outliers (default: 3.0).",
    )

    # -------------------- Group Parameters --------------------
    parser.add_argument(
        "--group_input_dir",
        type=str,
        required=False,
        help=(
            "Directory containing Z-score filtered CSV files "
            "(default: '<base_path>/data/interim/zscore')."
        ),
    )
    parser.add_argument(
        "--group_output_dir",
        type=str,
        required=False,
        help=(
            "Directory to save grouped CSV and FASTA outputs "
            "(default: '<base_path>/data/interim/group')."
        ),
    )
    parser.add_argument(
        "--group_aggregate_method",
        type=str,
        required=False,
        default="min",
        help="Aggregation method used in previous steps (default: 'min').",
    )
    parser.add_argument(
        "--group_cdhit_identity",
        type=float,
        required=False,
        default=0.9,
        help="CD-HIT identity threshold used in previous step (default: 0.9).",
    )
    parser.add_argument(
        "--group_threshold",
        type=float,
        required=False,
        default=3.0,
        help="Z-score threshold used in previous step (default: 3.0).",
    )

    # -------------------- Split Parameters --------------------
    parser.add_argument(
        "--split_input_dir",
        type=str,
        required=False,
        help=(
            "Directory containing grouped CSV files "
            "(default: '<base_path>/data/interim/group')."
        ),
    )
    parser.add_argument(
        "--split_output_dir",
        type=str,
        required=False,
        help=(
            "Directory to save train/test split CSV outputs "
            "(default: '<base_path>/data/processed')."
        ),
    )
    parser.add_argument(
        "--split_aggregate_method",
        type=str,
        required=False,
        default="min",
        help="Aggregation method used in previous steps (default: 'min').",
    )
    parser.add_argument(
        "--split_cdhit_identity",
        type=float,
        required=False,
        default=0.9,
        help="CD-HIT identity threshold used in previous step (default: 0.9).",
    )
    parser.add_argument(
        "--split_threshold",
        type=float,
        required=False,
        default=3.0,
        help="Z-score threshold used in previous step (default: 3.0).",
    )
    parser.add_argument(
        "--split_test_size",
        type=float,
        required=False,
        default=0.2,
        help="Proportion of dataset to reserve for test split (default: 0.2).",
    )
    parser.add_argument(
        "--split_n_bins",
        type=int,
        required=False,
        default=10,
        help="Number of bins used for Log MIC stratification (default: 10).",
    )
    parser.add_argument(
        "--split_random_state",
        type=int,
        required=False,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--split_n_splits",
        type=int,
        required=False,
        default=5,
        help="Number of folds for Stratified K-Fold splitting within training set (default: 5).",
    )

    # -------------------- iFeature Parameters --------------------
    parser.add_argument(
        "--ifeature_input_dir",
        type=str,
        required=False,
        help=(
            "Directory containing train/test FASTA files for each strain "
            "(default: '<base_path>/data/processed/')."
        ),
    )
    parser.add_argument(
        "--ifeature_output_dir",
        type=str,
        required=False,
        help=(
            "Directory to save iFeature-encoded outputs "
            "(default: '<base_path>/data/processed/ifeature')."
        ),
    )
    parser.add_argument(
        "--ifeature_config_path",
        type=str,
        required=False,
        help=(
            "Path to iFeature configuration YAML file "
            "(default: '<base_path>/configs/ifeature_config.yml')."
        ),
    )
    parser.add_argument(
        "--ifeature_n_splits",
        type=int,
        required=False,
        default=5,
        help="Number of K-Fold splits (if applicable, default: 5).",
    )

    # -------------------- CGR Parameters --------------------
    parser.add_argument(
        "--cgr_input_dir",
        type=str,
        required=False,
        help=(
            "Directory containing train/test FASTA files for each strain "
            "(default: '<base_path>/data/processed/')."
        ),
    )
    parser.add_argument(
        "--cgr_output_dir",
        type=str,
        required=False,
        help=(
            "Directory to save CGR-encoded feature outputs "
            "(default: '<base_path>/data/processed/cgr')."
        ),
    )
    parser.add_argument(
        "--cgr_aaindex_path",
        type=str,
        required=False,
        help=(
            "Path to AAindex property CSV file "
            "(default: '<base_path>/configs/AAindex_properties.csv')."
        ),
    )
    parser.add_argument(
        "--cgr_n_splits",
        type=int,
        required=False,
        default=5,
        help="Number of K-Fold splits (default: 5).",
    )
    parser.add_argument(
        "--cgr_resolution",
        type=int,
        required=False,
        default=16,
        help="Resolution of CGR grid (e.g., 8, 16, 32). Default: 16.",
    )
    parser.add_argument(
        "--cgr_kmer_k",
        type=int,
        required=False,
        default=3,
        help="Length of k-mer used for AAindex property mapping (default: 3).",
    )

    args = parser.parse_args()

    # -------------------- Stage Execution --------------------
    try:
        dispatch_stage(args)
        print("Pipeline execution completed successfully.")
        sys.exit(0)

    except Exception as e:
        print(f"[Pipeline Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
