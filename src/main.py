# pylint: disable=line-too-long, import-error, wrong-import-position, broad-exception-caught, too-many-branches
"""
ANIA - Project Main Entry Point

This script serves as the centralized command-line interface (CLI) for executing the ANIA pipeline,
including preprocessing, feature encoding, classical machine learning, and deep learning stages.

Supported stages include:
  - Data collection & cleaning

Each stage is modular and can be executed independently or in combination via command-line flags.

Usage Example:
  python main.py --stage collect
"""
# ============================== Standard Library Imports ==============================
import argparse
import importlib
import logging
import os
import sys

# ============================== Project Root Path Setup ==============================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
LOGGING_PATH = os.path.join(BASE_PATH, "src/utils/logging_toolkit/src/python")

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
    "collect": {
        "title": "Data Collecting",
        "import_path": "src.data.collect.run_collect_pipeline",
    },
    "clean": {
        "title": "Data Cleaning",
        "import_path": "src.data.clean.run_clean_pipeline",
    },
    "group": {
        "title": "Data Grouping",
        "import_path": "src.data.group.run_group_pipeline",
    },
    "spilt": {
        "title": "Data Splitting",
        "import_path": "src.data.split.run_split_pipeline",
    },
    "ifeature": {
        "title": "iFeature Extraction",
        "import_path": "src.features.ifeature_encoding.run_ifeature_pipeline",
    },
}


# ============================== Pipeline Dispatcher ==============================
def dispatch_stage(stage: str, args) -> None:
    """
    Dispatch execution to the appropriate pipeline stage using lazy import.

    This function dynamically imports and executes the pipeline stage function
    based on user input.

    Parameters
    ----------
    stage : str
        The pipeline stage to execute.
    args : argparse.Namespace
        Parsed command-line arguments containing stage-specific options.
    """
    # Validate stage
    if stage not in SUPPORTED_STAGES:
        available = ", ".join(SUPPORTED_STAGES.keys())
        raise ValueError(f"Unknown stage '{stage}'. Available stages: {available}.")

    # Validate logger
    if not args.log_path:
        raise ValueError("Please specify '--log_path'.")

    # Lazy load stage function
    stage_info = SUPPORTED_STAGES[stage]
    module_path, func_name = stage_info["import_path"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    stage_func = getattr(module, func_name)

    # Logger initialization
    log_config_file = os.path.join(BASE_PATH, "configs/general_logging.json")
    stage_log_path = os.path.abspath(args.log_path)
    os.makedirs(os.path.dirname(stage_log_path), exist_ok=True)
    logger = setup_logging(
        input_config_file=log_config_file,
        output_log_path=stage_log_path,
        logger_name=f"general_logger",
        handler_name="general",
    )
    logger.log_title(f"Running Stage: '{stage_info['title']}'", level=logging.INFO)

    # General stage
    stage_func(base_path=BASE_PATH, logger=logger)


# ============================== Main Entry ==============================
def main():
    """
    Main CLI entry point for the AMP-MIC pipeline.
    Parses CLI arguments and routes execution to the selected pipeline stages.
    """
    # -------------------- Argument Parser --------------------
    parser = argparse.ArgumentParser(
        description=(
            "ANIA - An Inception-Attention Network for Predicting the Minimum Inhibitory Concentration (MIC) of Antimicrobial Peptides\n"
            "\n"
            "Pipeline stages:\n"
            "  collect         Run data collection from external AMP databases\n"
            "  clean           Clean and aggregate data for downstream modeling\n"
            "  group           Assign MIC group labels to processed datasets\n"
            "  split           Split the processed dataset into train/test sets\n"
            "  ifeature        Extract AAC, PAAC, CTDD, GAAC features using iFeature\n"
            "\n"
            "Examples:\n"
            "  python main.py --stage collect --log_path logs/collect.log\n"
            "  python main.py --stage clean --log_path logs/clean.log\n"
            "  python main.py --stage group --log_path logs/group.log\n"
            "  python main.py --stage split --log_path logs/split.log\n"
            "  python main.py --stage ifeature --log_path logs/ifeature.log\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--stage",
        type=str,
        choices=list(SUPPORTED_STAGES.keys()),
        required=True,
        help="Pipeline stage(s) to run. Choose one or more of: "
        + ", ".join(SUPPORTED_STAGES.keys()),
    )

    parser.add_argument(
        "--log_path",
        type=str,
        default=None,
        help="Optional custom log file path.",
    )

    args = parser.parse_args()

    # -------------------- Stage Execution --------------------
    try:
        # Normalize stage key
        stage_key = args.stage.lower()

        # Dispatch selected stage function
        dispatch_stage(stage=stage_key, args=args)

        # Final success message
        print("Pipeline execution completed successfully.")
        sys.exit(0)

    except Exception as e:
        print(f"[Pipeline Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
