# pylint: disable=line-too-long, import-error, wrong-import-position, broad-exception-caught, too-many-branches
"""
ANIA - Project Main Entry Point

This script serves as the centralized command-line interface (CLI) for executing the ANIA pipeline,
including preprocessing, feature encoding, classical machine learning, and deep learning stages.

Supported stages include:
  - Data collection & cleaning
  - Feature extraction (e.g., iFeature)
  - ML model training & testing
  - Deep learning training & testing (e.g., ANIA)

Usage Example:
  python main.py --stage collect --log_path logs/collect.log
  python main.py --stage train_ml --log_path logs/train_ml.log --model_type random_forest xgboost
  python main.py --stage train_ania --strain "Pseudomonas aeruginosa" --log_path logs/train_ania.log
  python main.py --stage test_ania --strain "Pseudomonas aeruginosa" --log_path logs/test_ania.log
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
    "split": {
        "title": "Data Splitting",
        "import_path": "src.data.split.run_split_pipeline",
    },
    "ifeature": {
        "title": "iFeature Extraction",
        "import_path": "src.features.ifeature_encoding.run_ifeature_pipeline",
    },
    "train_ml": {
        "title": "ML Model Training",
        "import_path": "src.models.machine_learning.train.run_train_ml_pipeline",
    },
    "test_ml": {
        "title": "ML Model Testing",
        "import_path": "src.models.machine_learning.test.run_test_ml_pipeline",
    },
    "cgr": {
        "title": "CGR Encoding",
        "import_path": "src.features.cgr_encoding.run_cgr_pipeline",
    },
    "train_ania": {
        "title": "ANIA Model Training",
        "import_path": "src.models.deep_learning.train.run_train_ania_pipeline",
    },
    "test_ania": {
        "title": "ANIA Model Testing",
        "import_path": "src.models.deep_learning.test.run_test_dl_pipeline",
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

    # Dispatch based on stage type
    if stage == "train_ml":
        stage_func(
            base_path=BASE_PATH,
            logger=logger,
            model_type=args.model_type,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            cv=args.cv,
            loss_function=args.loss_function,
        )
    elif stage == "test_ml":
        stage_func(
            base_path=BASE_PATH,
            logger=logger,
            model_type=args.model_type,
        )
    elif stage == "train_ania":
        stage_func(
            base_path=BASE_PATH,
            strain=args.strain,
            logger=logger,
            train_split=args.train_split,
            patience=args.patience,
            device=args.device,
            random_search=args.random_search,
            num_random_samples=args.num_random_samples,
            model_output_path=args.model_output_path,
        )
    elif stage == "test_ania":
        stage_func(
            base_path=BASE_PATH,
            strain=args.strain,
            logger=logger,
            device=args.device,
            model_input_path=args.model_input_path,
            test_input_file=args.test_input_file,
            prediction_output_path=args.prediction_output_path,
        )
    else:
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
            "ANIA - An Inception-Attention Network for Predicting the Minimum Inhibitory Concentration (MIC) of Antimicrobial Peptides\n\n"
            "Pipeline stages:\n"
            "  collect         Run data collection from AMP databases\n"
            "  clean           Clean and aggregate AMP data\n"
            "  group           Assign MIC group labels to sequences\n"
            "  split           Stratified split into train/test sets\n"
            "  ifeature        Extract AAC, PAAC, CTDD, GAAC features\n"
            "  train_ml        Train classical ML models with GridSearchCV\n"
            "  test_ml         Test trained ML models and evaluate metrics\n"
            "  cgr             Encode sequences using Chaos Game Representation (CGR)\n"
            "  train_ania      Train ANIA deep learning model with GridSearch\n"
            "  test_ania       Test trained ANIA deep learning model\n\n"
            "Examples:\n"
            "  python main.py --stage ifeature --log_path logs/ifeature.log\n"
            "  python main.py --stage train_ml --log_path logs/train_ml.log --model_type ridge xgboost\n"
            "  python main.py --stage train_ania --strain 'Pseudomonas aeruginosa' --log_path logs/train_ania.log\n"
            "  python main.py --stage test_ania --strain 'Pseudomonas aeruginosa' --log_path logs/test_ania.log\n"
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

    # ML-Specific Options
    parser.add_argument(
        "--model_type",
        type=str,
        nargs="+",
        default=None,
        help="Model type(s) to run (e.g., linear, random_forest, xgboost, all).",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of CPU cores to use (-1 = all).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--loss_function",
        type=str,
        default="neg_mean_squared_error",
        help="Loss function for GridSearchCV scoring.",
    )

    # ANIA-Specific Options
    parser.add_argument(
        "--strain",
        type=str,
        default=None,
        help="Strain to train on (e.g., 'Escherichia coli', 'Pseudomonas aeruginosa', 'Staphylococcus aureus').",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Proportion of data to use for training (default is 0.8).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of epochs to wait for improvement in validation loss before early stopping (default is 10).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run the model on (e.g., 'cuda', 'cuda:0', 'cuda:0,1', default is 'cuda:0').",
    )
    parser.add_argument(
        "--random_search",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Whether to perform random search instead of full grid search (default is False).",
    )
    parser.add_argument(
        "--num_random_samples",
        type=int,
        default=50,
        help="Number of random hyperparameter combinations to sample if random_search is True (default is 50).",
    )
    parser.add_argument(
        "--model_output_path",
        type=str,
        default=None,
        help="Path to save the trained model (without the '.pt' extension).",
    )
    parser.add_argument(
        "--model_input_path",
        type=str,
        default=None,
        help="Path to the trained model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--test_input_file",
        type=str,
        default=None,
        help="Path to the test data CSV file for ANIA testing (e.g., data/processed/{suffix}/All_Integrated_aggregated.csv).",
    )
    parser.add_argument(
        "--prediction_output_path",
        type=str,
        default=None,
        help="Path to save the test set prediction results as CSV (e.g., test_predict.csv).",
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
