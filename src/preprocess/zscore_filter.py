# pylint: disable=import-error, wrong-import-position, too-many-locals
"""
Z-score Outlier Filtering Pipeline
"""
# ============================== Standard Library Imports ==============================
import logging
import os
import sys
import time

# ============================== Third-Party Library Imports ==============================
from scipy.stats import zscore

# ============================== Project Root Path Setup ==============================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
LOGGING_PATH = os.path.join(BASE_PATH, "src/utils/Logging-Toolkit/src/python")

# Fix Python Path
if BASE_PATH not in sys.path:
    sys.path.append(BASE_PATH)
if LOGGING_PATH not in sys.path:
    sys.path.append(LOGGING_PATH)

# ============================== Project-Specific Imports ==============================
# Submodule imports (external tools integrated into project)
from setup_logging import CustomLogger

# Local ANIA utility modules
from utils.io_utils import (
    directory_exists,
    file_exists,
    load_dataframe_by_columns,
    write_fasta_file,
)
from utils.log_utils import get_pipeline_completion_message, get_task_completion_message


# ============================== Custom Functions ==============================
def apply_zscore_filter(
    input_csv: str,
    output_dir: str,
    logger: CustomLogger,
    target_column: str = "Log MIC Value",
    threshold: float = 3.0,
) -> None:
    """
    Apply Z-score filtering to remove outliers beyond a specified threshold.

    Parameters
    ----------
    input_csv : str
        Path to the input CSV file.
    output_dir : str
        Directory to save filtered CSV and FASTA outputs.
    logger : CustomLogger
        Logger instance for structured logging.
    target_column : str, optional
        Column name used to compute Z-scores (default = "Log MIC Value").
    threshold : float, optional
        Absolute Z-score threshold for filtering (default = 3.0).

    Returns
    -------
    None
    """
    logger.info(f"/ Task: Apply Z-score filtering on '{os.path.basename(input_csv)}'")
    logger.add_divider(level=logging.INFO, length=120, border="-", fill="-")

    # Start timing
    start_time = time.time()

    try:
        # Load data for the target strain
        df = load_dataframe_by_columns(file_path=input_csv)

        # Compute Z-score
        df = df.copy()
        original_count = len(df)
        df["Zscore"] = zscore(df[target_column])
        df_filtered = df[df["Zscore"].abs() <= threshold].copy()
        df_filtered.drop(columns=["Zscore"], inplace=True)
        filtered_count = len(df_filtered)
        removed_count = original_count - filtered_count
        removed_ratio = (
            (removed_count / original_count * 100) if original_count > 0 else 0.0
        )

        # Log summary
        logger.log_with_borders(
            level=logging.INFO,
            message=(
                f"[ Z-score Filtering Summary ]\n"
                f"▸ Original records : {original_count}\n"
                f"▸ Removed outliers (|Z| > {threshold}) : {removed_count} ({removed_ratio:.2f}%)\n"
                f"▸ Retained records : {filtered_count}\n"
            ),
            border="|",
            length=120,
        )
        logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

        # Save filtered result
        input_filename = os.path.basename(input_csv)
        file_stem = os.path.splitext(input_filename)[0]  # remove .csv safely
        threshold_label = str(threshold).replace(".0", "")
        output_csv = os.path.join(output_dir, f"{file_stem}_z{threshold_label}.csv")
        output_fasta = os.path.join(output_dir, f"{file_stem}_z{threshold_label}.fasta")
        df_filtered.to_csv(output_csv, index=False)
        write_fasta_file(input_path=output_csv, output_fasta=output_fasta)
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Saved:\n'{output_csv}'\n'{output_fasta}'",
            border="|",
            length=120,
        )
        logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

        # Summary
        logger.log_with_borders(
            level=logging.INFO,
            message="\n".join(get_task_completion_message(start_time)),
            border="|",
            length=120,
        )
        logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

    except Exception:
        logger.exception(msg="Unexpected error in 'apply_zscore_filter()'.")
        raise


# ============================== Pipeline Entry Point ==============================
def run_zscore_pipeline(base_path: str, logger: CustomLogger, **kwargs) -> None:
    """
    Executes Z-score filtering on all strain-level CD-HIT filtered datasets.

    Parameters
    ----------
    base_path : str
        Project root path (used to build file paths).
    logger : CustomLogger
        Logger instance for structured logging.
    **kwargs : dict
        Optional overrides for input/output file paths.

    Returns
    -------
    None
    """
    # Start timing
    start_time = time.time()

    try:
        # -------------------- Retrieve input parameters (CLI or default) --------------------
        input_dir = kwargs.get(
            "zscore_input_dir", os.path.join(base_path, "data/interim/cdhit")
        )
        output_dir = kwargs.get(
            "zscore_output_dir", os.path.join(base_path, "data/interim/zscore")
        )
        agg_method = kwargs.get("zscore_aggregate_method", "min").lower()
        cdhit_identity = kwargs.get("zscore_cdhit_identity", 0.9)
        threshold = kwargs.get("zscore_threshold", 3.0)

        # Ensure directories exist
        for directory in [input_dir, output_dir]:
            if not directory_exists(dir_path=directory):
                os.makedirs(name=directory)

        # Mapping of full strain names to their corresponding suffixes
        strains = {
            "Escherichia coli": "EC",
            "Pseudomonas aeruginosa": "PA",
            "Staphylococcus aureus": "SA",
        }

        # Process each strain
        for _, (_, suffix) in enumerate(strains.items(), start=1):

            # Check input files exist
            identity_label = f"{cdhit_identity:.2f}".replace(".", "_")
            input_path = os.path.join(
                input_dir, f"{suffix}_agg_{agg_method}_cdhit{identity_label}.csv"
            )
            if not file_exists(file_path=input_path):
                raise FileNotFoundError(f"File not found: '{input_path}'")

            # Z-score Outlier Filtering
            apply_zscore_filter(
                input_csv=input_path,
                output_dir=output_dir,
                logger=logger,
                target_column="Log MIC Value",
                threshold=threshold,
            )
            logger.add_spacer(level=logging.INFO, lines=1)

        # -------------------- Pipeline summary --------------------
        logger.info("[ 'Pipeline Execution Summary' ]")
        logger.add_divider(level=logging.INFO, length=120, border="+", fill="=")
        logger.log_with_borders(
            level=logging.INFO,
            message="\n".join(get_pipeline_completion_message(start_time)),
            border="║",
            length=120,
        )
        logger.add_divider(level=logging.INFO, length=120, border="+", fill="=")

    except Exception:
        logger.exception("Critical failure in 'run_aggregate_pipeline()'")
        raise
