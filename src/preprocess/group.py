# pylint: disable=line-too-long, import-error, wrong-import-position, too-many-locals
"""
MIC Value Grouping Pipeline
"""
# ============================== Standard Library Imports ==============================
import logging
import os
import sys
import time

# ============================== Third-Party Library Imports ==============================
import pandas as pd

# ============================== Project Root Path Setup ==============================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
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
def assign_mic_group(
    input_csv: str,
    output_dir: str,
    logger: CustomLogger,
    target_column: str = "Log MIC Value",
) -> pd.DataFrame:
    """
    Apply grouping (low / medium / high) based on mean ± std thresholds of a target column.

    Parameters
    ----------
    input_csv : str
        Path to the input CSV file.
    output_dir : str
        Directory to save filtered CSV and FASTA outputs.
    logger : CustomLogger
        Logger instance for structured logging.
    target_column : str, optional
        Column used for grouping (default = "Log MIC Value").

    Returns
    -------
    None
    """
    logger.info(
        f"/ Task: Assign MIC group labels based on '{os.path.basename(input_csv)}'"
    )
    logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

    # Start timing
    start_time = time.time()

    try:
        # Load Data
        df = load_dataframe_by_columns(file_path=input_csv)

        # Compute statistics
        mean_val = df[target_column].mean()
        std_val = df[target_column].std()
        min_val = df[target_column].min()
        max_val = df[target_column].max()

        # Define bin edges and labels
        bin_edges = [
            min_val - 1e-8,
            mean_val - std_val,
            mean_val + std_val,
            max_val + 1e-8,
        ]
        labels = ["low", "medium", "high"]

        # Assign groups
        df["MIC Group"] = pd.cut(
            df[target_column], bins=bin_edges, labels=labels, include_lowest=True
        )

        # Log unified MIC Group Summary (Sample Counts + Bin Ranges)
        bin_ranges = [
            f"[{min_val:.4f}, {mean_val - std_val:.4f})",
            f"[{mean_val - std_val:.4f}, {mean_val + std_val:.4f})",
            f"[{mean_val + std_val:.4f}, {max_val:.4f}]",
        ]
        group_counts = df["MIC Group"].value_counts().to_dict()
        logger.log_with_borders(
            level=logging.INFO,
            message="[ MIC Group Summary ]\n"
            + "\n".join(
                [
                    f"▸ {label.capitalize():<7}: {group_counts.get(label, 0):<5} samples  | Range: {rng}"
                    for label, rng in zip(labels, bin_ranges)
                ]
            ),
            border="|",
            length=120,
        )
        logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

        # Save grouped result
        input_filename = os.path.basename(input_csv)
        file_stem = os.path.splitext(input_filename)[0]  # remove .csv safely
        output_csv = os.path.join(output_dir, f"{file_stem}.csv")
        df.to_csv(output_csv, index=False)
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Saved:\n'{output_csv}'",
            border="|",
            length=120,
        )
        for group in labels:
            df_group = df[df["MIC Group"] == group].copy()
            if not df_group.empty:
                csv_path = os.path.join(output_dir, f"{file_stem}_{group}.csv")
                fasta_path = os.path.join(output_dir, f"{file_stem}_{group}.fasta")
                df_group.to_csv(csv_path, index=False)
                write_fasta_file(input_path=csv_path, output_fasta=fasta_path)
                logger.log_with_borders(
                    level=logging.INFO,
                    message=f"'{csv_path}'\n'{fasta_path}'",
                    border="|",
                    length=120,
                )
            else:
                logger.log_with_borders(
                    level=logging.INFO,
                    message=f"No records in '{group}' group — FASTA and CSV not created.",
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

        return df

    except Exception:
        logger.exception(msg="Unexpected error in 'assign_mic_group()'.")
        raise


# ============================== Pipeline Entry Point ==============================
def run_group_pipeline(base_path: str, logger: CustomLogger, **kwargs) -> None:
    """
    Executes the MIC group assignment pipeline for strain-specific AMP datasets.

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
        # -------------------- Retrieve configuration --------------------
        input_dir = kwargs.get(
            "group_input_dir", os.path.join(base_path, "data/interim/zscore")
        )
        output_dir = kwargs.get(
            "group_output_dir", os.path.join(base_path, "data/interim/group")
        )
        agg_method = kwargs.get("group_aggregate_method", "min").lower()
        cdhit_identity = kwargs.get("group_cdhit_identity", 0.9)
        threshold = kwargs.get("group_threshold", 3.0)

        # Ensure directories exist
        for directory in [output_dir]:
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
            threshold_label = str(threshold).replace(".0", "")
            input_path = os.path.join(
                input_dir,
                f"{suffix}_agg_{agg_method}_cdhit{identity_label}_z{threshold_label}.csv",
            )
            if not file_exists(file_path=input_path):
                raise FileNotFoundError(f"File not found: '{input_path}'")

            # Assign MIC Groups
            assign_mic_group(
                input_csv=input_path,
                output_dir=output_dir,
                logger=logger,
                target_column="Log MIC Value",
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
        logger.exception("Critical failure in 'run_group_pipeline()'")
        raise
