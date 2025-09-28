# pylint: disable=import-error, wrong-import-position, too-many-arguments, too-many-positional-arguments, too-many-locals
"""
Aggregate AMP MIC datasets across multiple strains.
"""
# ============================== Standard Library Imports ==============================
import logging
import os
import sys
import time

# ============================== Third-Party Library Imports ==============================
import numpy as np
import pandas as pd

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
def aggregate_sequence_records(
    input_path: str,
    output_dir: str,
    strain: str,
    logger,
    agg_method: str = "min",
) -> None:
    """
    Aggregate duplicate sequence records by a specified method (min/mean/median),
    merge metadata, and save the aggregated dataset to a separate output directory.

    Parameters
    ----------
    input_path : str
        Path to the input CSV file (must exist and be validated externally).
    output_dir : str
        Directory to save the aggregated output file.
    strain : str
        Strain name (for logging context, e.g. "Escherichia coli").
    logger : CustomLogger
        Logger instance for structured logging.
    agg_method : str, optional
        Aggregation method for Log MIC Value. Default is 'min'.
        Options: ['min', 'mean', 'median']

    Returns
    -------
    None
    """
    logger.info(msg=f"/ Task: Aggregate 'Log MIC Value' for targeting '{strain}'")
    logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

    # Start timing
    start_time = time.time()

    try:
        # Load data for the target strain
        df = load_dataframe_by_columns(file_path=input_path)

        # Filter out invalid values (-inf, inf, NaN) in the "Log MIC Value" column
        df = df[np.isfinite(df["Log MIC Value"])].copy()
        before_count = df.shape[0]
        logger.log_with_borders(
            level=logging.INFO,
            message=(
                f"[ Initial Data Summary ]\n"
                f"▸ Total records loaded: {before_count}\n"
                f"▸ Removed invalid Log MIC (inf / -inf / NaN): {before_count - df.shape[0]}"
            ),
            border="|",
            length=120,
        )
        logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

        # Determine aggregation function
        if agg_method not in ["min", "mean", "median"]:
            raise ValueError(f"Unsupported aggregate_method: '{agg_method}'")

        # Aggregate Log MIC Value
        aggregated_df = df.groupby("Sequence", as_index=False).agg(
            {"Log MIC Value": agg_method}
        )

        # Extract metadata (first occurrence per Sequence)
        meta_df = df.drop_duplicates(subset=["Sequence"], keep="first").drop(
            columns=["Log MIC Value", "MIC", "MIC Value", "Unit"], errors="ignore"
        )

        # Merge aggregated values with metadata
        merged_df = pd.merge(aggregated_df, meta_df, on="Sequence", how="left")

        # Reorder columns (ensure consistent schema)
        cols = [
            "ID",
            "Sequence",
            "Targets",
            "Sequence Length",
            "Log MIC Value",
        ]
        merged_df = merged_df[cols]

        # Log summary
        logger.log_with_borders(
            level=logging.INFO,
            message=(
                f"[ Aggregation Summary ]\n"
                f"▸ Aggregation method used: '{agg_method}'\n"
                f"▸ Unique sequences after aggregation: {merged_df.shape[0]}"
            ),
            border="|",
            length=120,
        )
        logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

        # Save aggregated result
        input_filename = os.path.basename(input_path)
        file_stem = input_filename.replace("_All.csv", "")
        output_csv = os.path.join(output_dir, f"{file_stem}_agg_{agg_method}.csv")
        output_fasta = os.path.join(output_dir, f"{file_stem}_agg_{agg_method}.fasta")
        merged_df.to_csv(path_or_buf=output_csv, index=False)
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
        logger.exception(msg="Unexpected error in 'aggregate_sequence_records()'.")
        raise


# ============================== Pipeline Entry Point ==============================
def run_aggregate_pipeline(base_path: str, logger: CustomLogger, **kwargs) -> None:
    """
    Aggregate strain-specific AMP datasets.

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
            "aggregate_input_dir",
            os.path.join(base_path, "data/interim"),
        )
        output_dir = kwargs.get(
            "aggregate_output_dir",
            os.path.join(base_path, "data/interim/aggregated"),
        )
        agg_method = kwargs.get("aggregate_method", "min").lower()

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
        for _, (strain, suffix) in enumerate(strains.items(), start=1):

            # Check input files exist
            input_path = os.path.join(input_dir, f"{suffix}_All.csv")
            if not file_exists(file_path=input_path):
                raise FileNotFoundError(f"File not found: '{input_path}'")

            # Aggregation
            aggregate_sequence_records(
                input_path=input_path,
                output_dir=output_dir,
                strain=strain,
                logger=logger,
                agg_method=agg_method,
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
