# pylint: disable=line-too-long, import-error, wrong-import-position, too-many-locals, too-many-statements
"""
This module handles the grouping of antimicrobial peptide (AMP) data based on the
log-transformed Minimum Inhibitory Concentration (Log MIC Value). It assigns samples
into three balanced categories ('low', 'medium', 'high') using quantile-based binning
and exports group-wise datasets in both CSV and FASTA formats.

The grouping pipeline:
1. Loads cleaned and aggregated AMP data for each strain.
2. Applies quantile-based binning to split samples into three MIC groups.
3. Logs group boundaries and group sample counts for reproducibility.
4. Saves the grouped dataset with an added 'MIC Group' column.
5. Further splits the grouped data into separate CSV and FASTA files per group.

The core functions in this module are:
- `assign_mic_group()`: Assigns balanced group labels ('low', 'medium', 'high') using quantile cutoffs and saves the grouped dataset.
- `save_grouped_datasets()`: Splits the grouped data into three subsets and saves them as individual CSV and FASTA files.
- `run_group_pipeline()`: Entry point for executing the full MIC grouping pipeline across supported strains.
"""
# ============================== Standard Library Imports ==============================
import logging
import os
import sys

# ============================== Third-Party Library Imports ==============================
import pandas as pd

# ============================== Project Root Path Setup ==============================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
LOGGING_PATH = os.path.join(BASE_PATH, "src/utils/logging_toolkit/src/python")

# Fix Python Path
if BASE_PATH not in sys.path:
    sys.path.append(BASE_PATH)
if LOGGING_PATH not in sys.path:
    sys.path.append(LOGGING_PATH)

# ============================== Project-Specific Imports ==============================
# Logging configuration and custom logger
from setup_logging import CustomLogger

# Utility Functions
from src.utils.common import write_fasta_file


# ============================== Custom Function ==============================
def assign_mic_group(
    df: pd.DataFrame,
    output_path: str,
    logger: CustomLogger,
) -> pd.DataFrame:
    """
    Assigns a balanced group label (low, medium, high) to each sample based on Log MIC Value.

    This function uses quantile-based binning to ensure approximately equal sample sizes in
    each group. The group label is stored in a new column named 'MIC Group'.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing a 'Log MIC Value' column.
    output_path : str
        Full file path to save the resulting DataFrame with the 'MIC Group' column.
    logger : CustomLogger
        Logger for logging messages and debugging information.

    Returns
    -------
    pd.DataFrame
        The original DataFrame with an additional 'MIC Group' column.
    """
    try:
        logger.info(
            msg="/ Task: Assigning MIC group labels based on 'Log MIC Value' ('CSV' and 'FASTA')"
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Apply quantile-based binning into 3 groups and extract bin edges
        mic_grouped, bin_edges = pd.qcut(
            df["Log MIC Value"],
            q=3,
            labels=["low", "medium", "high"],
            retbins=True,
            duplicates="drop",  # in case some values are identical
        )
        df["MIC Group"] = mic_grouped

        # Log bin ranges
        bin_ranges = [
            f"[{bin_edges[i]:.4f}, {bin_edges[i+1]:.4f})"
            for i in range(len(bin_edges) - 2)
        ] + [
            f"[{bin_edges[-2]:.4f}, {bin_edges[-1]:.4f}]"
        ]  # last bin includes right edge

        logger.log_with_borders(
            level=logging.INFO,
            message="MIC Group bin ranges ('Log MIC Value'):\n"
            + "\n".join(
                [
                    f"  • {label.capitalize():<7}: {rng}"
                    for label, rng in zip(["low", "medium", "high"], bin_ranges)
                ]
            ),
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Log group sizes
        group_counts = df["MIC Group"].value_counts().to_dict()
        logger.log_with_borders(
            level=logging.INFO,
            message="Sample counts per MIC Group:\n"
            + "\n".join(
                [f"  • {k.capitalize():<7}: {v}" for k, v in group_counts.items()]
            ),
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Save grouped DataFrame to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

        # Also save to FASTA format using same base path
        fasta_path = os.path.splitext(output_path)[0] + ".fasta"
        write_fasta_file(df=df, output_fasta=fasta_path)
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Saved MIC-grouped dataset to:\n'{output_path}'\n'{fasta_path}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        return df

    except Exception:
        logger.exception(msg="Unexpected error in 'assign_mic_group()'.")
        raise


def save_grouped_datasets(
    df: pd.DataFrame,
    output_dir: str,
    strain_suffix: str,
    logger: CustomLogger,
) -> None:
    """
    Splits the MIC-grouped DataFrame into separate groups and saves each as a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the 'MIC Group' column.
    output_dir : str
        Base directory to save output files.
    strain_suffix : str
        Suffix identifier for the strain (e.g., 'EC', 'PA', 'SA').
    logger : CustomLogger
        Logger for structured logging.

    Returns
    -------
    None
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        logger.info(msg="/ Task: Saving per-group datasets ('CSV' and 'FASTA')")
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Loop over the three group labels
        for group in ["low", "medium", "high"]:
            group_df = df[df["MIC Group"] == group]
            output_csv = os.path.join(output_dir, f"{strain_suffix}_{group}.csv")
            group_df.to_csv(output_csv, index=False)

            # Write sequences to FASTA file
            output_fasta = os.path.join(output_dir, f"{strain_suffix}_{group}.fasta")
            write_fasta_file(
                df=group_df,
                output_fasta=output_fasta,
            )

            # Log save info
            logger.log_with_borders(
                level=logging.INFO,
                message=f"Saved '{group.capitalize()}' group ({len(group_df)} rows) to:\n'{output_csv}'\n'{output_fasta}'",
                border="|",
                length=100,
            )
            logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

    except Exception:
        logger.exception("Unexpected error in 'save_grouped_datasets()'.")
        raise


# ============================== Pipeline Entry Point ==============================
def run_group_pipeline(base_path: str, logger: CustomLogger) -> None:
    """
    Executes the MIC group assignment pipeline for strain-specific AMP datasets.

    This pipeline loads the cleaned and aggregated dataset for each strain, assigns samples into
    three balanced groups ('low', 'medium', 'high') based on their 'Log MIC Value', and saves
    the resulting annotated FASTA file.

    Parameters
    ----------
    base_path : str
        Absolute base path of the project (used to construct file paths).
    logger : CustomLogger
        Logger for structured logging throughout the pipeline.

    Returns
    -------
    None
        This function performs file I/O and logging side effects, and does not return any value.
    """
    try:
        # Mapping of full strain names to their corresponding suffixes
        strains = {
            "Escherichia coli": "EC",
            "Pseudomonas aeruginosa": "PA",
            "Staphylococcus aureus": "SA",
        }

        # Loop over each strain type
        for strain_index, (strain, suffix) in enumerate(strains.items(), start=1):

            # Logging: strain section
            logger.info(msg=f"/ {strain_index}")
            logger.add_divider(level=logging.INFO, length=40, border="+", fill="-")
            logger.log_with_borders(
                level=logging.INFO,
                message=f"Strain - '{strain}'",
                border="|",
                length=40,
            )
            logger.add_divider(level=logging.INFO, length=40, border="+", fill="-")
            logger.add_spacer(level=logging.INFO, lines=1)

            # Load data for the target strain
            data_input_file = os.path.join(
                base_path, f"data/processed/all/{suffix}.csv"
            )
            data = pd.read_csv(filepath_or_buffer=data_input_file)

            # Assign MIC group labels ('low', 'medium', 'high') using balanced tertile cut
            group_data_output_file = os.path.join(
                base_path,
                f"data/processed/all/{suffix}.csv",
            )
            data = assign_mic_group(
                df=data, logger=logger, output_path=group_data_output_file
            )
            logger.add_spacer()

            # Reload the aggregated data
            data_input_file = os.path.join(
                base_path,
                f"data/processed/all/{suffix}.csv",
            )
            data = pd.read_csv(filepath_or_buffer=data_input_file)

            # Save each MIC group ('low', 'medium', 'high') as a separate CSV file
            group_output_dir = os.path.join(base_path, "data/processed/group")
            save_grouped_datasets(
                df=data,
                output_dir=group_output_dir,
                strain_suffix=suffix,
                logger=logger,
            )
            logger.add_spacer()

    except Exception:
        logger.exception(msg="Unexpected error in 'run_collect_pipeline()'.")
        raise
