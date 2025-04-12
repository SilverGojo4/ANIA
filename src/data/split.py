# pylint: disable=line-too-long, import-error, wrong-import-position, too-many-locals, too-many-statements
"""
This module handles the final preparation of antimicrobial peptide (AMP) datasets
for model training and evaluation. It performs stratified train/test splitting
based on the 'Log MIC Value' distribution and exports both CSV and FASTA files.

Main functionalities:
1. Stratified splitting using binned 'Log MIC Value' for class balance.
2. Export of training and testing sets in both CSV and FASTA format.
3. Logging of split ratios and sample counts.

The core function in this module is:
- `split_and_save_dataset()`: Splits the dataset and saves outputs with a given file prefix.
"""
# ============================== Standard Library Imports ==============================
import logging
import os
import sys

# ============================== Third-Party Library Imports ==============================
import pandas as pd
from sklearn.model_selection import train_test_split

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
def split_and_save_dataset(
    df: pd.DataFrame,
    output_dir: str,
    strain_suffix: str,
    test_size: float,
    logger: CustomLogger,
) -> None:
    """
    Split dataset into training and testing sets and export to CSV and FASTA formats.

    The splitting preserves distribution by stratifying on binned 'Log MIC Value'.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to be split.
    output_dir : str
        Directory where the output files will be saved.
    strain_suffix : str
        Identifier for the strain (e.g., 'EC', 'PA', 'SA').
    test_size : float
        Proportion of dataset to allocate as test set.
    logger : CustomLogger
        Logger instance for recording progress and output files.

    Returns
    -------
    None
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(msg="/ Task: Splitting 'train/test' sets ('CSV' and 'FASTA')")
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Splitting dataset into {int((1 - test_size) * 100)} % train / {int(test_size * 100)} % test.",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Stratify using binned Log MIC Value
        log_mic_bins = pd.cut(df["Log MIC Value"], bins=10, labels=False)
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=42,
            stratify=log_mic_bins,
        )

        # Define file paths
        train_csv = os.path.join(output_dir, f"{strain_suffix}_train.csv")
        test_csv = os.path.join(output_dir, f"{strain_suffix}_test.csv")
        train_fasta = os.path.join(output_dir, f"{strain_suffix}_train.fasta")
        test_fasta = os.path.join(output_dir, f"{strain_suffix}_test.fasta")

        # Save CSV files
        train_df.to_csv(train_csv, index=False)
        test_df.to_csv(test_csv, index=False)

        logger.log_with_borders(
            level=logging.INFO,
            message=f"Training dataset saved ({len(train_df)}):\n'{train_csv}'\n'{train_fasta}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Testing dataset saved ({len(test_df)}):\n'{test_csv}'\n'{test_fasta}'",
            border="|",
            length=100,
        )

        # Save FASTA files
        write_fasta_file(train_df, output_fasta=train_fasta)
        write_fasta_file(test_df, output_fasta=test_fasta)
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

    except Exception:
        logger.exception(msg="Unexpected error in 'split_and_save_dataset()'.")
        raise


# ============================== Pipeline Entry Point ==============================
def run_split_pipeline(base_path: str, logger: CustomLogger) -> None:
    """
    Pipeline entry point for splitting AMP datasets (per strain) into train and test sets.

    This function loops over all target strains (e.g., EC, PA, SA), loads the cleaned dataset
    from `data/processed/all/`, and splits it into training/testing subsets using stratified
    sampling based on 'Log MIC Value'. The results are saved as CSV and FASTA files to
    `data/processed/split/`.

    Parameters
    ----------
    base_path : str
        Absolute base path of the project (used to construct file paths).
    logger : CustomLogger
        Logger instance for structured logging throughout the pipeline.

    Returns
    -------
    None
        This function performs file I/O and logging side effects, and does not return any value.
    """
    try:
        # Define strain mapping: Full name → short suffix used in filenames
        strains = {
            "Escherichia coli": "EC",
            "Pseudomonas aeruginosa": "PA",
            "Staphylococcus aureus": "SA",
        }

        # Loop over each strain and perform splitting
        for strain_index, (strain, suffix) in enumerate(strains.items(), start=1):

            # Logging: Mark section for current strain
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

            # Load the cleaned dataset for the current strain
            data_input_file = os.path.join(
                base_path, f"data/processed/all/{suffix}.csv"
            )
            data = pd.read_csv(filepath_or_buffer=data_input_file)

            # Define output directory for split results
            split_output_dir = os.path.join(base_path, "data/processed/split")

            # Perform stratified train/test splitting and save outputs
            split_and_save_dataset(
                df=data,
                output_dir=split_output_dir,
                strain_suffix=suffix,
                test_size=0.2,
                logger=logger,
            )

            logger.add_spacer()

    except Exception:
        logger.exception(msg="Unexpected error in 'run_split_pipeline()'.")
        raise
