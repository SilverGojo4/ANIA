# pylint: disable=line-too-long, import-error, wrong-import-position, too-many-arguments, too-many-positional-arguments, too-many-locals, too-many-statements
"""
Stratified Train/Test Split by Log MIC Value Pipeline
"""
# ============================== Standard Library Imports ==============================
import logging
import os
import sys
import time

# ============================== Third-Party Library Imports ==============================
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

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
def split_train_test_by_logmic(
    input_csv: str,
    output_dir: str,
    logger: CustomLogger,
    target_column: str = "Log MIC Value",
    test_size: float = 0.2,
    n_bins: int = 10,
    random_state: int = 42,
) -> None:
    """
    Perform stratified train/test split based on Log MIC value distribution.

    Parameters
    ----------
    input_csv : str
        Path to the input CSV file.
    output_dir : str
        Directory to save train/test split outputs.
    logger : CustomLogger
        Logger instance for structured logging.
    target_column : str, optional
        Column used for stratification (default = "Log MIC Value").
    test_size : float, optional
        Proportion of the dataset to include in the test split (default = 0.2).
    n_bins : int, optional
        Number of bins to discretize the Log MIC value for stratification (default = 10).
    random_state : int, optional
        Random seed for reproducibility (default = 42).

    Returns
    -------
    None
    """
    logger.info(
        f"/ Task: Stratified 'Train/Test' Split based on '{target_column}' from '{os.path.basename(input_csv)}'"
    )
    logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

    # Start timing
    start_time = time.time()

    try:
        # Load data
        df = load_dataframe_by_columns(file_path=input_csv)
        total_records = len(df)

        # Create stratification bins
        df["StratBin"] = pd.cut(df[target_column], bins=n_bins, labels=False)

        # Initialize stratified splitter
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )

        # Internal Function: Balance Checker
        def check_stratified_balance(df, train_idx, test_idx, logger):
            """
            Check stratified bin distribution between Train/Test splits.

            Parameters
            ----------
            df : pd.DataFrame
                Original dataframe containing the 'StratBin' column.
            train_idx : array-like
                Indices of the training set.
            test_idx : array-like
                Indices of the testing set.
            logger : CustomLogger
                Logger instance for structured logging.

            Returns
            -------
            None
            """
            # Compute normalized distributions
            total_counts = df["StratBin"].value_counts(normalize=True).sort_index()
            train_counts = (
                df.iloc[train_idx]["StratBin"].value_counts(normalize=True).sort_index()
            )
            test_counts = (
                df.iloc[test_idx]["StratBin"].value_counts(normalize=True).sort_index()
            )

            # Build comparison table
            compare = pd.DataFrame(
                {
                    "Total (%)": (total_counts * 100).round(2),
                    "Train (%)": (train_counts * 100).round(2),
                    "Test (%)": (test_counts * 100).round(2),
                }
            ).fillna(0.0)
            compare["Diff (Train-Test)"] = (
                (compare["Train (%)"] - compare["Test (%)"]).abs().round(2)
            )

            # Summary statistics
            mean_diff = compare["Diff (Train-Test)"].mean()
            max_diff = compare["Diff (Train-Test)"].max()
            if max_diff <= 1:
                balance_status = "Excellent balance"
            elif max_diff <= 2:
                balance_status = "Balanced (OK)"
            elif max_diff <= 5:
                balance_status = "Moderate imbalance"
            else:
                balance_status = "Imbalance detected"

            # Construct detailed rows
            lines = []
            for idx, row in compare.iterrows():
                lines.append(
                    f"▸ Bin {idx:<2d} - Total: {row['Total (%)']:>5.2f}%  "
                    f"Train: {row['Train (%)']:>5.2f}%  Test: {row['Test (%)']:>5.2f}%  "
                    f"Diff: {row['Diff (Train-Test)']:>4.2f}%"
                )
            summary_text = "\n".join(lines)
            results = (
                f"▸ Mean abs diff : {mean_diff:.2f}% | "
                f"Max abs diff  : {max_diff:.2f}% | "
                f"Result → '{balance_status}'"
            )
            logger.log_with_borders(
                level=logging.INFO,
                message=(
                    "[ Stratified Bin Balance Summary ]\n"
                    f"{results}\n"
                    f"{summary_text}\n"
                ),
                border="|",
                length=120,
            )
            logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

        # Log basic distribution check
        def describe_split(df_part: pd.DataFrame, name: str) -> str:
            """Return formatted description of Log MIC Value distribution."""
            desc = df_part[target_column].describe()
            return (
                f"{name} - Count: {int(desc['count'])}, "
                f"Mean: {desc['mean']:.4f}, Std: {desc['std']:.4f}, "
                f"Min: {desc['min']:.4f}, Max: {desc['max']:.4f}"
            )

        for train_idx, test_idx in splitter.split(df, df["StratBin"]):

            # Create train/test data
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()
            train_df.drop(columns=["StratBin"], inplace=True)
            test_df.drop(columns=["StratBin"], inplace=True)

            # Run post-split balance validation
            check_stratified_balance(df, train_idx, test_idx, logger)

            # Log summary
            logger.log_with_borders(
                level=logging.INFO,
                message=(
                    f"[ Train/Test Split Summary ]\n"
                    f"▸ Total records      : {total_records}\n"
                    f"▸ Train set size     : {len(train_df)} ({len(train_df)/total_records*100:.2f}%)\n"
                    f"▸ Test set size      : {len(test_df)} ({len(test_df)/total_records*100:.2f}%)\n"
                    f"▸ Stratification bins: {n_bins}\n"
                ),
                border="|",
                length=120,
            )
            logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")
            logger.log_with_borders(
                level=logging.INFO,
                message="[ Distribution Summary ]\n"
                f"▸ {describe_split(train_df, 'Train')}\n"
                f"▸ {describe_split(test_df, 'Test')}\n",
                border="|",
                length=120,
            )
            logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

            # Save CSVs and FASTA
            train_csv = os.path.join(output_dir, "train.csv")
            train_fasta = os.path.join(output_dir, "train.fasta")
            test_csv = os.path.join(output_dir, "test.csv")
            test_fasta = os.path.join(output_dir, "test.fasta")
            train_df.to_csv(train_csv, index=False)
            test_df.to_csv(test_csv, index=False)
            write_fasta_file(input_path=train_csv, output_fasta=train_fasta)
            write_fasta_file(input_path=test_csv, output_fasta=test_fasta)
            logger.log_with_borders(
                level=logging.INFO,
                message=f"Saved:\n'{train_csv}'\n'{test_csv}'\n'{train_fasta}'\n'{test_fasta}'",
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
        logger.exception("Unexpected error in 'split_train_test_by_logmic()'.")
        raise


def split_kfold_by_logmic(
    train_csv: str,
    output_dir: str,
    logger: CustomLogger,
    target_column: str = "Log MIC Value",
    n_splits: int = 5,
    n_bins: int = 10,
    random_state: int = 42,
) -> None:
    """
    Perform stratified K-Fold split within the training set based on Log MIC value distribution.

    Parameters
    ----------
    train_csv : str
        Path to the train.csv file generated by the outer hold-out split.
    output_dir : str
        Directory to save the K-Fold outputs (subfolder "folds" will be created).
    logger : CustomLogger
        Logger instance for structured logging.
    target_column : str, optional
        Column used for stratification (default = "Log MIC Value").
    n_splits : int, optional
        Number of folds for Stratified K-Fold (default = 5).
    n_bins : int, optional
        Number of bins to discretize the Log MIC value for stratification (default = 10).
    random_state : int, optional
        Random seed for reproducibility (default = 42).

    Returns
    -------
    None
    """
    logger.info(
        f"/ Task: Stratified {n_splits}-Fold Split within Train Set based on '{target_column}' from '{os.path.basename(train_csv)}'"
    )
    logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

    # Start timing
    start_time = time.time()

    try:
        # Load Training Data
        df = load_dataframe_by_columns(file_path=train_csv)
        total_records = len(df)

        # Create stratification bins
        df["StratBin"] = pd.cut(df[target_column], bins=n_bins, labels=False)

        # Prepare Output Directory
        folds_dir = os.path.join(output_dir, "folds")
        if not directory_exists(dir_path=folds_dir):
            os.makedirs(name=folds_dir)

        # Initialize StratifiedKFold
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )

        # -------------------- Internal Function: Fold Balance Checker --------------------
        def check_fold_balance(df, train_idx, val_idx, fold_id, logger):
            """
            Check stratified bin distribution between Train/Val folds.
            """
            # Compute normalized distributions
            total_counts = df["StratBin"].value_counts(normalize=True).sort_index()
            train_counts = (
                df.iloc[train_idx]["StratBin"].value_counts(normalize=True).sort_index()
            )
            val_counts = (
                df.iloc[val_idx]["StratBin"].value_counts(normalize=True).sort_index()
            )

            # Build comparison table
            compare = pd.DataFrame(
                {
                    "Total (%)": (total_counts * 100).round(2),
                    "Train (%)": (train_counts * 100).round(2),
                    "Val (%)": (val_counts * 100).round(2),
                }
            ).fillna(0.0)
            compare["Diff (Train-Val)"] = (
                (compare["Train (%)"] - compare["Val (%)"]).abs().round(2)
            )

            # Summary statistics
            mean_diff = compare["Diff (Train-Val)"].mean()
            max_diff = compare["Diff (Train-Val)"].max()
            if max_diff <= 1:
                balance_status = "Excellent balance"
            elif max_diff <= 2:
                balance_status = "Balanced (OK)"
            elif max_diff <= 5:
                balance_status = "Moderate imbalance"
            else:
                balance_status = "Imbalance detected"

            # Construct detailed rows
            lines = []
            for idx, row in compare.iterrows():
                lines.append(
                    f"▸ Bin {idx:<2d} - Total: {row['Total (%)']:>5.2f}%  "
                    f"Train: {row['Train (%)']:>5.2f}%  Val: {row['Val (%)']:>5.2f}%  "
                    f"Diff: {row['Diff (Train-Val)']:>4.2f}%"
                )
            summary_text = "\n".join(lines)
            results = (
                f"▸ Mean abs diff : {mean_diff:.2f}% | "
                f"Max abs diff  : {max_diff:.2f}% | "
                f"Result → '{balance_status}'"
            )
            logger.log_with_borders(
                level=logging.INFO,
                message=(
                    f"[ Fold {fold_id} Bin Balance Summary ]\n"
                    f"{results}\n"
                    f"{summary_text}\n"
                ),
                border="|",
                length=120,
            )
            logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

        def describe_split(df_part: pd.DataFrame, name: str) -> str:
            """Return formatted description of Log MIC Value distribution."""
            desc = df_part[target_column].describe()
            return (
                f"{name} - Count: {int(desc['count'])}, "
                f"Mean: {desc['mean']:.4f}, Std: {desc['std']:.4f}, "
                f"Min: {desc['min']:.4f}, Max: {desc['max']:.4f}"
            )

        # -------------------- Generate Folds --------------------
        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(df, df["StratBin"]), start=1
        ):
            # Create fold data
            train_fold = df.iloc[train_idx].copy()
            val_fold = df.iloc[val_idx].copy()
            train_fold.drop(columns=["StratBin"], inplace=True)
            val_fold.drop(columns=["StratBin"], inplace=True)

            # Check balance
            check_fold_balance(df, train_idx, val_idx, fold_idx, logger)

            # Log fold summary
            logger.log_with_borders(
                level=logging.INFO,
                message=(
                    f"[ Fold {fold_idx} Summary ]\n"
                    f"▸ Total records      : {total_records}\n"
                    f"▸ Train set size.    : {len(train_fold)} ({len(train_fold)/total_records*100:.2f}%)\n"
                    f"▸ Val set size.      : {len(val_fold)} ({len(val_fold)/total_records*100:.2f}%)\n"
                    f"▸ Stratification bins: {n_bins}\n"
                ),
                border="|",
                length=120,
            )
            logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")
            logger.log_with_borders(
                level=logging.INFO,
                message="[ Distribution Summary ]\n"
                f"▸ {describe_split(train_fold, 'Train')}\n"
                f"▸ {describe_split(val_fold, 'Val')}\n",
                border="|",
                length=120,
            )
            logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

            # Save fold CSVs and FASTA
            train_fold_csv = os.path.join(folds_dir, f"train_fold{fold_idx}.csv")
            val_fold_csv = os.path.join(folds_dir, f"val_fold{fold_idx}.csv")
            train_fold_fasta = os.path.join(folds_dir, f"train_fold{fold_idx}.fasta")
            val_fold_fasta = os.path.join(folds_dir, f"val_fold{fold_idx}.fasta")
            train_fold.to_csv(train_fold_csv, index=False)
            val_fold.to_csv(val_fold_csv, index=False)
            write_fasta_file(
                input_path=train_fold_csv,
                output_fasta=train_fold_fasta,
            )
            write_fasta_file(
                input_path=val_fold_csv,
                output_fasta=val_fold_fasta,
            )
            logger.log_with_borders(
                level=logging.INFO,
                message=f"Saved:\n'{train_fold_csv}'\n'{val_fold_csv}'\n'{train_fold_fasta}'\n'{val_fold_fasta}'",
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
        logger.exception("Unexpected error in 'split_kfold_by_logmic()'.")
        raise


# ============================== Pipeline Entry Point ==============================
def run_split_pipeline(base_path: str, logger: CustomLogger, **kwargs) -> None:
    """
    Executes the train/test split pipeline for strain-specific AMP datasets
    based on Log MIC Value stratification.

    Parameters
    ----------
    base_path : str
        Project root path (used to build file paths).
    logger : CustomLogger
        Logger instance for structured logging.
    **kwargs : dict
        Optional overrides for input/output file paths and parameters.

    Returns
    -------
    None
    """
    start_time = time.time()

    try:
        # -------------------- Retrieve configuration --------------------
        input_dir = kwargs.get(
            "split_input_dir", os.path.join(base_path, "data/interim/group")
        )
        output_dir = kwargs.get(
            "split_output_dir", os.path.join(base_path, "data/processed/")
        )
        agg_method = kwargs.get("split_aggregate_method", "min").lower()
        cdhit_identity = kwargs.get("split_cdhit_identity", 0.9)
        threshold = kwargs.get("split_threshold", 3.0)
        test_size = kwargs.get("split_test_size", 0.2)
        n_bins = kwargs.get("split_n_bins", 10)
        random_state = kwargs.get("split_random_state", 42)
        n_splits = kwargs.get("split_n_splits", 5)

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
            # Construct input path
            identity_label = f"{cdhit_identity:.2f}".replace(".", "_")
            threshold_label = str(threshold).replace(".0", "")
            input_path = os.path.join(
                input_dir,
                f"{suffix}_agg_{agg_method}_cdhit{identity_label}_z{threshold_label}.csv",
            )

            if not file_exists(file_path=input_path):
                raise FileNotFoundError(f"File not found: '{input_path}'")

            # Create species-specific output folder
            species_output_dir = os.path.join(output_dir, suffix)
            if not directory_exists(dir_path=species_output_dir):
                os.makedirs(name=species_output_dir)

            # Hold-out Split
            split_train_test_by_logmic(
                input_csv=input_path,
                output_dir=species_output_dir,
                logger=logger,
                target_column="Log MIC Value",
                test_size=test_size,
                n_bins=n_bins,
                random_state=random_state,
            )
            logger.add_spacer(level=logging.INFO, lines=1)

            # Step 2: K-Fold Split (on Train)
            train_csv = os.path.join(species_output_dir, "train.csv")
            split_kfold_by_logmic(
                train_csv=train_csv,
                output_dir=species_output_dir,
                logger=logger,
                target_column="Log MIC Value",
                n_splits=n_splits,
                n_bins=n_bins,
                random_state=random_state,
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
        logger.exception("Critical failure in 'run_split_pipeline()'")
        raise
