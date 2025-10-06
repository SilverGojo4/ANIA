# pylint: disable=import-error, wrong-import-position, too-many-locals, too-many-branches
"""
Run iFeature Feature Extraction
"""
# ============================== Standard Library Imports ==============================
import logging
import os
import subprocess
import sys
import time

# ============================== Third-Party Library Imports ==============================
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
    load_yml_config,
)
from utils.log_utils import get_pipeline_completion_message, get_task_completion_message


# ============================== Custom Functions ==============================
def format_command_for_logging(command: list[str]) -> str:
    """
    Format a subprocess command list into a readable multi-line string for structured logging.

    Parameters
    ----------
    command : list of str
        List representation of a command to be executed via subprocess,
        where each element is a separate token (e.g., ["python", "script.py", "--arg", "value"]).

    Returns
    -------
    str
        Formatted multi-line string representation of the command for logging display.
    """
    if not command:
        return ""

    formatted_lines = []

    # First element (python or executable)
    formatted_lines.append(f"{command[0]} \\")

    # Second element (script path if exists)
    if len(command) > 1:
        formatted_lines.append(f"  {command[1]} \\")

    # Remaining args
    args = command[2:]
    i = 0
    while i < len(args):
        if (
            args[i].startswith("--")
            and i + 1 < len(args)
            and not args[i + 1].startswith("--")
        ):
            # Pair argument with value
            formatted_lines.append(f"  {args[i]} {args[i+1]} \\")
            i += 2
        else:
            # Single flag or leftover token
            formatted_lines.append(f"  {args[i]} \\")
            i += 1

    # Remove trailing backslash from last line
    if formatted_lines[-1].endswith("\\"):
        formatted_lines[-1] = formatted_lines[-1].rstrip(" \\")

    return "\n".join(formatted_lines)


def convert_tsv_to_df(
    input_tsv: str,
    feature_prefix: str,
) -> pd.DataFrame:
    """
    Load a TSV file, rename feature columns by adding a specified prefix,
    and return the processed DataFrame.

    Parameters
    ----------
    input_tsv : str
        Path to the input TSV file.
    feature_prefix : str
        Prefix to be added before each feature column name.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with renamed feature columns.
    """
    # Load the TSV file into a DataFrame
    df = load_dataframe_by_columns(file_path=input_tsv)

    # Check if the file is empty
    if df.empty:
        raise ValueError(f"Input file '{input_tsv}' is empty.")

    # Ensure the first column is an identifier column (ID)
    first_col = df.columns[0]
    if first_col != "#":
        raise KeyError(
            f"The first column in the TSV file must be '#', but got '{first_col}'."
        )

    # Rename the first column from '#' to 'ID'
    df.rename(columns={"#": "ID"}, inplace=True)

    # Rename feature columns (exclude the first column, which is now 'ID')
    feature_cols = df.columns[1:]
    df.rename(
        columns={col: f"{feature_prefix}|{col}" for col in feature_cols},
        inplace=True,
    )

    return df


def ifeature_encoding(
    input_fasta: str,
    output_csv: str,
    feature_type: str,
    feature_config: dict,
    logger: CustomLogger,
) -> None:
    """
    Execute iFeature feature extraction for a given feature type based on preloaded configuration.

    Parameters
    ----------
    input_fasta : str
        Path to the input FASTA file.
    output_csv : str
        Path to the final output CSV file.
    feature_type : str
        Feature encoding type (key used to retrieve config, e.g., 'AAC', 'PAAC', etc.).
    feature_config : dict
        Configuration dictionary loaded from YAML, containing command templates for all features.
    logger : CustomLogger
        Logger instance for structured logging.

    Returns
    -------
    None
    """
    logger.info(f"/ Task: Run iFeature '{feature_type}' on '{input_fasta}'")
    logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

    # Start timing
    start_time = time.time()

    try:
        # Validate config
        if feature_type not in feature_config:
            raise KeyError(f"Feature type '{feature_type}' not found in configuration.")
        config = feature_config[feature_type]

        # Convert the user-provided CSV output path to an internal TSV path
        output_tsv = output_csv.replace(".csv", ".tsv")
        placeholders = {
            "input_fasta": input_fasta,
            "output_tsv": output_tsv,
            "output_csv": output_csv,
        }

        # Build the command
        if "cmd_template" in config:
            # single-string command
            command_str = config["cmd_template"].format(**placeholders)
            command = command_str.split()
        else:
            # script + args list
            script_path = config.get("script")
            if not script_path:
                raise ValueError(
                    f"Missing 'script' field for feature type '{feature_type}'"
                )

            args = config.get("args", [])
            formatted_args = [arg.format(**placeholders) for arg in args]
            command = ["python", script_path] + formatted_args

        # Log command
        formatted_cmd = format_command_for_logging(command)
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Running iFeature command:\n{formatted_cmd}",
            border="|",
            length=120,
        )
        logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

        # Run command
        subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600,
            check=True,
        )
        df = convert_tsv_to_df(
            input_tsv=output_tsv,
            feature_prefix=feature_type,
        )

        # Save processed DataFrame
        record_count = len(df)
        feature_count = df.shape[1] - 1  # exclude 'ID' column
        df.to_csv(output_csv, index=False)
        logger.log_with_borders(
            level=logging.INFO,
            message=(
                f"[ Feature Extraction Summary ]\n"
                f"▸ Feature Type     : '{feature_type}'\n"
                f"▸ Records (Samples): {record_count}\n"
                f"▸ Feature Columns  : {feature_count}\n"
            ),
            border="|",
            length=120,
        )
        logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Saved:\n'{output_csv}'",
            border="|",
            length=120,
        )
        logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

        # Delete temporary TSV
        os.remove(output_tsv)

        # Summary
        logger.log_with_borders(
            level=logging.INFO,
            message="\n".join(get_task_completion_message(start_time)),
            border="|",
            length=120,
        )
        logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

    except Exception:
        logger.exception(msg="Unexpected error in 'ifeature_encoding()'.")
        raise


def merge_csv_by_id_list(
    input_files: list[str],
    output_file: str,
    logger: CustomLogger,
) -> None:
    """
    Merge multiple CSV files based on the 'ID' column and save as a single merged file.

    Parameters
    ----------
    input_files : list of str
        List of CSV file paths to merge. All must contain an 'ID' column.
        The first file will serve as the base for merging.
    output_file : str
        Path to the final merged CSV file.
    logger : CustomLogger
        Logger instance for structured logging.

    Returns
    -------
    None
    """
    logger.info(f"/ Task: Merge {len(input_files)} feature CSV files → '{output_file}'")
    logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

    # Start timing
    start_time = time.time()

    try:

        # Validate input
        if len(input_files) < 2:
            logger.log_with_borders(
                level=logging.INFO,
                message="Only one file provided. Merge skipped.",
                border="|",
                length=120,
            )
            logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")
        else:
            dataframes = []
            for file_path in input_files:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: '{file_path}'")
                df = load_dataframe_by_columns(file_path)
                if df.empty:
                    raise ValueError(f"Error: '{file_path}' is empty.")
                dataframes.append(df)
                logger.log_with_borders(
                    level=logging.INFO,
                    message=f"Loaded: '{os.path.basename(file_path)}' (shape={df.shape})",
                    border="|",
                    length=120,
                )
            logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

            # Merge sequentially
            merged_df = dataframes[0]
            for df in dataframes[1:]:
                merged_df = merged_df.merge(df, on="ID", how="left")

            # Delete all intermediate files except merged_output
            for file_path in input_files:
                if file_path != output_file and os.path.exists(file_path):
                    os.remove(file_path)

            # Merge Summary
            logger.log_with_borders(
                level=logging.INFO,
                message=(
                    f"[ Merge Summary ]\n"
                    f"▸ Merged Files      : {len(input_files)}\n"
                    f"▸ Records (Samples) : {len(merged_df)}\n"
                    f"▸ Feature Columns   : {len(merged_df.columns)-1}"
                ),
                border="|",
                length=120,
            )
            logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

            # Save final merged CSV
            merged_df.to_csv(output_file, index=False)
            logger.log_with_borders(
                level=logging.INFO,
                message=f"Saved:\n'{output_file}'",
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
        logger.exception("Unexpected error in 'merge_csv_by_id_list()'.")
        raise


# ============================== Pipeline Entry Point ==============================
def run_ifeature_pipeline(
    base_path: str,
    logger: CustomLogger,
    **kwargs,
) -> None:
    """
    Executes iFeature feature encoding pipeline for selected feature types.

    Parameters
    ----------
    base_path : str
        Project root path (used to build file paths).
    logger : CustomLogger
        Logger instance for structured logging.
    **kwargs : dict
        Optional parameters such as input_dir, output_dir, identity.

    Returns
    -------
    None
    """
    # Start timing
    start_time = time.time()

    try:
        # -------------------- Retrieve input parameters (CLI or default) --------------------
        input_dir = kwargs.get(
            "ifeature_input_dir", os.path.join(base_path, "data/processed/")
        )
        output_dir = kwargs.get(
            "ifeature_output_dir", os.path.join(base_path, "data/processed/")
        )
        n_splits = kwargs.get("ifeature_n_splits", 5)
        config_path = kwargs.get(
            "ifeature_config_path",
            os.path.join(base_path, "configs/ifeature_config.yml"),
        )

        # Check input files exist
        for path in [config_path]:
            if not file_exists(file_path=path):
                raise FileNotFoundError(f"File not found: '{path}'")

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

        # Load feature configuration
        feature_config = load_yml_config(file_path=config_path)

        # Extract valid feature types (exclude special keys like "__global__")
        feature_types = [ft for ft in feature_config.keys() if not ft.startswith("__")]

        # Merge flag from YAML (__global__)
        merge_enabled = feature_config.get("__global__", {}).get(
            "merge_features", False
        )

        # Process each strain
        for _, (_, suffix) in enumerate(strains.items(), start=1):
            # Construct species-specific input/output folders
            species_input_dir = os.path.join(input_dir, suffix)
            species_output_dir = os.path.join(output_dir, suffix, "ifeature")
            folds_input_dir = os.path.join(species_input_dir, "folds")
            folds_output_dir = os.path.join(species_output_dir, "folds")
            for directory in [
                species_input_dir,
                species_output_dir,
                folds_input_dir,
                folds_output_dir,
            ]:
                if not directory_exists(dir_path=directory):
                    os.makedirs(name=directory)

            # Base splits (train/test)
            base_splits = {
                "train": os.path.join(species_input_dir, "train.fasta"),
                "test": os.path.join(species_input_dir, "test.fasta"),
            }

            # Fold splits (if folds directory exists)
            fold_splits = {}
            for i in range(1, n_splits + 1):
                fold_splits[f"train_fold{i}"] = os.path.join(
                    folds_input_dir, f"train_fold{i}.fasta"
                )
                fold_splits[f"val_fold{i}"] = os.path.join(
                    folds_input_dir, f"val_fold{i}.fasta"
                )

            # Combine all splits
            all_splits = {**base_splits, **fold_splits}
            for split_name, path in all_splits.items():
                if not file_exists(file_path=path):
                    raise FileNotFoundError(f"File not found: '{path}'")

                # Decide output dir
                if split_name.startswith("train_fold") or split_name.startswith(
                    "val_fold"
                ):
                    split_output_dir = folds_output_dir
                else:
                    split_output_dir = species_output_dir

                split_feature_files = []

                # Loop over all feature types
                for feature_type in feature_types:
                    # Construct output path
                    output_csv = os.path.join(
                        split_output_dir,
                        f"{split_name}_{feature_type}.csv",
                    )

                    # Run encoding
                    ifeature_encoding(
                        input_fasta=path,
                        output_csv=output_csv,
                        feature_type=feature_type,
                        feature_config=feature_config,
                        logger=logger,
                    )
                    split_feature_files.append(output_csv)
                    logger.add_spacer(level=logging.INFO, lines=1)

                # Optional Merge
                if merge_enabled and len(split_feature_files) > 1:
                    merged_output = os.path.join(split_output_dir, f"{split_name}.csv")
                    merge_csv_by_id_list(
                        input_files=split_feature_files,
                        output_file=merged_output,
                        logger=logger,
                    )
                    logger.add_spacer(level=logging.INFO, lines=1)

        # -------------------- Final summary --------------------
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
        logger.exception("Critical failure in 'run_ifeature_pipeline()'.")
        raise
