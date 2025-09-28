# pylint: disable=line-too-long, too-many-lines, import-error, wrong-import-position, singleton-comparison, too-many-locals, too-many-branches, too-many-statements, too-many-nested-blocks
"""
Collect and Preprocess AMP MIC Data from Multiple Databases
"""
# ============================== Standard Library Imports ==============================
import logging
import math
import os
import re
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
from utils.io_utils import directory_exists, file_exists, load_dataframe_by_columns
from utils.log_utils import get_pipeline_completion_message, get_task_completion_message


# ============================== Custom Functions ==============================
def filter_natural_aa(
    df: pd.DataFrame,
    logger: CustomLogger,
) -> pd.DataFrame:
    """
    Filter out sequences that do not contain only natural amino acids.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing biological sequences.
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing only sequences with natural amino acids.
    """
    logger.info("/ Task: Filter sequences containing only natural amino acids")
    logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

    # Start timing
    start_time = time.time()

    try:
        # Define the set of natural amino acids
        natural_aa_set = set("RHKDESTNQCGPAILMFWYV")

        # Initialize the 'Nature' column to indicate whether a sequence contains only natural amino acids
        df["Nature"] = False

        # Check each sequence to ensure it contains only natural amino acids
        df["Nature"] = df["Sequence"].apply(
            lambda seq: all(char in natural_aa_set for char in str(seq))
        )

        # Retain only sequences that passed the filter
        filtered_df = df[df["Nature"] == True].copy()

        # Log filtering statistics (total, kept, removed)
        total_count = df.shape[0]
        kept_count = filtered_df.shape[0]
        removed_count = total_count - kept_count
        logger.log_with_borders(
            level=logging.INFO,
            message=(
                f"[ Natural AA Filtering Summary ]\n"
                f"▸ Total sequences: {total_count}\n"
                f"▸ Kept (only natural AAs): {kept_count}\n"
                f"▸ Removed (contained non-natural AAs): {removed_count}"
            ),
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

        return filtered_df

    except Exception:
        logger.exception(msg="Unexpected error in 'filter_natural_aa()'.")
        raise


def calculate_sequence_length(
    df: pd.DataFrame,
    logger: CustomLogger,
) -> pd.DataFrame:
    """
    Calculate the length of each sequence in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the input sequences (must include 'Sequence' column).
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    pd.DataFrame
        A DataFrame with an added column 'Sequence Length' indicating the length of each sequence.
    """
    logger.info("/ Task: Calculate sequence lengths for each record")
    logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

    # Start timing
    start_time = time.time()

    try:
        # Calculate sequence lengths
        df["Sequence Length"] = df["Sequence"].apply(lambda seq: len(str(seq)))

        # Log summary statistics
        total_count = df.shape[0]
        min_len = int(df["Sequence Length"].min())
        max_len = int(df["Sequence Length"].max())
        bin_edges = list(range(0, max_len + 10, 10))
        bin_counts = (
            pd.cut(
                df["Sequence Length"],
                bins=bin_edges,
                right=True,
                include_lowest=True,
            )
            .value_counts()
            .sort_index()
        )
        bin_summary = "\n".join(
            [
                f"  - {int(abs(interval.left))}-{int(abs(interval.right))}: {count}"  # type: ignore
                for interval, count in bin_counts.items()
            ]
        )
        logger.log_with_borders(
            level=logging.INFO,
            message=(
                f"[ Sequence Length Calculation Summary ]\n"
                f"▸ Total sequences processed: {total_count}\n"
                f"▸ Min length: {min_len}\n"
                f"▸ Max length: {max_len}\n"
                f"▸ Distribution by length bins (step = 10):\n{bin_summary}"
            ),
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
        logger.exception(msg="Unexpected error in 'calculate_sequence_length()'.")
        raise


def sequence_length_cut(
    df: pd.DataFrame,
    lower_bound: int,
    upper_bound: int,
    logger: CustomLogger,
) -> pd.DataFrame:
    """
    Filter sequences by length, retaining only sequences with lengths between the lower and upper bounds.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the input sequences.
    lower_bound : int
        The lower bound for sequence length filtering.
    upper_bound : int
        The upper bound for sequence length filtering.
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing only sequences with lengths between the lower and upper bounds.
    """
    logger.info(
        f"/ Task: Filter sequences with lengths between {lower_bound} and {upper_bound}"
    )
    logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

    # Start timing
    start_time = time.time()

    try:
        # Apply filter
        filtered_df = df[
            (df["Sequence Length"] > lower_bound)
            & (df["Sequence Length"] < upper_bound)
        ].copy()

        # Log summary statistics
        total_count = df.shape[0]
        kept_count = filtered_df.shape[0]
        removed_count = total_count - kept_count
        logger.log_with_borders(
            level=logging.INFO,
            message=(
                f"[ Sequence Length Filtering Summary ]\n"
                f"▸ Total sequences processed: {total_count}\n"
                f"▸ Kept (lengths within '{lower_bound}-{upper_bound}'): {kept_count}\n"
                f"▸ Removed (outside range): {removed_count}"
            ),
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

        return filtered_df

    except Exception:
        logger.exception(msg="Unexpected error in 'sequence_length_cut()'.")
        raise


def calculate_molecular_weight(
    df: pd.DataFrame,
    logger: CustomLogger,
) -> pd.DataFrame:
    """
    Calculate the molecular weight of sequences.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the input sequences.
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    pd.DataFrame
        A DataFrame updated with the molecular weights.
    """
    logger.info("/ Task: Calculate molecular weights for each sequence")
    logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

    # Start timing
    start_time = time.time()

    try:
        # Define the atomic weight of each amino acid
        weights = {
            "A": 71.08,
            "C": 103.15,
            "D": 115.09,
            "E": 129.12,
            "F": 147.18,
            "G": 57.05,
            "H": 137.14,
            "I": 113.16,
            "K": 128.18,
            "L": 113.16,
            "M": 131.20,
            "N": 114.11,
            "P": 97.12,
            "Q": 128.13,
            "R": 156.19,
            "S": 87.08,
            "T": 101.11,
            "V": 99.13,
            "W": 186.22,
            "Y": 163.18,
        }

        # Calculate molecular weight for each sequence
        df["Molecular Weight"] = df["Sequence"].apply(
            lambda seq: sum(weights.get(aa, 0) for aa in str(seq))
        )

        # Log summary statistics
        total_count = df.shape[0]
        min_weight = round(df["Molecular Weight"].min(), 2)
        max_weight = round(df["Molecular Weight"].max(), 2)
        mean_weight = round(df["Molecular Weight"].mean(), 2)
        logger.log_with_borders(
            level=logging.INFO,
            message=(
                f"[ Molecular Weight Calculation Summary ]\n"
                f"▸ Total sequences processed: {total_count}\n"
                f"▸ Min molecular weight: {min_weight}\n"
                f"▸ Max molecular weight: {max_weight}\n"
                f"▸ Mean molecular weight: {mean_weight}"
            ),
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
        logger.exception(msg="Unexpected error in 'calculate_molecular_weight()'.")
        raise


def calculate_concentration(
    con: str,
    logger: CustomLogger,
) -> float:
    """
    Parse and calculate numeric concentration values from heterogeneous MIC string formats.

    This function supports multiple formats such as:
    - Range format: "10-20 μM"
    - Approximation format: "10 ± 2 μM"
    - Inequality format: "<5 μM" or ">10 μg/ml"
    - Simple numeric format: "15 μM"

    Parameters
    ----------
    con : str
        The string containing concentration information.
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    float
        The calculated concentration value.
    """
    try:
        # Handle various formats of the concentration string and extract the numeric value
        if "-" in con:
            # Case: MIC value given as a range (e.g., "10-20"), extract the first numeric value
            con_value = float(re.findall(r"-?\d+\.?\d*", con)[0])

        elif "±" in con:
            # Case: Range value specified with ± symbol
            a = float(re.findall(r"-?\d+\.?\d*", con)[0])  # First numeric value
            b = float(re.findall(r"-?\d+\.?\d*", con)[1])  # Second numeric value
            con_value = round(a - b, 3)  # Calculate difference

        elif "<" in con or ">" in con:
            # Case: Less than (<) or greater than (>) values
            con_value = float(re.findall(r"-?\d+\.?\d*", con)[0])

        else:
            # Case: Simple numeric value
            con_value = float(re.findall(r"-?\d+\.?\d*", con)[0])

        return con_value

    except Exception:
        logger.exception(msg="Unexpected error in 'calculate_concentration()'.")
        raise


def transform_concentration(
    new_df: pd.DataFrame,
    logger: CustomLogger,
) -> pd.DataFrame:
    """
    Transform the concentration values in the DataFrame based on the units.

    Parameters
    ----------
    new_df : pd.DataFrame
        A DataFrame containing concentration and unit information.
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    pd.DataFrame
        A DataFrame with updated concentration values.
    """
    logger.info(
        "/ Task: Transform 'concentration units' and compute logarithmic values"
    )
    logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

    # Start timing
    start_time = time.time()

    try:
        # Identify rows with unit 'μg/ml' and transform their MIC values
        mask = new_df["Unit"] == "μg/ml"
        converted_count = mask.sum()
        if converted_count > 0:
            new_df.loc[mask, "MIC Value"] = (
                pd.to_numeric(
                    new_df.loc[mask, "MIC Value"], errors="coerce"
                )  # ensure numeric
                / new_df.loc[mask, "Molecular Weight"]  # convert to molar
                * 1000  # scale μg/ml to μM
            )

        # Calculate the logarithmic MIC Value for all rows
        # If MIC Value is zero or negative, use `-inf` as a placeholder
        new_df["Log MIC Value"] = new_df["MIC Value"].apply(
            lambda x: (
                math.log10(x) if x > 0 else float("-inf")
            )  # or use 0 if appropriate
        )

        # Log summary statistics
        total_count = new_df.shape[0]
        invalid_count = (new_df["MIC Value"] <= 0).sum()
        logger.log_with_borders(
            level=logging.INFO,
            message=(
                f"[ Concentration Transformation Summary ]\n"
                f"▸ Total records processed: {total_count}\n"
                f"▸ Converted units (μg/ml → μM): {converted_count}\n"
                f"▸ Invalid MIC values (≤ 0): {invalid_count}"
            ),
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

        return new_df

    except Exception:
        logger.exception(msg="Unexpected error in 'transform_concentration()'.")
        raise


def filter_save_by_target_strains(
    input_df: pd.DataFrame,
    output_csv: str,
    keyword: str,
    logger: CustomLogger,
) -> None:
    """
    Filters, saves, and logs information for a specific target keyword from a DataFrame.

    Parameters
    ----------
    input_df : pd.DataFrame
        DataFrame to filter.
    output_csv : str
        File path for saving the filtered DataFrame.
    keyword : str
        Keyword used for filtering the 'Targets' column.
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    None
    """
    logger.info(
        f"/ Task: Extract and save AMP records targeting '{keyword}' from 'Targets' column"
    )
    logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

    # Start timing
    start_time = time.time()

    try:
        # Filter Records
        filtered_df = input_df[input_df["Targets"].str.contains(keyword)]

        # Log total and filtered counts
        total_count = input_df.shape[0]
        filtered_count = filtered_df.shape[0]
        percent = (filtered_count / total_count) * 100 if total_count > 0 else 0.0
        logger.log_with_borders(
            level=logging.INFO,
            message=(
                f"[ Filtering Summary ]\n"
                f"▸ Total records in input: {total_count}\n"
                f"▸ Filtered records matched '{keyword}': {filtered_count} "
                f"({percent:.2f}%)"
            ),
            border="|",
            length=120,
        )
        logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

        # Log the top N target counts with borders
        for target, count in filtered_df.Targets.value_counts()[:10].items():
            logger.log_with_borders(
                level=logging.INFO,
                message=f"{target}: {count}",
                border="|",
                length=120,
            )

        # Add a divider after logging
        logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

        # Save output
        filtered_df.to_csv(path_or_buf=output_csv, index=False)
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Saved:\n'{output_csv}'",
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
        logger.exception("Unexpected error in 'filter_save_by_target_strains()'")
        raise


# ============================== Pipeline Entry Point ==============================
def run_collect_dbaasp(base_path: str, logger: CustomLogger, **kwargs) -> None:
    """
    Processes DBAASP data, filters by strain, and saves the results.

    Parameters
    ----------
    base_path : str
        Project root path.
    logger : CustomLogger
        Logger instance for tracking progress.
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
        input_path = kwargs.get(
            "collect_input_path",
            os.path.join(base_path, "data/raw/peptides-complete1220.csv"),
        )
        output_dir = kwargs.get(
            "collect_output_dir",
            os.path.join(base_path, "data/interim"),
        )

        # Check input files exist
        for path in [input_path]:
            if not file_exists(file_path=path):
                raise FileNotFoundError(f"File not found: '{path}'")

        # Ensure output directories exist
        for directory in [output_dir]:
            if not directory_exists(dir_path=directory):
                os.makedirs(name=directory)

        # Load DBAASP data
        dbaasp_df = load_dataframe_by_columns(file_path=input_path)

        # Select data of monomer type
        dbaasp_df = dbaasp_df[dbaasp_df["COMPLEXITY"] == "Monomer"]

        # Set target activity concentration to 10000 where it is missing
        dbaasp_df.loc[
            dbaasp_df["TARGET ACTIVITY - CONCENTRATION"].isnull(),
            ["TARGET ACTIVITY - CONCENTRATION"],
        ] = "10000"

        # Rename columns
        dbaasp_df = dbaasp_df.rename(
            columns={
                "SEQUENCE": "Sequence",
                "TARGET ACTIVITY - TARGET SPECIES": "Targets",
                "TARGET ACTIVITY - CONCENTRATION": "MIC",
                "TARGET ACTIVITY - UNIT": "Unit",
            }
        )

        # Filter only natural amino acids
        dbaasp_df = filter_natural_aa(dbaasp_df, logger)
        logger.add_spacer(level=logging.INFO, lines=1)

        # Calculate sequence length
        dbaasp_df = calculate_sequence_length(dbaasp_df, logger)
        logger.add_spacer(level=logging.INFO, lines=1)

        # Filter by length range (you can pass in kwargs)
        lower_bound = 5
        upper_bound = 61
        dbaasp_df = sequence_length_cut(dbaasp_df, lower_bound, upper_bound, logger)
        logger.add_spacer(level=logging.INFO, lines=1)

        # Calculate molecular weight
        dbaasp_df = calculate_molecular_weight(dbaasp_df, logger)
        logger.add_spacer(level=logging.INFO, lines=1)

        # Calculate DBAASP concentration
        concentration = []

        # Normalize heterogeneous MIC strings into unified numeric values (μM or μg/ml)
        for con in dbaasp_df["MIC"]:
            if "-" in con:
                # If the concentration contains a '-', extract the first numeric value
                concentration.append(float(re.findall(r"-?\d+\.?\d*", con)[0]))
            elif "±" in con:
                # If the concentration contains '±', calculate the mean value
                a = float(re.findall(r"-?\d+\.?\d*", con)[0])
                b = float(re.findall(r"-?\d+\.?\d*", con)[1])
                c = round(a - b, 3)
                if c < 0:
                    c = a
                concentration.append(c)
            elif "<" in con:
                # If the concentration contains '<', extract the numeric value
                concentration.append(float(re.findall(r"-?\d+\.?\d*", con)[0]))
            elif ">" in con:
                # If the concentration contains '>', extract the numeric value
                concentration.append(float(re.findall(r"-?\d+\.?\d*", con)[0]))
            else:
                # Otherwise, directly extract the numeric value
                concentration.append(float(re.findall(r"-?\d+\.?\d*", con)[0]))

        # Store the new concentration data in the DataFrame
        dbaasp_df["MIC Value"] = concentration

        # Calculate new concentration units (if the original unit is µg/ml)
        new_con = (
            dbaasp_df.loc[dbaasp_df["Unit"] == "µg/ml"]["MIC Value"]
            / dbaasp_df.loc[dbaasp_df["Unit"] == "µg/ml"]["Molecular Weight"]
        ) * 1000

        # Update the concentration data in the DataFrame
        dbaasp_df.loc[dbaasp_df["Unit"] == "µg/ml", ["MIC Value"]] = new_con

        # Calculate the logarithm of the concentration values
        dbaasp_df["Log MIC Value"] = dbaasp_df["MIC Value"].map(
            lambda x: math.log10(x) if x > 0 else 0
        )

        # Initialize an empty DataFrame to store DBAASP data
        dbaasp = pd.DataFrame()

        # Mapping of full strain names to their corresponding suffixes
        strains = {
            "Escherichia coli": "EC",
            "Pseudomonas aeruginosa": "PA",
            "Staphylococcus aureus": "SA",
        }

        # Iterate over each strain
        for _, (strain, _) in enumerate(strains.items(), start=1):
            # Select rows related to the current strain
            new_row = dbaasp_df[dbaasp_df["Targets"].str.contains(strain) == True]

            # Append the selected rows to the DBAASP DataFrame
            dbaasp = pd.concat([dbaasp, new_row], ignore_index=True)

        # Add 'DBAASP_' prefix to the IDs
        dbaasp["ID"] = dbaasp["ID"].map(lambda x: f"DBAASP_{x}")

        # Select rows where the activity measure group is 'MIC'
        dbaasp = dbaasp[dbaasp["TARGET ACTIVITY - ACTIVITY MEASURE GROUP"] == "MIC"]

        # Select the required columns
        dbaasp = dbaasp[
            [
                "ID",
                "Sequence",
                "Targets",
                "Sequence Length",
                "Molecular Weight",
                "MIC",
                "MIC Value",
                "Unit",
                "Log MIC Value",
            ]
        ]

        # Filter and save DBAASP data
        for _, (strain, suffix) in enumerate(strains.items(), start=1):
            output_csv = os.path.join(output_dir, f"{suffix}_DBAASP.csv")
            filter_save_by_target_strains(
                input_df=dbaasp,
                output_csv=output_csv,
                keyword=strain,
                logger=logger,
            )
            logger.add_spacer(level=logging.INFO, lines=1)

    except Exception:
        logger.exception("Critical failure in 'run_collect_dbaasp()'")
        raise

    # Final summary block
    logger.info("[ 'Pipeline Execution Summary' ]")
    logger.add_divider(level=logging.INFO, length=120, border="+", fill="=")
    logger.log_with_borders(
        level=logging.INFO,
        message="\n".join(get_pipeline_completion_message(start_time)),
        border="║",
        length=120,
    )
    logger.add_divider(level=logging.INFO, length=120, border="+", fill="=")


def run_collect_dbamp(base_path: str, logger: CustomLogger, **kwargs) -> None:
    """
    Processes dbAMP data, filters by strain, and saves the results.

    Parameters
    ----------
    base_path : str
        Project root path.
    logger : CustomLogger
        Logger instance for tracking progress.
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
        input_path = kwargs.get(
            "collect_input_path",
            os.path.join(base_path, "data/raw/dbAMP3_pepinfo.xlsx"),
        )
        output_dir = kwargs.get(
            "collect_output_dir",
            os.path.join(base_path, "data/interim"),
        )

        # Check input files exist
        for path in [input_path]:
            if not file_exists(file_path=path):
                raise FileNotFoundError(f"File not found: '{path}'")

        # Ensure output directories exist
        for directory in [output_dir]:
            if not directory_exists(dir_path=directory):
                os.makedirs(name=directory)

        # Load dbAMP data
        dbamp_df = load_dataframe_by_columns(file_path=input_path)

        # Rename columns
        dbamp_df = dbamp_df.rename(
            columns={
                "Seq": "Sequence",
            }
        )

        # Remove rows where the 'Targets' column is null
        dbamp_df = dbamp_df[dbamp_df["Targets"].notnull()]

        # Filter only natural amino acids
        dbamp_df = filter_natural_aa(dbamp_df, logger)
        logger.add_spacer(level=logging.INFO, lines=1)

        # Calculate sequence length
        dbamp_df = calculate_sequence_length(dbamp_df, logger)
        logger.add_spacer(level=logging.INFO, lines=1)

        # Filter by length range (you can pass in kwargs)
        lower_bound = 5
        upper_bound = 61
        dbamp_df = sequence_length_cut(dbamp_df, lower_bound, upper_bound, logger)
        logger.add_spacer(level=logging.INFO, lines=1)

        # Calculate molecular weight
        dbamp_df = calculate_molecular_weight(dbamp_df, logger)
        logger.add_spacer(level=logging.INFO, lines=1)

        # Reset the index of the DataFrame
        dbamp_df = dbamp_df.reset_index(drop=True)

        # Initialize an empty DataFrame to store the results
        dbamp = pd.DataFrame(
            columns=[
                "ID",
                "Sequence",
                "Targets",
                "Sequence Length",
                "Molecular Weight",
                "MIC",
                "MIC Value",
                "Unit",
                "Log MIC Value",
            ]
        )

        # Mapping of full strain names to their corresponding suffixes
        strains = {
            "Escherichia coli": "EC",
            "Pseudomonas aeruginosa": "PA",
            "Staphylococcus aureus": "SA",
        }

        # Define a regular expression to match MIC values
        regex = r"\(MIC=\S+\)"

        # Iterate over each strain
        for _, (strain, _) in enumerate(strains.items(), start=1):
            for i in range(len(dbamp_df)):
                # Extract information related to the strain
                res1 = [
                    string
                    for string in dbamp_df["Targets"][i].split("&&")
                    if strain in string
                ]
                # If multiple strain-related MIC values are found, prioritize the first one
                if len(res1) >= 2:
                    final = []
                    # Iterate over each matched result
                    for _, item in enumerate(res1):
                        if "MIC" not in item:
                            continue
                        # Find the MIC value using the regular expression and calculate the concentration
                        res2 = re.search(regex, item)
                        if res2 is None:
                            continue
                        res3 = calculate_concentration(con=res2.group(0), logger=logger)
                        x = (res2.group(0), res3)
                        final.append(x)
                    # If no valid MIC value is found, continue to the next iteration
                    if not final:
                        continue
                    # sorted_by_second = sorted(final, key=lambda tup: tup[1])
                    res2 = final[0]
                    res3 = final[0][1]
                else:
                    if len(res1) == 0:
                        continue

                    if "MIC" not in res1[0]:
                        continue

                    res2 = re.search(regex, res1[0])
                    if res2 is None:
                        continue

                    res3 = calculate_concentration(con=res2.group(0), logger=logger)

                # Determine the unit of concentration
                unit = "µM" if "µM" in res2[0] else "μg/ml"
                # Construct a new row of data
                new_row = {
                    "ID": dbamp_df.loc[i].dbAMP_ID,
                    "Sequence": dbamp_df.loc[i].Sequence,
                    "Targets": strain,
                    "Sequence Length": dbamp_df.loc[i]["Sequence Length"],
                    "Molecular Weight": dbamp_df.loc[i]["Molecular Weight"],
                    "MIC": res2[0],
                    "MIC Value": res3,
                    "Unit": unit,
                }

                # Add the new row to the DataFrame
                dbamp = pd.concat([dbamp, pd.DataFrame([new_row])], ignore_index=True)

        # Transform the concentration values' units and calculate their logarithmic values
        dbamp = transform_concentration(new_df=dbamp, logger=logger)
        logger.add_spacer(level=logging.INFO, lines=1)

        # Filter and save dbAMP data
        for _, (strain, suffix) in enumerate(strains.items(), start=1):
            output_csv = os.path.join(output_dir, f"{suffix}_dbAMP.csv")

            filter_save_by_target_strains(
                input_df=dbamp,
                output_csv=output_csv,
                keyword=strain,
                logger=logger,
            )
            logger.add_spacer(level=logging.INFO, lines=1)

    except Exception:
        logger.exception("Critical failure in 'run_collect_dbamp()'")
        raise

    # Final summary block
    logger.info("[ 'Pipeline Execution Summary' ]")
    logger.add_divider(level=logging.INFO, length=120, border="+", fill="=")
    logger.log_with_borders(
        level=logging.INFO,
        message="\n".join(get_pipeline_completion_message(start_time)),
        border="║",
        length=120,
    )
    logger.add_divider(level=logging.INFO, length=120, border="+", fill="=")


def run_collect_dramp(base_path: str, logger: CustomLogger, **kwargs) -> None:
    """
    Processes Dramp data, filters by strain, and saves the results.

    Parameters
    ----------
    base_path : str
        Project root path.
    logger : CustomLogger
        Logger instance for tracking progress.
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
        input_path = kwargs.get(
            "collect_input_path",
            os.path.join(base_path, "data/raw/DRAMP.xlsx"),
        )
        output_dir = kwargs.get(
            "collect_output_dir",
            os.path.join(base_path, "data/interim"),
        )

        # Check input files exist
        for path in [input_path]:
            if not file_exists(file_path=path):
                raise FileNotFoundError(f"File not found: '{path}'")

        # Ensure output directories exist
        for directory in [output_dir]:
            if not directory_exists(dir_path=directory):
                os.makedirs(name=directory)

        # Load Dramp data
        dramp_df = load_dataframe_by_columns(file_path=input_path)

        # Filter only natural amino acids
        dramp_df = filter_natural_aa(dramp_df, logger)
        logger.add_spacer(level=logging.INFO, lines=1)

        # Calculate sequence length
        dramp_df = calculate_sequence_length(dramp_df, logger)
        logger.add_spacer(level=logging.INFO, lines=1)

        # Filter by length range (you can pass in kwargs)
        lower_bound = 5
        upper_bound = 61
        dramp_df = sequence_length_cut(dramp_df, lower_bound, upper_bound, logger)
        logger.add_spacer(level=logging.INFO, lines=1)

        # Calculate molecular weight
        dramp_df = calculate_molecular_weight(dramp_df, logger)
        logger.add_spacer(level=logging.INFO, lines=1)

        # Reset the index of the DataFrame
        dramp_df = dramp_df.reset_index(drop=True)

        # Insert an underscore '_' between "DRAMP" and the numbers in each value of the 'DRAMP_ID' column
        dramp_df["DRAMP_ID"] = dramp_df["DRAMP_ID"].str.replace(
            "DRAMP", "DRAMP_", regex=True
        )

        # Ensure 'Target_Organism' is a string (avoid TypeError in re.split)
        dramp_df["Target_Organism"] = dramp_df["Target_Organism"].fillna("").astype(str)

        # Initialize an empty DataFrame to store the results
        dramp = pd.DataFrame(
            columns=[
                "ID",
                "Sequence",
                "Targets",
                "Sequence Length",
                "Molecular Weight",
                "MIC",
                "MIC Value",
                "Unit",
                "Log MIC Value",
            ]
        )

        # Mapping of full strain names to their corresponding suffixes
        strains = {
            "Escherichia coli": "EC",
            "Pseudomonas aeruginosa": "PA",
            "Staphylococcus aureus": "SA",
        }

        # Define a regular expression to match MIC values
        regex = r"\(MIC=\S+\)"

        # Iterate over each strain
        for _, (strain, _) in enumerate(strains.items(), start=1):
            for _, row in dramp_df.iterrows():
                # Extract information related to the strain
                res1 = [
                    string
                    for string in re.split(",| ;|##", row["Target_Organism"])
                    if strain in string
                ]
                # If no strain-related information is found, continue to the next iteration
                if len(res1) == 0:
                    continue
                # If strain-related information is found
                if len(res1) >= 2:
                    final = []
                    for _, item in enumerate(res1):
                        if "MIC" not in item:
                            continue
                        # Find the MIC value using the regular expression and calculate the concentration
                        res2 = re.search(regex, item)
                        if res2 is None:
                            continue
                        res3 = calculate_concentration(con=res2.group(0), logger=logger)
                        x = (res2.group(0), res3)
                        final.append(x)
                    if not final:
                        continue
                    # sorted_by_second = sorted(final, key=lambda tup: tup[1])
                    res2 = final[0]
                    res3 = final[0][1]
                else:
                    if len(res1) == 0:
                        continue

                    if "MIC" not in res1[0]:
                        continue

                    res2 = re.search(regex, res1[0])
                    if res2 is None:
                        continue

                    res3 = calculate_concentration(con=res2.group(0), logger=logger)

                # Determine the unit of concentration
                unit = (
                    "μM"
                    if "μM" in res2[0]
                    else (
                        "μg/ml"
                        if ("μg/ml" in res2[0]) or ("μg/mL" in res2[0])
                        else "NONE"
                    )
                )
                # Construct a new row of data
                new_row = {
                    "ID": row.DRAMP_ID,
                    "Sequence": row.Sequence,
                    "Targets": strain,
                    "Sequence Length": row["Sequence Length"],
                    "Molecular Weight": row["Molecular Weight"],
                    "MIC": res2[0],
                    "MIC Value": res3,
                    "Unit": unit,
                }
                # Add the new row to the DataFrame
                dramp = pd.concat([dramp, pd.DataFrame([new_row])], ignore_index=True)

        # Remove rows where the concentration unit could not be determined
        dramp = dramp[dramp["Unit"] != "NONE"]

        # Transform the concentration values' units and calculate their logarithmic values
        dramp = transform_concentration(new_df=dramp, logger=logger)
        logger.add_spacer(level=logging.INFO, lines=1)

        # Filter and save dbAMP data
        for _, (strain, suffix) in enumerate(strains.items(), start=1):
            output_csv = os.path.join(output_dir, f"{suffix}_Dramp.csv")
            filter_save_by_target_strains(
                input_df=dramp,
                output_csv=output_csv,
                keyword=strain,
                logger=logger,
            )
            logger.add_spacer(level=logging.INFO, lines=1)

    except Exception:
        logger.exception("Critical failure in 'run_collect_dramp()'")
        raise

    # Final summary block
    logger.info("[ 'Pipeline Execution Summary' ]")
    logger.add_divider(level=logging.INFO, length=120, border="+", fill="=")
    logger.log_with_borders(
        level=logging.INFO,
        message="\n".join(get_pipeline_completion_message(start_time)),
        border="║",
        length=120,
    )
    logger.add_divider(level=logging.INFO, length=120, border="+", fill="=")


def run_merge_all_sources(base_path: str, logger: CustomLogger, **kwargs) -> None:
    """
    Merge processed data from DBAASP, dbAMP, and DRAMP sources by target strain.

    Parameters
    ----------
    base_path : str
        Project root path.
    logger : CustomLogger
        Logger instance for tracking progress and errors.
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
        io_dir = kwargs.get(
            "collect_output_dir",
            os.path.join(base_path, "data/interim"),
        )

        # Ensure output directories exist
        for directory in [io_dir]:
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

            # Start timing
            start_time_task = time.time()
            logger.info(msg=f"/ Task: Merge and save all data targeting '{strain}'")
            logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

            # Build file paths dynamically
            dbaasp_path = os.path.join(io_dir, f"{suffix}_DBAASP.csv")
            dbamp_path = os.path.join(io_dir, f"{suffix}_dbAMP.csv")
            dramp_path = os.path.join(io_dir, f"{suffix}_Dramp.csv")
            output_path = os.path.join(io_dir, f"{suffix}_All.csv")

            # Check input files exist
            for path in [dbaasp_path, dbamp_path, dramp_path]:
                if not file_exists(file_path=path):
                    raise FileNotFoundError(f"File not found: '{path}'")

            # Load Data
            df_dbaasp = load_dataframe_by_columns(file_path=dbaasp_path)
            df_dbamp = load_dataframe_by_columns(file_path=dbamp_path)
            df_dramp = load_dataframe_by_columns(file_path=dramp_path)

            # Log Dataset Shapes
            logger.log_with_borders(
                level=logging.INFO,
                message=(
                    f"[ Dataset Summary for '{strain}' ]\n"
                    f"▸ DBAASP: {df_dbaasp.shape[0]} rows\n"
                    f"▸ dbAMP : {df_dbamp.shape[0]} rows\n"
                    f"▸ DRAMP : {df_dramp.shape[0]} rows"
                ),
                border="|",
                length=120,
            )

            # Concatenate Data
            merged_df = pd.concat(
                [df_dbaasp, df_dbamp, df_dramp], axis=0, ignore_index=True
            )
            logger.log_with_borders(
                level=logging.INFO,
                message=f"▸ Concatenated dataset: {merged_df.shape[0]} total rows",
                border="|",
                length=120,
            )
            logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

            # Save Output
            merged_df.to_csv(path_or_buf=output_path, index=False)
            logger.log_with_borders(
                level=logging.INFO,
                message=f"Saved:\n'{output_path}'",
                border="|",
                length=120,
            )
            logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

            # Summary
            logger.log_with_borders(
                level=logging.INFO,
                message="\n".join(get_task_completion_message(start_time_task)),
                border="|",
                length=120,
            )
            logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")
            logger.add_spacer(level=logging.INFO, lines=1)

        # Final summary block
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
        logger.exception("Critical failure in 'run_merge_all_sources()'")
        raise
