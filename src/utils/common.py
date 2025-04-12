# pylint: disable=
"""
Common Utility Functions for ANIA

This module provides shared helper functions used across multiple components
of the AMP-MIC pipeline, including configuration parsing, hyperparameter access,
and feature extraction.

Main functionalities:
- `read_json_config()`: Reads and parses a JSON configuration file.
"""
# ============================== Standard Library Imports ==============================
import json
from typing import Dict, List, Set

# ============================== Third-Party Library Imports ==============================
import pandas as pd


# ============================== Custom Function ==============================
def write_fasta_file(df: pd.DataFrame, output_fasta: str) -> None:
    """
    Write sequence data to a FASTA file.

    Each sequence is written with an identifier and its corresponding sequence.
    The identifier is stripped of leading/trailing spaces and all spaces are
    replaced with underscores `_`.

    The sequence itself is also stripped of spaces to ensure correct FASTA format.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing 'ID' and 'Sequence' columns.
    output_fasta : str
        The file path to save the FASTA file.

    Returns
    -------
    None
    """
    try:
        # Check whether required columns exist in the DataFrame
        check_required_columns(df, required_columns=["ID", "Sequence"])

        # Open the output file for writing in FASTA format
        with open(output_fasta, "w", encoding="utf-8") as f:
            # Iterate over each row in the DataFrame
            for _, row in df.iterrows():
                # Clean sequence and identifier
                sequence = row["Sequence"].strip().replace(" ", "")
                identifier = row["ID"].strip().replace(" ", "_")
                # Write in FASTA format
                f.write(f">{identifier}\n{sequence}\n")

    except Exception as e:
        raise RuntimeError(f"Unexpected error in 'write_fasta_file()': {str(e)}") from e


def check_required_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    """
    Check whether the input DataFrame contains all required columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    required_columns : List[str]
        List of column names that must be present in the DataFrame.

    Returns
    -------
    None
    """
    missing_columns: Set[str] = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def read_json_config(config_path: str) -> Dict:
    """
    Read and parse a JSON configuration file.

    Parameters
    ----------
    config_path : str
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed JSON content.
    """
    try:
        # Load JSON file
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        return config

    except FileNotFoundError as exc:
        raise FileNotFoundError("FileNotFoundError in 'read_json_config()'.") from exc

    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(
            msg="JSONDecodeError in 'read_json_config()'.",
            doc=exc.doc,
            pos=exc.pos,
        ) from exc

    except Exception as exc:
        raise RuntimeError("Unexpected error in 'read_json_config()'.") from exc
