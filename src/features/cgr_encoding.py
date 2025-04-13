# pylint: disable=line-too-long, import-error, wrong-import-position, too-many-locals
"""
This module provides functionality for encoding antimicrobial peptide (AMP) sequences into
Chaos Game Representation (CGR) features for downstream machine learning tasks.

The CGR encoding process converts symbolic amino acid sequences into numerical 2D matrices
at varying resolutions (e.g., 8x8, 16x16, etc.), effectively capturing spatial patterns
and sequence composition. The resulting features are flattened and stored as vectors.

The processing pipeline:
1. Reads FASTA-formatted sequence files (train/test split per strain).
2. Uses the R package `kaos` (via rpy2) to compute CGR matrices at multiple resolutions.
3. Flattens CGR matrices into 1D vectors and writes them to CSV files.
4. Merges CGR features with corresponding dataset CSVs using sequence IDs.

The core functions in this module are:
- `encode_fasta_to_cgr()`: Performs CGR encoding via `kaos::cgr()` and outputs features as CSV.
- `merge_csv_by_id()`: Merges CGR features into the main dataset using the 'ID' column.
- `run_cgr_pipeline()`: Main pipeline entry point that orchestrates CGR encoding for each strain and dataset type.

This module is typically invoked through the main CLI interface (`main.py --stage cgr`).
"""
# ============================== Standard Library Imports ==============================
import logging
import os
import sys

# ============================== Third-Party Library Imports ==============================
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector

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


# ============================== Custom Function ==============================
def compute_cgr(
    sequences: list,
    resolution: int,
) -> tuple[list, list]:
    """
    Compute CGR features for a list of sequences.

    Parameters
    ----------
    sequences : list
        List of sequences to encode.
    resolution : int
        Resolution for the CGR computation (e.g., 16 for a 16x16 matrix).

    Returns
    -------
    tuple
        - cgr_vectors: List of flattened CGR vectors (one per sequence).
        - cgr_results: List of CGR result objects (for visualization or further processing).
    """
    # Initialize the kaos R package
    kaos = importr("kaos")

    # Process sequences for CGR encoding
    cgr_vectors = []
    cgr_results = []
    for _, seq in enumerate(sequences):
        r_sequence = StrVector(list(seq))

        # Compute CGR matrix
        cgr_result = kaos.cgr(r_sequence, seq_base="AMINO", res=resolution)
        cgr_matrix = np.array(ro.r("matrix")(cgr_result.rx2("matrix")))

        # Flatten the CGR matrix into a 1D vector
        flattened_vector = cgr_matrix.flatten(order="F")
        cgr_vectors.append(flattened_vector)
        cgr_results.append(cgr_result)

    return cgr_vectors, cgr_results


def map_kmers(
    sequences: list, cgr_results: list, resolution: int, k: int = 3
) -> list[dict]:
    """
    Map k-mers from multiple sequences to their corresponding CGR pixels.

    Parameters
    ----------
    sequences : list
        List of amino acid sequences.
    cgr_results : list
        List of CGR result objects from kaos::cgr().
    resolution : int
        Grid resolution of the CGR (e.g., 8, 16, etc.).
    k : int
        Length of k-mer.

    Returns
    -------
    list of dict
        A list where each element is a dictionary mapping (row, col) to list of k-mers
        for each sequence.
    """

    def map_coords(x_coords, y_coords, res):
        x_pixel = np.ceil((x_coords + 1) * res / 2).astype(int) - 1
        y_pixel = np.ceil((y_coords + 1) * res / 2).astype(int) - 1
        return y_pixel, x_pixel

    all_pixel_kmer_maps = []
    for seq, cgr_result in zip(sequences, cgr_results):
        x = np.array(cgr_result.rx2("x"))
        y = np.array(cgr_result.rx2("y"))
        row_idx, col_idx = map_coords(x, y, resolution)

        pixel_to_kmers = {}
        for i in range(len(seq) - k + 1):
            kmer = seq[i : i + k]
            r = row_idx[i + k - 1]
            c = col_idx[i + k - 1]
            pixel = (r, c)
            if pixel not in pixel_to_kmers:
                pixel_to_kmers[pixel] = []
            pixel_to_kmers[pixel].append(kmer)

        all_pixel_kmer_maps.append(pixel_to_kmers)

    return all_pixel_kmer_maps


def compute_props(
    all_pixel_kmer_maps: list[dict],
    property_table: pd.DataFrame,
    property_id: str,
) -> list[dict]:
    """
    Compute pixelwise property values based on **summed property of unique k-mers** per pixel.

    Parameters
    ----------
    all_pixel_kmer_maps : list of dict
        List of pixel-to-kmer mappings for each sequence.
    property_table : pd.DataFrame
        AAindex-style table with amino acids as index.
    property_id : str
        The column to use as property value.

    Returns
    -------
    list of dict
        One dictionary per sequence, mapping (row, col) to final property value.
    """
    all_property_maps = []

    for pixel_kmers in all_pixel_kmer_maps:
        pixel_map = {}

        for pixel, kmer_list in pixel_kmers.items():
            unique_kmers = set(kmer_list)
            kmer_sums = []

            for kmer in unique_kmers:
                try:
                    aa_sum = sum(property_table.loc[aa, property_id] for aa in kmer)  # type: ignore
                    kmer_sums.append(aa_sum)
                except KeyError:
                    continue  # ignore unknown AA

            pixel_map[pixel] = np.mean(kmer_sums) if kmer_sums else 0.0

        all_property_maps.append(pixel_map)

    return all_property_maps


def encode_fasta_to_cgr(
    input_fasta: str,
    output_file: str,
    resolution: int,
    logger: CustomLogger,
) -> None:
    """
    Compute CGR encoding for sequences in a FASTA file and save results as a CSV.

    Parameters
    ----------
    input_fasta : str
        Path to the input FASTA file.
    output_file : str
        Path to the final output CSV file.
    resolution : int
        Resolution for the CGR computation.
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    None
    """
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)

        # Log the start of CGR computation
        logger.info(
            msg=f"/ Task: Starting 'CGR' feature extraction. (resolution = {resolution})"
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Read sequences from FASTA
        sequence_ids = []
        sequences = []
        with open(input_fasta, "r", encoding="utf-8") as file:
            sequence = ""
            seq_id = None
            for line in file:
                line = line.strip()
                if line.startswith(">"):  # Header line
                    if sequence and seq_id:  # Store previous sequence and ID
                        sequence_ids.append(seq_id)
                        sequences.append(sequence)
                        sequence = ""
                    seq_id = line[1:]  # Remove '>' and store as ID
                else:
                    sequence += line
            if sequence and seq_id:
                sequence_ids.append(seq_id)
                sequences.append(sequence)  # Add last sequence

        # Ensure ID and Sequence count match
        if len(sequence_ids) != len(sequences):
            raise ValueError("Mismatch between FASTA IDs and sequences.")

        # Check if any sequences were read
        if len(sequences) == 0:
            raise ValueError(f"Error: No valid sequences found in '{input_fasta}'.")

        # Compute CGR features
        cgr_vectors, cgr_results = compute_cgr(
            sequences=sequences,
            resolution=resolution,
        )

        # Create DataFrame for CGR features
        cgr_df = pd.DataFrame(
            cgr_vectors,
            columns=[
                f"CGR(resolution={resolution}) | {i+1}-{j+1}"
                for i in range(resolution)
                for j in range(resolution)
            ],
        )
        cgr_df.insert(0, "ID", sequence_ids)  # ID column consistent with iFeature

        # Save the CGR features to a CSV file
        cgr_df.to_csv(output_file, index=False)

        # Log successful completion
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Successfully saved 'CGR' results:\n'{output_file}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

    except Exception:
        logger.exception(msg="Unexpected error in 'encode_fasta_to_cgr()'.")
        raise


def encode_fasta_to_cgr_multi(
    input_fasta: str,
    output_file: str,
    resolution: int,
    logger: CustomLogger,
) -> None:
    """
    Compute CGR encoding for sequences in a FASTA file and save results as a CSV.

    Parameters
    ----------
    input_fasta : str
        Path to the input FASTA file.
    output_file : str
        Path to the final output CSV file.
    resolution : int
        Resolution for the CGR computation.
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    None
    """
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)

        # Log the start of CGR computation
        logger.info(
            msg=f"/ Task: Starting 'CGR' feature extraction. (resolution = {resolution})"
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Read sequences from FASTA
        sequence_ids = []
        sequences = []
        with open(input_fasta, "r", encoding="utf-8") as file:
            sequence = ""
            seq_id = None
            for line in file:
                line = line.strip()
                if line.startswith(">"):  # Header line
                    if sequence and seq_id:  # Store previous sequence and ID
                        sequence_ids.append(seq_id)
                        sequences.append(sequence)
                        sequence = ""
                    seq_id = line[1:]  # Remove '>' and store as ID
                else:
                    sequence += line
            if sequence and seq_id:
                sequence_ids.append(seq_id)
                sequences.append(sequence)  # Add last sequence

        # Ensure ID and Sequence count match
        if len(sequence_ids) != len(sequences):
            raise ValueError("Mismatch between FASTA IDs and sequences.")

        # Check if any sequences were read
        if len(sequences) == 0:
            raise ValueError(f"Error: No valid sequences found in '{input_fasta}'.")

        # Compute CGR features
        cgr_vectors, cgr_results = compute_cgr(
            sequences=sequences,
            resolution=resolution,
        )

        # Create DataFrame for CGR features
        cgr_df = pd.DataFrame(
            cgr_vectors,
            columns=[
                f"CGR(resolution={resolution}) | {i+1}-{j+1}"
                for i in range(resolution)
                for j in range(resolution)
            ],
        )
        cgr_df.insert(0, "ID", sequence_ids)  # ID column consistent with iFeature

        # Compute k-mer to pixel mappings
        pixel_kmer_maps = map_kmers(sequences, cgr_results, resolution, k=3)

        # Load AAindex property table
        property_table = pd.read_csv(
            os.path.join(BASE_PATH, "configs/AAindex_properties.csv"),
            index_col="AminoAcid",
        )

        # Step 3: Compute property maps (summed unique k-mer mode)
        for property_id in [
            "ARGP820101",
            "CHAM830107",
            "FAUJ880103",
            "GRAR740102",
            "JANJ780101",
            "KYTJ820101",
            "NAKH920104",
            "ROSM880102",
            "WERD780104",
            "ZIMJ680101",
        ]:
            property_maps = compute_props(
                pixel_kmer_maps, property_table, property_id=property_id
            )

            # Step 4: Convert property maps into vectors (flatten to match CGR format)
            property_vectors = []
            for prop_map in property_maps:
                matrix = np.zeros((resolution, resolution))
                for (r, c), val in prop_map.items():
                    matrix[r, c] = val
                property_vectors.append(matrix.flatten(order="C"))

            # Step 5: Convert to DataFrame and merge
            prop_df = pd.DataFrame(
                property_vectors,
                columns=[
                    f"PROP({property_id}) | {i+1}-{j+1}"
                    for i in range(resolution)
                    for j in range(resolution)
                ],
            )
            cgr_df = pd.concat([cgr_df, prop_df], axis=1)

        # Save the CGR features to a CSV file
        cgr_df.to_csv(output_file, index=False)

        # Log successful completion
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Successfully saved 'CGR' results:\n'{output_file}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

    except Exception:
        logger.exception(msg="Unexpected error in 'encode_fasta_to_cgr()'.")
        raise


def merge_csv_by_id(
    file1: str,
    file2: str,
    logger: CustomLogger,
) -> None:
    """
    Merge two CSV files based on the 'ID' column and overwrite the first file.

    Parameters
    ----------
    file1 : str
        Path to the first CSV file (main data).
    file2 : str
        Path to the second CSV file (feature data).
    logger : CustomLogger
        Logger instance for recording the process.

    Returns
    -------
    None
    """
    try:
        # Log the start of merging
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message="Starting merge of based on 'ID' column.",
            border="|",
            length=100,
        )

        # Load the first CSV file (main data)
        df1 = pd.read_csv(file1)

        # Load the second CSV file (feature data)
        df2 = pd.read_csv(file2)

        # Check if the files are empty
        if df1.empty:
            raise ValueError(f"Error: '{file1}' is empty.")
        if df2.empty:
            raise ValueError(f"Error: '{file2}' is empty.")

        # Ensure both files contain the 'ID' column
        if "ID" not in df1.columns:
            raise KeyError(f"Missing 'ID' column in '{file1}'.")
        if "ID" not in df2.columns:
            raise KeyError(f"Missing 'ID' column in '{file2}'.")

        # Merge the two dataframes based on the 'ID' column (left join)
        merged_df = df1.merge(df2, on="ID", how="left")

        # Save the merged dataframe back to file1 (overwrite)
        merged_df.to_csv(file1, index=False)

        # Log the successful merge
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Successfully merged '{file2}' into '{file1}'.",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Delete file2 after successful merge
        os.remove(file2)
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Deleted temporary CSV file:\n'{file2}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

    except Exception:
        logger.exception(msg="Unexpected error in 'merge_csv_by_id()'.")
        raise


# ============================== Pipeline Entry Point ==============================
def run_cgr_pipeline(base_path: str, logger: CustomLogger) -> None:
    """
    Entry point for the CGR-based encoding pipeline.

    This function encodes AMP sequences into Chaos Game Representation (CGR) features at multiple
    resolutions (e.g., 8×8, 16×16, 32×32, 64×64). It processes both train and test sets for each
    defined bacterial strain. The extracted CGR features are saved and merged into the corresponding
    dataset CSV for downstream modeling.

    For each (strain, dataset type) pair, the following steps are executed:
    1. Load the FASTA-formatted sequence file.
    2. Encode the sequence into CGR features for multiple image resolutions.
    3. Save each resolution-specific CGR feature file as a CSV.
    4. Merge the CGR features into the main CSV dataset (by ID).

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

            # Define input and output paths
            input_fasta_file = os.path.join(
                base_path, f"data/processed/all/{suffix}.fasta"
            )
            output_file = os.path.join(base_path, f"data/processed/all/{suffix}.csv")
            output_cgr_file = os.path.join(
                base_path, f"data/processed/all/{suffix}_cgr.csv"
            )

            # Perform CGR feature extraction
            encode_fasta_to_cgr_multi(
                input_fasta=input_fasta_file,
                output_file=output_cgr_file,
                resolution=16,
                logger=logger,
            )

            # Insert a blank line in the log for readability
            logger.add_spacer(level=logging.INFO, lines=1)

            # Merge the AAC features into the training dataset
            logger.info(msg="/ Task: Merging 'CGR' features")
            merge_csv_by_id(
                file1=output_file,
                file2=output_cgr_file,
                logger=logger,
            )

            # Insert a blank line in the log for readability
            logger.add_spacer(level=logging.INFO, lines=1)

            # Loop over each strain type
            for _, group in enumerate(["low", "medium", "high"], start=1):

                # Define input and output paths
                input_fasta_file = os.path.join(
                    base_path, f"data/processed/group/{suffix}_{group}.fasta"
                )
                output_file = os.path.join(
                    base_path, f"data/processed/group/{suffix}_{group}.csv"
                )
                output_cgr_file = os.path.join(
                    base_path, f"data/processed/group/{suffix}_{group}_cgr.csv"
                )

                # Perform CGR feature extraction
                encode_fasta_to_cgr_multi(
                    input_fasta=input_fasta_file,
                    output_file=output_cgr_file,
                    resolution=16,
                    logger=logger,
                )

                # Insert a blank line in the log for readability
                logger.add_spacer(level=logging.INFO, lines=1)

                # Merge the AAC features into the training dataset
                logger.info(msg="/ Task: Merging 'CGR' features")
                merge_csv_by_id(
                    file1=output_file,
                    file2=output_cgr_file,
                    logger=logger,
                )

                # Insert a blank line in the log for readability
                logger.add_spacer(level=logging.INFO, lines=1)

            # Loop over each strain type
            for _, group in enumerate(["train", "test"], start=1):

                # Define input and output paths
                input_fasta_file = os.path.join(
                    base_path, f"data/processed/split/{suffix}_{group}.fasta"
                )
                output_file = os.path.join(
                    base_path, f"data/processed/split/{suffix}_{group}.csv"
                )
                output_cgr_file = os.path.join(
                    base_path, f"data/processed/split/{suffix}_{group}_cgr.csv"
                )

                # Perform CGR feature extraction
                encode_fasta_to_cgr_multi(
                    input_fasta=input_fasta_file,
                    output_file=output_cgr_file,
                    resolution=16,
                    logger=logger,
                )

                # Insert a blank line in the log for readability
                logger.add_spacer(level=logging.INFO, lines=1)

                # Merge the AAC features into the training dataset
                logger.info(msg="/ Task: Merging 'CGR' features")
                merge_csv_by_id(
                    file1=output_file,
                    file2=output_cgr_file,
                    logger=logger,
                )

                # Insert a blank line in the log for readability
                logger.add_spacer(level=logging.INFO, lines=1)

    except Exception:
        logger.exception(msg="Unexpected error in 'run_cgr_pipeline()'.")
        raise
