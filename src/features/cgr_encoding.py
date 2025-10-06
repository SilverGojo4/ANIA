# pylint: disable=import-error, wrong-import-position, too-many-locals, too-many-arguments, too-many-positional-arguments, too-many-statements
"""
Run CGR Feature Extraction
"""
# ============================== Standard Library Imports ==============================
import logging
import os
import sys
import time

# ============================== Third-Party Library Imports ==============================
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector

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


def cgr_encoding(
    input_fasta: str,
    output_csv: str,
    resolution: int,
    kmer_k: int,
    aaindex_property: pd.DataFrame,
    logger: CustomLogger,
) -> None:
    """
    Execute Chaos Game Representation (CGR) feature extraction for a given FASTA file.

    Parameters
    ----------
    input_fasta : str
        Path to the input FASTA file.
    output_csv : str
        Path to save the final output CSV file.
    resolution : int
        Resolution of the CGR encoding (e.g., 8, 16, 32).
    kmer_k : int
        Length of k-mers used for property mapping (e.g., 3 for tri-peptides).
    aaindex_property : pd.DataFrame
        Pre-loaded AAindex property table (with amino acids as rows and property IDs as columns).
    logger : CustomLogger
        Logger instance for structured logging.

    Returns
    -------
    None
    """
    logger.info(
        f"/ Task: Run CGR Encoding (resolution = {resolution}) on '{input_fasta}'"
    )
    logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

    # Start timing
    start_time = time.time()

    try:
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

        # Log Base Layer Summary
        logger.log_with_borders(
            level=logging.INFO,
            message=(
                f"[ CGR Base Layer ]\n"
                f"▸ Resolution       : '{resolution}x{resolution}'\n"
                f"▸ Records (Samples): {len(cgr_df)}\n"
                f"▸ Feature Columns  : {cgr_df.shape[1] - 1}"
            ),
            border="|",
            length=120,
        )

        # Compute k-mer to pixel mappings
        pixel_kmer_maps = map_kmers(
            sequences=sequences,
            cgr_results=cgr_results,
            resolution=resolution,
            k=kmer_k,
        )

        # Process AAindex Property Layers
        if "AminoAcid" not in aaindex_property.columns:
            raise KeyError("AAindex table must include 'AminoAcid' column.")
        property_table = aaindex_property.set_index("AminoAcid")
        property_ids = list(property_table.columns)

        # Compute property maps (per sequence)
        for property_id in property_ids:
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
            logger.log_with_borders(
                level=logging.INFO,
                message=(
                    f"▸ Added AAindex Property: '{property_id}' ({prop_df.shape[1]} columns)"
                ),
                border="|",
                length=120,
            )

        # Save the CGR features to a CSV file
        logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")
        cgr_df.to_csv(output_csv, index=False)
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
        logger.exception(msg="Unexpected error in 'cgr_encoding()'.")
        raise


# ============================== Pipeline Entry Point ==============================
def run_cgr_pipeline(
    base_path: str,
    logger: CustomLogger,
    **kwargs,
) -> None:
    """
    Executes CGR feature encoding pipeline for selected resolution.

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
            "cgr_input_dir", os.path.join(base_path, "data/processed/")
        )
        output_dir = kwargs.get(
            "cgr_output_dir", os.path.join(base_path, "data/processed/")
        )
        aaindex_csv = kwargs.get(
            "cgr_aaindex_path",
            os.path.join(base_path, "configs/AAindex_properties.csv"),
        )
        n_splits = kwargs.get("cgr_n_splits", 5)
        cgr_res = kwargs.get("cgr_resolution", 16)
        kmer_k = kwargs.get("cgr_kmer_k", 3)

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

        # Load AAindex property
        aaindex_property = load_dataframe_by_columns(file_path=aaindex_csv)

        # Process each strain
        for _, (_, suffix) in enumerate(strains.items(), start=1):
            # Construct species-specific input/output folders
            species_input_dir = os.path.join(input_dir, suffix)
            species_output_dir = os.path.join(output_dir, suffix, "cgr")
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

                # Construct output path
                output_csv = os.path.join(
                    split_output_dir,
                    f"{split_name}_res{cgr_res}.csv",
                )

                # Run encoding
                cgr_encoding(
                    input_fasta=path,
                    output_csv=output_csv,
                    resolution=cgr_res,
                    kmer_k=kmer_k,
                    aaindex_property=aaindex_property,
                    logger=logger,
                )
                split_feature_files.append(output_csv)
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
