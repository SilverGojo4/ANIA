# pylint: disable=import-error, wrong-import-position, too-many-arguments, too-many-positional-arguments, too-many-locals
"""
CD-HIT Redundancy Filtering Pipeline
"""
# ============================== Standard Library Imports ==============================
import logging
import os
import subprocess
import sys
import time
from typing import Dict

# ============================== Third-Party Library Imports ==============================
from Bio import SeqIO

# ============================== Project Root Path Setup ==============================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
LOGGING_PATH = os.path.join(BASE_PATH, "src/utils/Logging-Toolkit/src/python")

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
def run_cd_hit(
    input_fasta: str,
    input_metadata: str,
    output_fasta: str,
    logger: CustomLogger,
    cdhit_identity: float = 0.9,
    cdhit_word_size: int = 5,
    cdhit_memory: int = 16000,
    cdhit_threads: int = 4,
) -> Dict[str, int]:
    """
    Run CD-HIT on a FASTA file to remove redundant sequences and
    retain corresponding metadata records.

    Parameters
    ----------
    input_fasta : str
        Path to the input FASTA file.
    input_metadata : str
        Path to the input metadata CSV file.
    output_fasta : str
        Path to save the filtered FASTA file.
    logger : CustomLogger
        Logger instance for structured logging.
    cdhit_identity : float, optional
        Sequence identity threshold (default = 0.9).
    cdhit_word_size : int, optional
        Word size (n-mer) parameter (default = 5).
    cdhit_memory : int, optional
        Memory limit in MB (default = 16000).
    cdhit_threads : int, optional
        Number of CPU threads (default = 4).

    Returns
    -------
    Dict[str, int]
        Statistics summary: original, filtered, and removed counts.
    """
    logger.info(f"/ Task: Run CD-HIT on '{os.path.basename(input_fasta)}'")
    logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

    # Start timing
    start_time = time.time()

    try:
        # Count original sequences
        original_count = sum(1 for _ in SeqIO.parse(input_fasta, "fasta"))

        # Build CD-HIT command
        cmd = [
            "cd-hit",
            "-i",
            input_fasta,
            "-o",
            output_fasta,
            "-c",
            str(cdhit_identity),
            "-n",
            str(cdhit_word_size),
            "-M",
            str(cdhit_memory),
            "-T",
            str(cdhit_threads),
            "-d",
            "0",  # full defline
        ]

        # Log command
        formatted_cmd = (
            "cd-hit \\\n"
            f"  -i {input_fasta} \\\n"
            f"  -o {output_fasta} \\\n"
            f"  -c {cdhit_identity} \\\n"
            f"  -n {cdhit_word_size} \\\n"
            f"  -M {cdhit_memory} \\\n"
            f"  -T {cdhit_threads} \\\n"
            f"  -d 0"
        )
        logger.log_with_borders(
            level=logging.INFO,
            message=("Running CD-HIT command:\n" f"{formatted_cmd}"),
            border="|",
            length=120,
        )
        logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

        # Run command
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Count filtered sequences
        retained_ids = [record.id for record in SeqIO.parse(output_fasta, "fasta")]
        filtered_count = len(retained_ids)
        removed_count = original_count - filtered_count
        removed_ratio = (
            (removed_count / original_count) * 100 if original_count > 0 else 0.0
        )

        # Log summary
        logger.log_with_borders(
            level=logging.INFO,
            message=(
                f"[ CD-HIT Filtering Summary ]\n"
                f"▸ Input file : '{os.path.basename(input_fasta)}'\n"
                f"▸ Original sequences : {original_count}\n"
                f"▸ Removed (redundant) : {removed_count} ({removed_ratio:.2f}%)\n"
                f"▸ Filtered sequences : {filtered_count}\n"
                f"▸ Identity threshold : {cdhit_identity}\n"
                f"▸ Word size : {cdhit_word_size}\n"
            ),
            border="|",
            length=120,
        )
        logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")

        # Filter metadata by ID
        df_meta = load_dataframe_by_columns(file_path=input_metadata)
        df_filtered = df_meta[df_meta["ID"].isin(retained_ids)].copy()

        # Save filtered metadata
        before_count = len(df_meta)
        df_filtered = df_meta[df_meta["ID"].isin(retained_ids)].copy()
        after_count = len(df_filtered)
        output_metadata = output_fasta.replace(".fasta", ".csv")
        df_filtered.to_csv(output_metadata, index=False)
        logger.log_with_borders(
            level=logging.INFO,
            message=(
                f"[ Metadata Filtering Summary ]\n"
                f"▸ Original records : {before_count}\n"
                f"▸ Retained records : {after_count} ({after_count/before_count*100:.2f}%)"
            ),
            border="|",
            length=120,
        )
        logger.add_divider(level=logging.INFO, length=120, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Saved:\n'{output_fasta}'\n'{output_metadata}'",
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

        return {
            "original": original_count,
            "filtered": filtered_count,
            "removed": removed_count,
        }

    except Exception:
        logger.exception("Unexpected error in 'run_cd_hit()'.")
        raise


# ============================== Pipeline Entry Point ==============================
def run_cd_hit_pipeline(
    base_path: str,
    logger: CustomLogger,
    **kwargs,
) -> None:
    """
    Executes CD-HIT filtering on all grouped FASTA datasets.

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
            "cdhit_input_dir", os.path.join(base_path, "data/interim/aggregated")
        )
        output_dir = kwargs.get(
            "cdhit_output_dir", os.path.join(base_path, "data/interim/cdhit")
        )
        agg_method = kwargs.get("cdhit_aggregate_method", "min").lower()
        cdhit_identity = kwargs.get("cdhit_identity", 0.9)
        cdhit_word_size = kwargs.get("cdhit_word_size", 5)
        cdhit_memory = kwargs.get("cdhit_memory", 16000)
        cdhit_threads = kwargs.get("cdhit_threads", 4)

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
        for _, (_, suffix) in enumerate(strains.items(), start=1):

            # Check input files exist
            input_path = os.path.join(input_dir, f"{suffix}_agg_{agg_method}.fasta")
            input_metadata_path = os.path.join(
                input_dir, f"{suffix}_agg_{agg_method}.csv"
            )
            if not file_exists(file_path=input_path):
                raise FileNotFoundError(f"File not found: '{input_path}'")

            # Run CD-HIT
            identity_label = f"{cdhit_identity:.2f}".replace(".", "_")
            output_path = os.path.join(
                output_dir, f"{suffix}_agg_{agg_method}_cdhit{identity_label}.fasta"
            )
            run_cd_hit(
                input_fasta=input_path,
                input_metadata=input_metadata_path,
                output_fasta=output_path,
                logger=logger,
                cdhit_identity=cdhit_identity,
                cdhit_word_size=cdhit_word_size,
                cdhit_memory=cdhit_memory,
                cdhit_threads=cdhit_threads,
            )
            logger.add_spacer(level=logging.INFO, lines=1)

        # Final summary
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
        logger.exception("Critical failure in 'run_cd_hit_pipeline()'.")
        raise
