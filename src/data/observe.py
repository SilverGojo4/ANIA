# pylint: disable=line-too-long, import-error, wrong-import-position, too-many-locals, broad-exception-caught, too-many-branches, too-many-statements, too-many-nested-blocks
"""
AMP Data Observe (MIC Distribution)
"""
# ========== Standard Library Imports ==========
import argparse
import json
import logging
import os
import sys
import warnings

# ========== Third-Party Library Imports ==========
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from scipy import stats

# ============================== Project Root Path Setup ==============================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
LOGGING_PATH = os.path.join(BASE_PATH, "src/utils/logging_toolkit/src/python")

# Fix Python Path
if BASE_PATH not in sys.path:
    sys.path.append(BASE_PATH)
if LOGGING_PATH not in sys.path:
    sys.path.append(LOGGING_PATH)

# ============================== Font Configuration ==============================
TIMES_NEW_ROMAN_BD = FontProperties(
    fname="/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf"
)
TIMES_NEW_ROMAN = FontProperties(
    fname="/usr/share/fonts/truetype/msttcorefonts/times.ttf"
)

# ============================== Project-Specific Imports ==============================
# Logging configuration and custom logger
from setup_logging import CustomLogger, setup_logging

# ========== Initialize Logger ==========
input_config_file = os.path.join(BASE_PATH, "configs/general_logging.json")
output_log_path = os.path.join(BASE_PATH, "logs/visualization.log")
general_logger = setup_logging(
    input_config_file=input_config_file,
    logger_name="general_logger",
    handler_name="general",
    output_log_path=output_log_path,
)
general_logger.info(msg="#" * 40 + " 'AMP -Data Observe' " + "#" * 40)
general_logger.info(msg="" * 100)

# ========== Disable Warnings ==========
warnings.filterwarnings("ignore", module="matplotlib")
warnings.filterwarnings("ignore", module="seaborn")
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("seaborn").setLevel(logging.ERROR)


def log_blank_lines(line_count: int, logger: CustomLogger) -> None:
    """
    Logs empty messages to insert newlines into the log output.

    Parameters
    ----------
    line_count : int
        The number of newlines to log.
    logger : CustomLogger
        The logger used for logging messages.

    Returns
    -------
    None
    """
    for _ in range(line_count):
        logger.info(msg="")


def log_task_status(
    task_id: int, message: str, logger: CustomLogger, divider_length: int = 50
) -> None:
    """
    Logs a structured mission status message using the provided logger.

    Parameters
    ----------
    task_id : int
        Identifier for the current task.
    message : str
        Custom message describing the task status.
    logger : CustomLogger
        Logger instance for structured logging.
    divider_length : int
        Length of the divider lines in the log.

    Returns
    -------
    None
    """
    # Log the mission number, indicating progress
    logger.info(msg=f"/{task_id}")

    # Add a divider to separate sections of the log
    logger.add_divider(level=logging.INFO, length=divider_length, border="+", fill="-")

    # Log the custom message with borders for emphasis
    logger.log_with_borders(
        level=logging.INFO,
        message=message,
        border="|",
        length=divider_length,
    )

    # Add another divider to mark the end of the section
    logger.add_divider(level=logging.INFO, length=divider_length, border="+", fill="-")

    # Add a newline to improve readability between log entries
    log_blank_lines(line_count=1, logger=logger)


def load_visualization_config(
    config_path: str,
    plot_type: str,
    logger: CustomLogger,
) -> dict:
    """
    Load visualization settings for a specific plot type from a JSON file.

    Parameters
    ----------
    config_path : str
        Path to the JSON configuration file.
    plot_type : str
        The type of plot (e.g., 'RegressionPlot').
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    dict
        Dictionary containing visualization settings for the given plot type.
    """
    try:
        # Log the start of visualization settings loading
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Loading visualization settings:\n'{config_path}'.",
            border="|",
            length=110,
        )

        # Load JSON file
        with open(config_path, "r", encoding="utf-8") as file:
            all_settings = json.load(file)

        # Ensure the plot type exists in the config
        if plot_type not in all_settings:
            raise KeyError(
                f"Plot type '{plot_type}' not found in visualization config file."
            )

        settings = all_settings[plot_type]

        # Log successful loading
        logger.add_divider(level=logging.INFO, length=110, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Successfully loaded visualization settings for '{plot_type}'.",
            border="|",
            length=110,
        )
        logger.add_divider(level=logging.INFO, length=110, border="+", fill="-")

        return settings

    except FileNotFoundError:
        # Log and raise file not found errors
        logger.error(
            msg=f"FileNotFoundError in 'load_visualization_config': File '{config_path}' not found."
        )
        raise

    except json.JSONDecodeError:
        # Log and raise JSON parsing errors
        logger.error(
            msg=f"JSONDecodeError in 'load_visualization_config': Error parsing '{config_path}'."
        )
        raise

    except Exception as e:
        # Log and raise unexpected errors
        logger.error(msg=f"Unexpected error in 'load_visualization_config': {str(e)}")
        raise


def plot_sequence_length_vs_log_mic(
    csv_path: str,
    log_mic_cols: list,
    visualization_config_path: str,
    output_dir: str,
    logger: CustomLogger,
    sequence_col: str = "Sequence",
) -> None:
    """
    Plot histogram and regression of sequence length vs. Log MIC value for each specified MIC column.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing sequences and Log MIC values.
    log_mic_cols : list
        List of column names for Log MIC values.
    visualization_config_path : str
        Path to the JSON file specifying visualization settings.
    output_dir : str
        Directory to save the output plots.
    logger : CustomLogger
        Logger instance for tracking process.
    sequence_col : str, default 'Sequence'
        Column containing peptide sequences.
    """
    try:
        logger.add_divider(level=logging.INFO, length=110, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Loading:\n'{csv_path}'",
            border="|",
            length=110,
        )
        logger.add_divider(level=logging.INFO, length=110, border="+", fill="-")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File '{csv_path}' not found.")
        df = pd.read_csv(csv_path)

        config = load_visualization_config(
            visualization_config_path, "SequenceLengthVsMIC", logger
        )

        legend_font = FontProperties(
            fname="/usr/share/fonts/truetype/msttcorefonts/times.ttf",
            size=config.get("legend_font_size", 10),
        )

        if sequence_col not in df.columns:
            raise ValueError(f"'{sequence_col}' column not found.")
        df["Sequence Length"] = df[sequence_col].apply(len)

        for log_mic_col in log_mic_cols:
            if log_mic_col not in df.columns:
                logger.error(f"'{log_mic_col}' not found in CSV. Skipping.")
                continue

            try:
                fig, ax1 = plt.subplots(figsize=tuple(config["plot_size"]))
                ax1.hist(
                    df["Sequence Length"],
                    bins=range(5, max(df["Sequence Length"]) + 1),
                    color=config["hist_color"],
                    edgecolor=config["hist_edgecolor"],
                    alpha=config["hist_alpha"],
                )
                ax1.set_xlabel(
                    "Sequence Length",
                    fontsize=config["label_font_size"],
                    fontproperties=TIMES_NEW_ROMAN_BD,
                )
                ax1.set_ylabel(
                    "Count",
                    fontsize=config["label_font_size"],
                    fontproperties=TIMES_NEW_ROMAN_BD,
                )
                ax1.tick_params(axis="x", labelsize=config["tick_font_size"])
                ax1.tick_params(axis="y", labelsize=config["tick_font_size"])
                for label in ax1.get_xticklabels() + ax1.get_yticklabels():
                    label.set_fontproperties(TIMES_NEW_ROMAN)
                    label.set_fontsize(config["tick_font_size"])

                ax2 = ax1.twinx()
                scatter = ax2.scatter(
                    df["Sequence Length"],
                    df[log_mic_col],
                    color=config["scatter_color"],
                    alpha=config["scatter_alpha"],
                    s=config["scatter_size"],
                )
                sns.regplot(
                    x="Sequence Length",
                    y=log_mic_col,
                    data=df,
                    scatter=False,
                    ax=ax2,
                    color=config["regression_color"],
                    line_kws={"linewidth": config["regression_linewidth"]},
                )
                ax2.set_ylabel(
                    "Log MIC Value",
                    fontsize=config["label_font_size"],
                    fontproperties=TIMES_NEW_ROMAN_BD,
                )
                ax2.tick_params(axis="y", labelsize=config["tick_font_size"])
                for label in ax2.get_yticklabels():
                    label.set_fontproperties(TIMES_NEW_ROMAN)
                    label.set_fontsize(config["tick_font_size"])

                # Compute regression line parameters
                slope, intercept = np.polyfit(df["Sequence Length"], df[log_mic_col], 1)
                logger.log_with_borders(
                    level=logging.INFO,
                    message=f"{log_mic_col} Trend → Slope = {slope:.4f}, Intercept = {intercept:.4f}",
                    border="|",
                    length=110,
                )
                logger.add_divider(level=logging.INFO, length=110, border="+", fill="-")

                # Add legend showing slope/intercept
                regression_label = f"Slope: {slope:.4f}, Intercept: {intercept:.4f}"
                legend_elements = [
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="None",
                        color=config["scatter_color"],
                        label=f"Log MIC Value",
                        markerfacecolor=config["scatter_color"],
                        markersize=6,
                    ),
                    Line2D(
                        [0],
                        [0],
                        color=config["regression_color"],
                        lw=config["regression_linewidth"],
                        label=regression_label,
                    ),
                ]
                ax2.legend(
                    handles=legend_elements,
                    loc=config.get("legend_loc", "upper right"),
                    fontsize=config.get("legend_font_size", 10),
                    frameon=config.get("legend_frameon", True),
                    facecolor=config.get("legend_facecolor", "white"),
                    fancybox=True,
                    framealpha=config.get("legend_bbox_alpha", 0.8),
                    prop=legend_font,
                )

                output_path = os.path.join(output_dir, f"SeqLen_vs_MIC.png")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                plt.tight_layout()
                plt.savefig(
                    output_path,
                    dpi=config["dpi"],
                    bbox_inches="tight",
                    transparent=config["transparent"],
                )
                plt.close()

                logger.log_with_borders(
                    level=logging.INFO,
                    message=f"Saved:\n'{output_path}'",
                    border="|",
                    length=110,
                )
                logger.add_divider(level=logging.INFO, length=110, border="+", fill="-")

            except Exception as plot_err:
                logger.error(f"Plotting error for '{log_mic_col}': {plot_err}")
                continue

    except Exception as e:
        logger.error(f"Fatal error in 'plot_sequence_length_vs_log_mic': {e}")
        raise


def plot_log_mic_distribution(
    csv_path: str,
    log_mic_cols: list,
    visualization_config_path: str,
    output_dir: str,
    logger,
) -> None:
    """
    Plot dual-axis Log MIC distribution:
    - Left Y: Count (histogram)
    - Right Y: Density (KDE)

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing Log MIC values.
    log_mic_cols : list
        List of column names for Log MIC values.
    visualization_config_path : str
        Path to the JSON file specifying visualization settings.
    output_dir : str
        Directory to save the output plots.
    logger : CustomLogger
        Logger instance for tracking progress and errors.
    """
    try:
        logger.add_divider(level=logging.INFO, length=110, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO, message=f"Loading: {csv_path}", border="|", length=110
        )

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File '{csv_path}' not found.")
        df = pd.read_csv(csv_path)

        config = load_visualization_config(
            visualization_config_path, "LogMICDistribution", logger
        )

        for col in log_mic_cols:
            if col not in df.columns:
                logger.error(f"'{col}' not found in CSV. Skipping.")
                continue

            try:
                fig, ax1 = plt.subplots(figsize=tuple(config["plot_size"]))

                # 左側 Y 軸：Count
                counts, bins, _ = ax1.hist(
                    df[col].dropna(),
                    bins=config["bins"],
                    color=config["hist_color"],
                    edgecolor=config["hist_edgecolor"],
                    alpha=config["hist_alpha"],
                )
                ax1.set_xlabel(
                    "Log MIC Value",
                    fontsize=config["label_font_size"],
                    fontproperties=TIMES_NEW_ROMAN_BD,
                )
                ax1.set_ylabel(
                    "Count",
                    fontsize=config["label_font_size"],
                    fontproperties=TIMES_NEW_ROMAN_BD,
                )
                ax1.tick_params(axis="x", labelsize=config["tick_font_size"])
                ax1.tick_params(axis="y", labelsize=config["tick_font_size"])

                # 右側 Y 軸：KDE 密度
                ax2 = ax1.twinx()
                sns.kdeplot(
                    df[col].dropna(),
                    ax=ax2,
                    color=config["kde_color"],
                    linewidth=config["kde_linewidth"],
                )
                ax2.set_ylabel(
                    "Density",
                    fontsize=config["label_font_size"],
                    fontproperties=TIMES_NEW_ROMAN_BD,
                )
                ax2.tick_params(axis="y", labelsize=config["tick_font_size"])

                # 套用字型與大小到所有刻度
                for ax in [ax1, ax2]:
                    for label in ax.get_xticklabels() + ax.get_yticklabels():
                        label.set_fontproperties(TIMES_NEW_ROMAN)
                        label.set_fontsize(config["tick_font_size"])

                # 儲存圖片
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(
                    output_dir, f"Log_MIC_Distribution_{col}.png"
                )
                plt.tight_layout()
                plt.savefig(
                    output_path,
                    dpi=config["dpi"],
                    bbox_inches="tight",
                    transparent=config["transparent"],
                )
                plt.close()

                logger.log_with_borders(
                    level=logging.INFO,
                    message=f"Saved Log MIC distribution plot: {output_path}",
                    border="|",
                    length=110,
                )
                logger.add_divider(level=logging.INFO, length=110, border="+", fill="-")

            except Exception as plot_err:
                logger.error(f"Error plotting '{col}': {plot_err}")
                continue

        logger.log_with_borders(
            level=logging.INFO,
            message="Finished all Log MIC distribution plots.",
            border="|",
            length=110,
        )

    except Exception as e:
        logger.error(f"Fatal error in 'plot_log_mic_distribution': {e}")
        raise


def main():
    """
    Run sequence visualization tasks for each bacterial strain:
    - Plot Sequence Length vs. Log MIC
    - Plot Log MIC distribution (histogram + KDE)
    """
    try:
        suffixs = ["EC", "PA", "SA"]
        strains = [
            "Escherichia coli",
            "Pseudomonas aeruginosa",
            "Staphylococcus aureus",
        ]
        strain_mapping = dict(zip(strains, suffixs))

        visualization_config_path = os.path.join(
            BASE_PATH, "configs/visualization.json"
        )

        for i, strain in enumerate(strains):
            log_task_status(
                task_id=i + 1,
                message=f"Target: '{strain}'",
                logger=general_logger,
                divider_length=45,
            )

            data_input_path = os.path.join(
                BASE_PATH, f"data/processed/all/{strain_mapping[strain]}.csv"
            )
            save_dir = os.path.join(
                BASE_PATH, f"outputs/results/{strain_mapping[strain]}/distribution/"
            )

            general_logger.info(msg=f"/ Task: Plot sequence length vs. Log MIC.")
            plot_sequence_length_vs_log_mic(
                csv_path=data_input_path,
                log_mic_cols=["Log MIC Value"],
                visualization_config_path=visualization_config_path,
                output_dir=save_dir,
                logger=general_logger,
            )

            log_blank_lines(line_count=1, logger=general_logger)

            general_logger.info(msg=f"/ Task: Plot Log MIC distribution histogram.")
            plot_log_mic_distribution(
                csv_path=data_input_path,
                log_mic_cols=["Log MIC Value"],
                visualization_config_path=visualization_config_path,
                output_dir=save_dir,
                logger=general_logger,
            )

            log_blank_lines(line_count=1, logger=general_logger)

    except Exception:
        general_logger.exception(msg="Critical failure in main process.")
        sys.exit(1)


if __name__ == "__main__":
    main()
