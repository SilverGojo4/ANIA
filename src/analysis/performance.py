# pylint: disable=line-too-long, too-many-lines, import-error, wrong-import-position, too-many-arguments, too-many-positional-arguments, too-many-locals, too-many-statements, broad-exception-caught
"""
AMP - Visualize ML Model Predictions and Evaluation Metrics
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
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

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
general_logger.info(msg="#" * 40 + " 'AMP -Visualize ML Model Predictions' " + "#" * 40)
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
            length=100,
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
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Successfully loaded visualization settings for '{plot_type}'.",
            border="|",
            length=100,
        )

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


def plot_regression(
    csv_path: str,
    true_col: str,
    pred_col: str,
    visualization_config_path: str,
    output_path: str,
    logger: CustomLogger,
) -> None:
    """
    Generate a regression scatter plot comparing true values and predicted values.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing actual and predicted values.
    true_col : str
        Column name for the actual values.
    pred_col : str
        Column name for the predicted values.
    visualization_config_path : str
        Path to the JSON configuration file for visualization settings.
    output_path : str
        Path to save the output regression plot (PNG format).
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    None
    """
    try:
        # Load visualization configuration
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
        config = load_visualization_config(
            visualization_config_path, "RegressionPlot", logger
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Log start of visualization
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Loading CSV file for regression plot visualization:\n'{csv_path}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Check if file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File '{csv_path}' not found.")

        # Load CSV file
        df = pd.read_csv(csv_path)

        # Ensure specified columns exist
        if true_col not in df.columns or pred_col not in df.columns:
            raise ValueError(
                f"Column '{true_col}' or '{pred_col}' not found in the dataset."
            )

        # Extract true and predicted values
        y_true = df[true_col]
        y_pred = df[pred_col]

        # Compute regression metrics
        mae = np.mean(np.abs(y_true - y_pred))
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        pcc = np.corrcoef(y_true, y_pred)[0, 1]  # Pearson Correlation Coefficient

        # Log computed metrics
        logger.log_with_borders(
            level=logging.INFO,
            message=(
                f"Evaluation Metrics:\n"
                f"Mean Absolute Error (MAE): {mae:.4f}\n"
                f"Mean Squared Error (MSE): {mse:.4f}\n"
                f"Root Mean Squared Error (RMSE): {rmse:.4f}\n"
                f"R² Score: {r2:.4f}\n"
                f"Pearson Correlation Coefficient (PCC): {pcc:.4f}"
            ),
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # # Set global font settings
        # plt.rcParams["font.family"] = config["font_family"]
        # plt.rcParams["font.size"] = config["font_size"]

        # Plot regression scatter plot
        plt.figure(figsize=tuple(config["plot_size"]))

        # Scatter plot with regression line
        sns.regplot(
            x=y_true,
            y=y_pred,
            scatter_kws={
                "color": config["scatter_color"],
                "alpha": config["scatter_alpha"],
                "s": config["scatter_size"],
                "edgecolor": config["scatter_edgecolor"],
            },
            line_kws={
                "color": config["regression_line_color"],
                "linewidth": config["regression_line_width"],
            },
            ci=config["regression_ci"],
        )
        plt.draw()

        # Identity line (y = x)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            config["identity_line_style"],
            color=config["identity_line_color"],
            alpha=config["identity_line_alpha"],
            linewidth=config["identity_line_width"],
        )

        # Labels and title
        plt.xlabel(
            f"Actual {true_col}",
            fontsize=config["label_font_size"],
            fontproperties=TIMES_NEW_ROMAN_BD,
        )
        plt.ylabel(
            f"{pred_col}",
            fontsize=config["label_font_size"],
            fontproperties=TIMES_NEW_ROMAN_BD,
        )

        # Get the current axis object
        ax = plt.gca()

        # Apply tick font size
        ax.tick_params(axis="x", labelsize=config["tick_font_size"])
        ax.tick_params(axis="y", labelsize=config["tick_font_size"])

        # Apply tick font family using FontProperties
        for label in ax.get_xticklabels():
            label.set_fontname("Times New Roman")
            label.set_fontsize(config["tick_font_size"])

        for label in ax.get_yticklabels():
            label.set_fontname("Times New Roman")
            label.set_fontsize(config["tick_font_size"])

        # Place text using axis-relative positioning (0,0) is bottom-left, (1,1) is top-right
        x_text_pos = config["text_anchor_x"]
        y_text_pos = config["text_anchor_y"]
        stats_text = f"MSE: {mse:.4f}\nR²: {r2:.4f}\nPearson: {pcc:.4f}"
        plt.text(
            x_text_pos,
            y_text_pos,
            stats_text,
            fontsize=config["stats_font_size"],
            bbox={"facecolor": config["bbox_facecolor"], "alpha": config["bbox_alpha"]},
            transform=ax.transAxes,  # Make coordinates relative to the axis
            verticalalignment="top",  # Align text to the top of the box
            horizontalalignment="left",  # Align text to the left
            fontproperties=TIMES_NEW_ROMAN_BD,
        )

        # Apply tight layout
        plt.tight_layout()

        # Save the plot
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(
            output_path,
            dpi=config["dpi"],
            bbox_inches="tight",
            transparent=config["transparent"],
        )
        plt.close()

        # Log successful save
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Regression plot saved:\n'{output_path}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError in 'plot_regression': {str(e)}")
        raise

    except ValueError as e:
        logger.error(f"ValueError in 'plot_regression': {str(e)}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error in 'plot_regression': {str(e)}")
        raise


def plot_residuals(
    csv_path: str,
    true_col: str,
    pred_col: str,
    visualization_config_path: str,
    output_path: str,
    logger: CustomLogger,
) -> None:
    """
    Generate a residual plot to visualize prediction errors.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing actual and predicted values.
    true_col : str
        Column name for the actual values.
    pred_col : str
        Column name for the predicted values.
    visualization_config_path : str
        Path to the JSON configuration file for visualization settings.
    output_path : str
        Path to save the residual plot (PNG format).
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    None
    """
    try:
        # Load visualization configuration
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
        config = load_visualization_config(
            visualization_config_path, "ResidualPlot", logger
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Log start of residual plot generation
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Loading CSV file for residual plot visualization:\n'{csv_path}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Check if file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File '{csv_path}' not found.")

        # Load CSV file
        df = pd.read_csv(csv_path)

        # Ensure specified columns exist
        if true_col not in df.columns or pred_col not in df.columns:
            raise ValueError(
                f"Column '{true_col}' or '{pred_col}' not found in the dataset."
            )

        # Extract true and predicted values
        y_true = df[true_col]
        y_pred = df[pred_col]

        # Compute residuals
        residuals = y_true - y_pred

        # Compute key residual statistics
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        residual_min = np.min(residuals)
        residual_max = np.max(residuals)
        residual_median = np.median(residuals)
        residual_iqr = np.percentile(residuals, 75) - np.percentile(residuals, 25)
        residual_range = residual_max - residual_min

        # Log residual statistics
        logger.log_with_borders(
            level=logging.INFO,
            message=(
                f"Residual Statistics:\n"
                f"Mean: {residual_mean:.4f}\n"
                f"Standard Deviation: {residual_std:.4f}\n"
                f"Min: {residual_min:.4f}\n"
                f"Max: {residual_max:.4f}\n"
                f"Median: {residual_median:.4f}\n"
                f"IQR (Interquartile Range): {residual_iqr:.4f}\n"
                f"Range: {residual_range:.4f}"
            ),
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # # Set global font settings
        # plt.rcParams["font.family"] = config["font_family"]
        # plt.rcParams["font.size"] = config["font_size"]

        # Create residual plot
        plt.figure(figsize=tuple(config["plot_size"]))
        sns.residplot(
            x=y_pred,
            y=residuals,
            scatter_kws={
                "color": config["scatter_color"],
                "alpha": config["scatter_alpha"],
                "s": config["scatter_size"],
                "edgecolor": config["scatter_edgecolor"],
            },
            lowess=config["lowess_smoothing"],
            line_kws={
                "color": config["lowess_line_color"],
                "linewidth": config["lowess_line_width"],
            },
        )

        # Draw a horizontal reference line at zero
        plt.axhline(
            y=0,
            color=config["reference_line_color"],
            linestyle=config["reference_line_style"],
            alpha=config["reference_line_alpha"],
            linewidth=config["reference_line_width"],
        )

        # Apply x-axis limit using min/max
        x_min, x_max = min(y_pred), max(y_pred)
        x_range = x_max - x_min
        plt.xlim([x_min - 0.05 * x_range, x_max + 0.05 * x_range])

        # Apply y-axis limit using min/max
        y_min, y_max = min(residuals), max(residuals)
        y_range = y_max - y_min
        plt.ylim([y_min - 0.1 * y_range, y_max + 0.1 * y_range])

        # Labels
        plt.xlabel(
            f"{pred_col}",
            fontsize=config["label_font_size"],
            fontproperties=TIMES_NEW_ROMAN_BD,
        )
        plt.ylabel(
            "Residuals",
            fontsize=config["label_font_size"],
            fontproperties=TIMES_NEW_ROMAN_BD,
        )

        # Adjust tick labels
        plt.xticks(fontsize=config["tick_font_size"], fontproperties=TIMES_NEW_ROMAN)
        plt.yticks(fontsize=config["tick_font_size"], fontproperties=TIMES_NEW_ROMAN)

        # Apply tight layout
        plt.tight_layout()

        # Save the plot
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(
            output_path,
            dpi=config["dpi"],
            bbox_inches="tight",
            transparent=config["transparent"],
        )
        plt.close()

        # Log successful save
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Residual plot saved:\n'{output_path}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError in 'plot_residuals': {str(e)}")
        raise

    except ValueError as e:
        logger.error(f"ValueError in 'plot_residuals': {str(e)}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error in 'plot_residuals': {str(e)}")
        raise


def plot_residual_distribution(
    csv_path: str,
    true_col: str,
    pred_col: str,
    visualization_config_path: str,
    output_path: str,
    logger: CustomLogger,
) -> None:
    """
    Generate a histogram and KDE plot of residuals.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing actual and predicted values.
    true_col : str
        Column name for the actual values.
    pred_col : str
        Column name for the predicted values.
    visualization_config_path : str
        Path to the JSON configuration file for visualization settings.
    output_path : str
        Path to save the residual histogram (PNG format).
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    None
    """
    try:
        # Load visualization configuration
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
        config = load_visualization_config(
            visualization_config_path, "ResidualHistogram", logger
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Log start of residual histogram generation
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Loading CSV file for residual histogram visualization:\n'{csv_path}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Check if file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File '{csv_path}' not found.")

        # Load CSV file
        df = pd.read_csv(csv_path)

        # Ensure specified columns exist
        if true_col not in df.columns or pred_col not in df.columns:
            raise ValueError(
                f"Column '{true_col}' or '{pred_col}' not found in the dataset."
            )

        # Extract true and predicted values
        y_true = df[true_col]
        y_pred = df[pred_col]

        # Compute residuals
        residuals = y_true - y_pred

        # Compute mean and std for normal distribution comparison
        mu, sigma = np.mean(residuals), np.std(residuals)

        # # Set global font settings
        # plt.rcParams["font.family"] = config["font_family"]
        # plt.rcParams["font.size"] = config["font_size"]

        # Create figure
        plt.figure(figsize=tuple(config["plot_size"]))

        # Plot histogram
        sns.histplot(
            residuals,  # type: ignore
            bins=config["bins"],
            kde=False,
            color=config["hist_color"],
            edgecolor=config["hist_edgecolor"],
            alpha=config["hist_alpha"],
        )

        # Overlay KDE plot
        sns.kdeplot(
            residuals,  # type: ignore
            color=config["kde_color"],
            linewidth=config["kde_linewidth"],
            label="KDE",
        )

        # Overlay normal distribution curve
        x_vals = np.linspace(min(residuals), max(residuals), 100)
        normal_dist = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x_vals - mu) / sigma) ** 2
        )
        plt.plot(
            x_vals,
            normal_dist
            * len(residuals)
            * (max(residuals) - min(residuals))
            / config["bins"],
            linestyle=config["norm_line_style"],
            color=config["norm_line_color"],
            linewidth=config["norm_linewidth"],
            label="Normal Dist.",
        )

        # Labels
        plt.xlabel(
            "Residuals",
            fontsize=config["label_font_size"],
            fontproperties=TIMES_NEW_ROMAN_BD,
        )
        plt.ylabel(
            "Frequency",
            fontsize=config["label_font_size"],
            fontproperties=TIMES_NEW_ROMAN_BD,
        )

        # Adjust tick labels
        plt.xticks(fontsize=config["tick_font_size"], fontproperties=TIMES_NEW_ROMAN)
        plt.yticks(fontsize=config["tick_font_size"], fontproperties=TIMES_NEW_ROMAN)

        # Add legend with configurable border color
        plt.legend(
            fontsize=config["legend_font_size"],
            edgecolor=config.get("legend_edgecolor", "black"),
        )

        # Apply tight layout
        plt.tight_layout()

        # Save the plot
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(
            output_path,
            dpi=config["dpi"],
            bbox_inches="tight",
            transparent=config["transparent"],
        )
        plt.close()

        # Log successful save
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Residual histogram saved:\n'{output_path}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError in 'plot_residual_distribution': {str(e)}")
        raise

    except ValueError as e:
        logger.error(f"ValueError in 'plot_residual_distribution': {str(e)}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error in 'plot_residual_distribution': {str(e)}")
        raise


def plot_qq(
    csv_path: str,
    true_col: str,
    pred_col: str,
    visualization_config_path: str,
    output_path: str,
    logger: CustomLogger,
) -> None:
    """
    Generate a Q-Q plot for residuals to check normality.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing actual and predicted values.
    true_col : str
        Column name for the actual values.
    pred_col : str
        Column name for the predicted values.
    visualization_config_path : str
        Path to the JSON configuration file for visualization settings.
    output_path : str
        Path to save the Q-Q plot (PNG format).
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    None
    """
    try:
        # Load visualization configuration
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
        config = load_visualization_config(visualization_config_path, "QQPlot", logger)
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Log start of Q-Q plot generation
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Loading CSV file for Q-Q plot visualization:\n'{csv_path}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Check if file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File '{csv_path}' not found.")

        # Load CSV file
        df = pd.read_csv(csv_path)

        # Ensure specified columns exist
        if true_col not in df.columns or pred_col not in df.columns:
            raise ValueError(
                f"Column '{true_col}' or '{pred_col}' not found in the dataset."
            )

        # Extract true and predicted values
        y_true = df[true_col]
        y_pred = df[pred_col]

        # Compute residuals
        residuals = y_true - y_pred

        # # Set global font settings
        # plt.rcParams["font.family"] = config["font_family"]
        # plt.rcParams["font.size"] = config["font_size"]

        # Create figure
        plt.figure(figsize=tuple(config["plot_size"]))

        # Generate Q-Q plot
        (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
        plt.scatter(
            osm,
            osr,
            color=config["point_color"],
            alpha=config["point_alpha"],
            edgecolor=config["point_edgecolor"],
            s=config["point_size"],
        )

        # Log Q-Q correlation coefficient (r)
        logger.log_with_borders(
            level=logging.INFO,
            message=(
                f"Q-Q Plot Normality Test:\n" f"Correlation Coefficient (r): {r:.4f}"
            ),
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Identity line (y = x)
        plt.plot(
            osm,
            slope * osm + intercept,  # type: ignore
            linestyle=config["identity_line_style"],
            color=config["identity_line_color"],
            linewidth=config["identity_line_width"],
        )

        # Labels
        plt.xlabel(
            "Theoretical Quantiles",
            fontsize=config["label_font_size"],
            fontproperties=TIMES_NEW_ROMAN_BD,
        )
        plt.ylabel(
            "Sample Quantiles",
            fontsize=config["label_font_size"],
            fontproperties=TIMES_NEW_ROMAN_BD,
        )

        # Adjust tick labels
        plt.xticks(fontsize=config["tick_font_size"], fontproperties=TIMES_NEW_ROMAN)
        plt.yticks(fontsize=config["tick_font_size"], fontproperties=TIMES_NEW_ROMAN)

        # Apply tight layout
        plt.tight_layout()

        # Save the plot
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(
            output_path,
            dpi=config["dpi"],
            bbox_inches="tight",
            transparent=config["transparent"],
        )
        plt.close()

        # Log successful save
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Q-Q plot saved:\n'{output_path}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError in 'plot_qq': {str(e)}")
        raise

    except ValueError as e:
        logger.error(f"ValueError in 'plot_qq': {str(e)}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error in 'plot_qq': {str(e)}")
        raise


def plot_residual_boxplot(
    csv_path: str,
    true_col: str,
    pred_col: str,
    visualization_config_path: str,
    output_path: str,
    logger: CustomLogger,
) -> None:
    """
    Generate a boxplot of residuals grouped by predicted value bins.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing actual and predicted values.
    true_col : str
        Column name for the actual values.
    pred_col : str
        Column name for the predicted values.
    visualization_config_path : str
        Path to the JSON configuration file for visualization settings.
    output_path : str
        Path to save the boxplot of residuals (PNG format).
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    None
    """
    try:
        # Load visualization configuration
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
        config = load_visualization_config(
            visualization_config_path, "ResidualBoxplot", logger
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Log start of residual boxplot generation
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Loading CSV file for residual boxplot visualization:\n'{csv_path}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Check if file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File '{csv_path}' not found.")

        # Load CSV file
        df = pd.read_csv(csv_path)

        # Ensure specified columns exist
        if true_col not in df.columns or pred_col not in df.columns:
            raise ValueError(
                f"Column '{true_col}' or '{pred_col}' not found in the dataset."
            )

        # Extract true and predicted values
        y_true = df[true_col]
        y_pred = df[pred_col]

        # Compute residuals
        residuals = y_true - y_pred

        # Define bins
        num_bins = config["bins"]
        df["Predicted Bin"] = pd.cut(y_pred, bins=num_bins)

        # Calculate IQR (Interquartile Range)
        q1, q3 = np.percentile(residuals, [25, 75])
        iqr = q3 - q1

        # Compute 1.5*IQR bounds for outlier detection
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Count the number of outliers
        num_outliers = np.sum((residuals < lower_bound) | (residuals > upper_bound))

        # Count the number of samples in each predicted bin
        bin_counts = df["Predicted Bin"].value_counts().sort_index()

        # Log the statistics
        logger.log_with_borders(
            level=logging.INFO,
            message=(
                f"Residual Boxplot Statistics:\n"
                f"Outliers (beyond 1.5*IQR): {num_outliers}\n"
                f"Sample Count per Bin:\n{bin_counts.to_string()}"
            ),
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # # Set global font settings
        # plt.rcParams["font.family"] = config["font_family"]
        # plt.rcParams["font.size"] = config["font_size"]

        # Create figure
        plt.figure(figsize=tuple(config["plot_size"]))

        # Boxplot
        sns.boxplot(
            x=df["Predicted Bin"],
            y=residuals,
            color=config["box_color"],
            linewidth=config["box_linewidth"],
            fliersize=config["flier_size"],
            flierprops={
                "marker": config["flier_marker"],
                "color": config["flier_color"],
                "alpha": config["flier_alpha"],
            },
        )

        # Labels
        plt.xlabel(
            f"{pred_col} (Binned)",
            fontsize=config["label_font_size"],
            fontproperties=TIMES_NEW_ROMAN_BD,
        )
        plt.ylabel(
            "Residuals",
            fontsize=config["label_font_size"],
            fontproperties=TIMES_NEW_ROMAN_BD,
        )

        # Adjust tick labels
        plt.xticks(
            rotation=45,
            fontsize=config["tick_font_size"],
            fontproperties=TIMES_NEW_ROMAN,
        )
        plt.yticks(fontsize=config["tick_font_size"], fontproperties=TIMES_NEW_ROMAN)

        # Apply tight layout
        plt.tight_layout()

        # Save the plot
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(
            output_path,
            dpi=config["dpi"],
            bbox_inches="tight",
            transparent=config["transparent"],
        )
        plt.close()

        # Log successful save
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Residual boxplot saved:\n'{output_path}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError in 'plot_residual_boxplot': {str(e)}")
        raise

    except ValueError as e:
        logger.error(f"ValueError in 'plot_residual_boxplot': {str(e)}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error in 'plot_residual_boxplot': {str(e)}")
        raise


def compute_model_performance(
    model_files: dict,
    true_col: str,
    pred_col: str,
    model_name_mapping: dict,
    logger: CustomLogger,
) -> dict:
    """
    Compute performance metrics (RMSE, R², PCC, MAE) for multiple models.

    Parameters
    ----------
    model_files : dict
        Dictionary mapping model names to their corresponding CSV file paths.
    true_col : str
        Column name for the actual values.
    pred_col : str
        Column name for the predicted values.
    model_name_mapping : dict
        Dictionary mapping internal model names to display-friendly names.
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    dict
        Dictionary containing model performance metrics.
    """
    performance_metrics = {}

    for model, file_path in model_files.items():
        try:
            # Get user-defined model name or fallback to the original model name
            display_model_name = model_name_mapping.get(model, model)

            # Log file processing
            logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
            logger.log_with_borders(
                level=logging.INFO,
                message=f"Processing model '{display_model_name}' from CSV file:\n'{file_path}'",
                border="|",
                length=100,
            )
            logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File '{file_path}' not found.")

            # Load CSV file
            df = pd.read_csv(file_path)

            # Ensure required columns exist
            if true_col not in df.columns or pred_col not in df.columns:
                raise ValueError(
                    f"Column '{true_col}' or '{pred_col}' not found in the dataset."
                )

            # Extract true and predicted values
            y_true = df[true_col]
            y_pred = df[pred_col]

            # Compute performance metrics
            mae = np.mean(np.abs(y_true - y_pred))
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            pcc = np.corrcoef(y_true, y_pred)[0, 1]  # Pearson Correlation Coefficient

            # Store metrics
            performance_metrics[model] = {
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R²": r2,
                "PCC": pcc,
            }

            # Log computed metrics
            logger.log_with_borders(
                level=logging.INFO,
                message=(
                    f"Evaluation Metrics:\n"
                    f"Mean Absolute Error (MAE): {mae:.4f}\n"
                    f"Mean Squared Error (MSE): {mse:.4f}\n"
                    f"Root Mean Squared Error (RMSE): {rmse:.4f}\n"
                    f"R² Score: {r2:.4f}\n"
                    f"Pearson Correlation Coefficient (PCC): {pcc:.4f}"
                ),
                border="|",
                length=100,
            )

        except FileNotFoundError as e:
            logger.error(f"FileNotFoundError in 'compute_model_performance': {str(e)}")
            raise

        except ValueError as e:
            logger.error(f"ValueError in 'compute_model_performance': {str(e)}")
            raise

        except Exception as e:
            logger.error(f"Unexpected error in 'compute_model_performance': {str(e)}")
            raise

    return performance_metrics


def plot_model_performance(
    model_files: dict,
    true_col: str,
    pred_col: str,
    model_name_mapping: dict,
    visualization_config_path: str,
    output_path: str,
    logger: CustomLogger,
) -> None:
    """
    Compute and visualize model performance metrics using a grouped bar plot.

    Parameters
    ----------
    model_files : dict
        Dictionary mapping model names to their corresponding CSV file paths.
    true_col : str
        Column name for the actual values.
    pred_col : str
        Column name for the predicted values.
    model_name_mapping : dict
        Dictionary mapping internal model names to display-friendly names.
    visualization_config_path : str
        Path to the JSON configuration file for visualization settings.
    output_path : str
        Path to save the performance bar chart (PNG format).
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    None
    """
    try:
        # Step 1: Compute performance metrics for all models
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message="Computing performance metrics for all models.",
            border="|",
            length=100,
        )

        # Compute model performance metrics
        performance_metrics = compute_model_performance(
            model_files, true_col, pred_col, model_name_mapping, logger
        )

        # Step 2: Rename dictionary keys using display-friendly model names
        renamed_performance_metrics = {
            model_name_mapping.get(model, model): metrics
            for model, metrics in performance_metrics.items()
        }

        # Load visualization configuration
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
        config = load_visualization_config(
            visualization_config_path, "ModelPerformance", logger
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Log start of model performance visualization
        logger.log_with_borders(
            level=logging.INFO,
            message="Generating model performance comparison bar chart.",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Convert renamed dictionary to a DataFrame
        performance_df = pd.DataFrame.from_dict(
            renamed_performance_metrics, orient="index"
        )

        # Transpose DataFrame so metrics are grouped together
        performance_df = performance_df.T  # Metrics on X-axis, models as groups

        # Drop RMSE row if present
        if "RMSE" in performance_df.index:
            logger.info("Excluding RMSE from visualization.")
            performance_df = performance_df.drop(index="RMSE")

        # Step 3: Log the extracted model names and metrics
        models = list(performance_df.columns)  # Mapped model names
        metrics = list(performance_df.index)  # Extracted from transposed DataFrame

        logger.log_with_borders(
            level=logging.INFO,
            message=(
                f"Performance Metrics Extracted:\n"
                f"Models (Friendly Names): {models}\n"
                f"Metrics: {metrics}"
            ),
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # # Step 4: Set global font settings
        # plt.rcParams["font.family"] = config["font_family"]
        # plt.rcParams["font.size"] = config["font_size"]

        # Step 5: Create the grouped bar plot
        plt.figure(figsize=tuple(config["plot_size"]))
        performance_df.plot(
            kind="bar",
            alpha=config["bar_alpha"],
            width=config["bar_width"],  # Use width from JSON
            color=config.get(
                "bar_colors", None
            ),  # Use color scheme from JSON if provided
        )

        # Labels
        plt.xlabel(
            "Performance Metrics",
            fontsize=config["label_font_size"],
            fontproperties=TIMES_NEW_ROMAN_BD,
        )
        plt.ylabel(
            "Performance",
            fontsize=config["label_font_size"],
            fontproperties=TIMES_NEW_ROMAN_BD,
        )

        # Adjust tick labels for better readability
        plt.xticks(
            rotation=config["xticks_rotation"],
            fontsize=config["tick_font_size"],
            fontproperties=TIMES_NEW_ROMAN,
        )
        plt.yticks(fontsize=config["tick_font_size"], fontproperties=TIMES_NEW_ROMAN)

        # Step 6: Move the legend outside the plot while ensuring correct model names
        plt.legend(
            title=None,
            labels=models,  # Ensure legend uses friendly model names
            fontsize=config["legend_font_size"],
            edgecolor=config["legend_edgecolor"],
            loc=config["legend_location"],  # JSON-defined location
            bbox_to_anchor=tuple(config["legend_bbox_anchor"]),  # JSON-defined position
        )

        # Step 7: Apply tight layout and save the plot
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(
            output_path,
            dpi=config["dpi"],
            bbox_inches="tight",
            transparent=config["transparent"],
        )
        plt.close()

        # Step 8: Log the successful save
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Model performance comparison chart saved:\n'{output_path}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

    except KeyError as e:
        logger.error(
            f"KeyError in 'plot_model_performance': Missing key {str(e)} in performance metrics."
        )
        raise

    except Exception as e:
        logger.error(f"Unexpected error in 'plot_model_performance': {str(e)}")
        raise


def plot_model_performance_compare(
    performance_metrics: dict,
    output_path: str,
    logger,
) -> None:
    """
    Visualize model performance metrics using a grouped bar plot.

    Parameters
    ----------
    performance_metrics : dict
        Dictionary mapping model names to a dict of metrics (e.g., MAE, R2, PCC).
        Example:
        {
            "Model A": {"MAE": 0.5, "R2": 0.7, "PCC": 0.8, "RMSE": 0.6},
            "Model B": {"MAE": 0.4, "R2": 0.75, "PCC": 0.82, "RMSE": 0.5}
        }
    output_path : str
        Path to save the performance bar chart.
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    None
    """

    try:
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message="Generating model performance comparison bar chart.",
            border="|",
            length=100,
        )

        # Load data into DataFrame
        performance_df = pd.DataFrame.from_dict(performance_metrics, orient="index")

        # Drop RMSE if present
        if "RMSE" in performance_df.columns:
            logger.info("Excluding RMSE from visualization.")
            performance_df = performance_df.drop(columns="RMSE")

        # Transpose so metrics are x-axis
        performance_df = performance_df.T  # Rows: metrics, Columns: models

        models = list(performance_df.columns)
        metrics = list(performance_df.index)

        logger.log_with_borders(
            level=logging.INFO,
            message=(
                f"Performance Metrics Extracted:\n"
                f"Models: {models}\n"
                f"Metrics: {metrics}"
            ),
            border="|",
            length=100,
        )

        # Plot settings (no JSON)
        config = {
            "plot_size": (10, 6),
            "bar_alpha": 0.8,
            "bar_width": 0.8,
            "bar_colors": [
                "#82B0D2",
                "#FA7F6F",
                "#8ECFC9",
                "#FFBE7A",
                "#BEB8DC",
                "#E7DAD2",
                "#A1C181",
                "#F5A623",
            ],
            "label_font_size": 14,
            "tick_font_size": 14,
            "legend_font_size": 10,
            "legend_edgecolor": "black",
            "legend_location": "upper left",
            "legend_bbox_anchor": (1.02, 1),
            "xticks_rotation": 45,
            "dpi": 300,
            "transparent": False,
        }

        # Plot bar chart
        plt.figure(figsize=config["plot_size"])
        performance_df.plot(
            kind="bar",
            alpha=config["bar_alpha"],
            width=config["bar_width"],
            color=config["bar_colors"][: len(models)],
        )

        plt.xlabel(
            "Performance Metrics",
            fontsize=config["label_font_size"],
            fontproperties=TIMES_NEW_ROMAN_BD,
        )
        plt.ylabel(
            "Performance",
            fontsize=config["label_font_size"],
            fontproperties=TIMES_NEW_ROMAN_BD,
        )
        plt.xticks(
            rotation=config["xticks_rotation"],
            fontsize=config["tick_font_size"],
            fontproperties=TIMES_NEW_ROMAN,
        )
        plt.yticks(fontsize=config["tick_font_size"], fontproperties=TIMES_NEW_ROMAN)

        plt.legend(
            title=None,
            labels=models,
            fontsize=config["legend_font_size"],
            edgecolor=config["legend_edgecolor"],
            loc=config["legend_location"],
            bbox_to_anchor=config["legend_bbox_anchor"],
        )

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(
            output_path,
            dpi=config["dpi"],
            bbox_inches="tight",
            transparent=config["transparent"],
        )
        plt.close()

        logger.log_with_borders(
            level=logging.INFO,
            message=f"Model performance comparison chart saved:\n'{output_path}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

    except Exception as e:
        logger.error(f"Error in 'plot_model_performance': {str(e)}")
        raise


def main():
    """
    Main function to process antimicrobial resistance (AMR) prediction results and generate various evaluation plots.

    This function automates the loading, computation, and visualization of machine learning (ML) model predictions
    for different bacterial strains. It systematically evaluates the models by generating multiple statistical plots
    and performance comparisons.

    Workflow:
    ---------
    1. Define bacterial strains for analysis:
       - Escherichia coli (EC)
       - Pseudomonas aeruginosa (PA)
       - Staphylococcus aureus (SA)
    2. Define the target column names for true and predicted values (`Log MIC Value` and `Predicted Log MIC Value`).
    3. Load visualization settings from a JSON configuration file (`visualization.json`).
    4. Iterate through each bacterial strain and perform the following steps for each ML model:
       - Load prediction results from CSV files.
       - Generate and save:
         - Regression scatter plot (`plot_regression`)
         - Residual plot (`plot_residuals`)
         - Residual histogram (`plot_residual_distribution`)
         - Normal Q-Q plot (`plot_qq`)
         - Residual boxplot (`plot_residual_boxplot`)
       - Log progress and insert blank lines for improved log readability.
    5. Generate and save a comparative performance bar chart for all models in each strain.
    6. Separate log entries between strains with visual dividers for clarity.

    Error Handling:
    ---------------
    - `FileNotFoundError`: Raised when expected input datasets or configuration files are missing.
    - `KeyError`: Raised when required columns (`Log MIC Value`, `Predicted Log MIC Value`) are absent in the dataset.
    - `ValueError`: Raised if numerical values are missing or invalid in the dataset.
    - Any unexpected exceptions are logged with detailed traceback before terminating execution.

    Outputs:
    --------
    - Visualization plots:
      - Regression scatter plots (`*_regression.png`)
      - Residual plots (`*_residual.png`)
      - Residual histograms (`*_residual_hist.png`)
      - Normal Q-Q plots (`*_qq.png`)
      - Residual boxplots (`*_residual_boxplot.png`)
    - Model performance comparison bar chart (`all_model_performance.png`)
    - Log files (`logs/vis.log`) capturing:
      - Model evaluation metrics:
        - Mean Absolute Error (MAE)
        - Mean Squared Error (MSE)
        - Root Mean Squared Error (RMSE)
        - R² Score
        - Pearson Correlation Coefficient (PCC)
      - Execution progress and any encountered errors for reproducibility.
    """
    try:
        # Mapping of strain names to file suffixes
        suffixs = ["EC", "PA", "SA"]
        strains = [
            "Escherichia coli",
            "Pseudomonas aeruginosa",
            "Staphylococcus aureus",
        ]
        strain_mapping = dict(zip(strains, suffixs))

        # Define the target column and metadata columns
        true_col = "Log MIC Value"
        pred_col = "Predicted Log MIC Value"

        # Path to the visualization settings configuration file
        visualization_config_path = os.path.join(
            BASE_PATH, "configs/visualization.json"
        )

        # Define model names
        models = [
            # "elastic_net",
            "gradient_boosting",
            # "lasso",
            # "linear",
            # "random_forest",
            # "ridge",
            # "svm",
            # "xgboost",
            # "elastic_net_cgr",
            "gradient_boosting_cgr",
            # "lasso_cgr",
            # "linear_cgr",
            # "random_forest_cgr",
            # "ridge_cgr",
            # "svm_cgr",
            # "xgboost_cgr",
            "ania_test",
        ]

        # Define model name mapping for better visualization
        model_name_mapping = {
            "random_forest": "Random Forest",
            "xgboost": "XGBoost",
            "svm": "Support Vector Machine",
            "lin_reg": "Linear Regression",
            "random_forest_cgr": "Random Forest (CGR)",
            "xgboost_cgr": "XGBoost (CGR)",
            "svm_cgr": "Support Vector Machine (CGR)",
            "lin_reg_cgr": "Linear Regression (CGR)",
            "ania": "ANIA",
        }

        # Iterate through each bacterial strain
        for i, strain in enumerate(strains):

            # Dictionary to store CSV file paths for each model
            model_files = {}

            for model in models:

                # Define input CSV file
                input_csv_file = os.path.join(
                    BASE_PATH,
                    f"experiments/{strain_mapping[strain]}/predictions/{model}.csv",
                )

                # Store model file path
                model_files[model] = input_csv_file

                # # ========== Generate and save the regression plot ==========
                # general_logger.info(
                #     msg=f"/ Task: Generating regression plot for '{strain}' using '{model}' predictions"
                # )
                # regression_output_png_file = os.path.join(
                #     BASE_PATH,
                #     f"outputs/results/{strain_mapping[strain]}/prediction/{model}_regression.png",
                # )
                # plot_regression(
                #     csv_path=input_csv_file,
                #     true_col=true_col,
                #     pred_col=pred_col,
                #     visualization_config_path=visualization_config_path,
                #     output_path=regression_output_png_file,
                #     logger=general_logger,
                # )

                # # Insert a blank line in the log for readability
                # log_blank_lines(line_count=1, logger=general_logger)

                # # ========== Generate and save the residual plot ==========
                # general_logger.info(
                #     msg=f"/ Task: Generating residual plot for '{strain}' using '{model}' predictions"
                # )
                # residual_output_png_file = os.path.join(
                #     BASE_PATH,
                #     f"outputs/results/{strain_mapping[strain]}/prediction/{model}_residual.png",
                # )
                # plot_residuals(
                #     csv_path=input_csv_file,
                #     true_col=true_col,
                #     pred_col=pred_col,
                #     visualization_config_path=visualization_config_path,
                #     output_path=residual_output_png_file,
                #     logger=general_logger,
                # )

                # # Insert a blank line in the log for readability
                # log_blank_lines(line_count=1, logger=general_logger)

                # # ========== Generate and save the residual histogram ==========
                # general_logger.info(
                #     msg=f"/ Task: Generating residual histogram for '{strain}' using '{model}' predictions"
                # )
                # residual_hist_output_png_file = os.path.join(
                #     BASE_PATH,
                #     f"outputs/results/{strain_mapping[strain]}/prediction/{model}_residual_hist.png",
                # )
                # plot_residual_distribution(
                #     csv_path=input_csv_file,
                #     true_col=true_col,
                #     pred_col=pred_col,
                #     visualization_config_path=visualization_config_path,
                #     output_path=residual_hist_output_png_file,
                #     logger=general_logger,
                # )

                # # Insert a blank line in the log for readability
                # log_blank_lines(line_count=1, logger=general_logger)

                # # ========== Generate and save the normal Q-Q plot ==========
                # general_logger.info(
                #     msg=f"/ Task: Generating normal Q-Q plot for '{strain}' using '{model}' predictions"
                # )
                # qq_output_png_file = os.path.join(
                #     BASE_PATH,
                #     f"outputs/results/{strain_mapping[strain]}/prediction/{model}_qq.png",
                # )
                # plot_qq(
                #     csv_path=input_csv_file,
                #     true_col=true_col,
                #     pred_col=pred_col,
                #     visualization_config_path=visualization_config_path,
                #     output_path=qq_output_png_file,
                #     logger=general_logger,
                # )

                # # Insert a blank line in the log for readability
                # log_blank_lines(line_count=1, logger=general_logger)

                # # ========== Generate and save the residual boxplot ==========
                # general_logger.info(
                #     msg=f"/ Task: Generating residual boxplot for '{strain}' using '{model}' predictions"
                # )
                # residual_boxplot_output_png_file = os.path.join(
                #     BASE_PATH,
                #     f"outputs/results/{strain_mapping[strain]}/prediction/{model}_residual_boxplot.png",
                # )
                # plot_residual_boxplot(
                #     csv_path=input_csv_file,
                #     true_col=true_col,
                #     pred_col=pred_col,
                #     visualization_config_path=visualization_config_path,
                #     output_path=residual_boxplot_output_png_file,
                #     logger=general_logger,
                # )

                # # Insert a blank line in the log for readability
                # log_blank_lines(line_count=1, logger=general_logger)

            # ========== Generate and save the model performance comparison ==========
            general_logger.info(
                msg=f"/ Task: Generating model performance comparison for '{strain}'"
            )
            performance_output_png_file = os.path.join(
                BASE_PATH,
                f"outputs/results/{strain_mapping[strain]}/prediction/all_model_performance.png",
            )

            plot_model_performance(
                model_files=model_files,
                true_col=true_col,
                pred_col=pred_col,
                model_name_mapping=model_name_mapping,
                visualization_config_path=visualization_config_path,
                output_path=performance_output_png_file,
                logger=general_logger,
            )

            # Insert a blank line in the log for readability
            log_blank_lines(line_count=1, logger=general_logger)

            if i < 2:
                general_logger.info(msg="=" * 100)
                log_blank_lines(line_count=1, logger=general_logger)

        performance_metrics_saureus = {
            "ESKAPEE-MICpred": {
                "MAE": 0.5183,
                "MSE": 0.4016,
                "R2": -0.3147,
                "PCC": 0.2304,
            },
            "esAMPMIC": {"MAE": 0.3531, "MSE": 0.2637, "R2": 0.1368, "PCC": 0.5486},
            "AMPActiPred": {"MAE": 0.3708, "MSE": 0.2428, "R2": -0.1491, "PCC": 0.3400},
            "ANIA": {"MAE": 0.3786, "MSE": 0.2348, "R2": 0.2313, "PCC": 0.6227},
        }
        performance_metrics_ecoli = {
            "ESKAPEE-MICpred": {
                "MAE": 0.4481,
                "MSE": 0.3397,
                "R2": -0.2673,
                "PCC": 0.4652,
            },
            "esAMPMIC": {"MAE": 0.3186, "MSE": 0.1734, "R2": 0.3530, "PCC": 0.6045},
            "AMPActiPred": {"MAE": 0.3433, "MSE": 0.2720, "R2": 0.0218, "PCC": 0.4651},
            "ANIA": {"MAE": 0.3347, "MSE": 0.1754, "R2": 0.3457, "PCC": 0.7046},
        }
        performance_metrics_paeruginosa = {
            "ESKAPEE-MICpred": {
                "MAE": 0.5061,
                "MSE": 0.3481,
                "R2": -0.1469,
                "PCC": 0.3668,
            },
            "esAMPMIC": {"MAE": 0.4090, "MSE": 0.3462, "R2": -0.1408, "PCC": 0.4712},
            "AMPActiPred": {"MAE": 0.3859, "MSE": 0.2112, "R2": 0.1958, "PCC": 0.4791},
            "ANIA": {"MAE": 0.3687, "MSE": 0.2335, "R2": 0.2307, "PCC": 0.5902},
        }
        plot_model_performance_compare(
            performance_metrics_saureus,
            "/home/user_312554028/AMP-MIC/outputs/results/SA/prediction/existing_tool_performance.png",
            general_logger,
        )
        plot_model_performance_compare(
            performance_metrics_ecoli,
            "/home/user_312554028/AMP-MIC/outputs/results/EC/prediction/existing_tool_performance.png",
            general_logger,
        )
        plot_model_performance_compare(
            performance_metrics_paeruginosa,
            "/home/user_312554028/AMP-MIC/outputs/results/PA/prediction/existing_tool_performance.png",
            general_logger,
        )

    except Exception:
        # Log any error encountered during the processing pipeline
        general_logger.exception(msg="Critical failure in main process.")
        sys.exit(1)


if __name__ == "__main__":
    main()
