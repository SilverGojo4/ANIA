# pylint: disable=too-many-locals
"""
Pixel-wise MIC Correlation Visualization

This module provides functions to:
- Merge prediction and feature data
- Compute pixel-level correlation between encoded features and MIC
- Visualize correlation and average heatmaps
"""

# ============================== Standard Library Imports ==============================
import os
import sys
from typing import Optional, Union

# ============================== Third-Party Library Imports ==============================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.font_manager import FontProperties

# ============================== Project Root Path Setup ==============================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if BASE_PATH not in sys.path:
    sys.path.append(BASE_PATH)

# ============================== Font Configuration ==============================
TIMES_NEW_ROMAN_BD = FontProperties(
    fname="/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf"
)
TIMES_NEW_ROMAN = FontProperties(
    fname="/usr/share/fonts/truetype/msttcorefonts/times.ttf"
)


# ============================== Core Functions ==============================
def merge_prediction_and_features(
    pred_path: str, feature_path: str, top_n: int = None  # type: ignore
) -> pd.DataFrame:
    """
    Merge prediction and feature CSVs. Optionally select top-N and bottom-N samples.

    Parameters
    ----------
    pred_path : str
        Path to prediction CSV with 'ID' and 'Predicted Log MIC Value'.
    feature_path : str
        Path to feature CSV with full feature set and 'ID'.
    top_n : int, optional
        If set, selects top-N and bottom-N samples based on predicted MIC.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame.
    """
    df_predict = pd.read_csv(pred_path)
    # df_predict = df_predict.iloc[:, :7]
    df_feature = pd.read_csv(feature_path)

    if top_n is not None:
        top = df_predict.nlargest(top_n, "Predicted Log MIC Value")
        bottom = df_predict.nsmallest(top_n, "Predicted Log MIC Value")
        # top = df_predict.nlargest(top_n, "Log MIC Value")
        # bottom = df_predict.nsmallest(top_n, "Log MIC Value")
        selected = pd.concat([top, bottom]).drop_duplicates(subset="ID")
    else:
        selected = df_predict

    merged = pd.merge(selected, df_feature, on="ID", how="inner").drop(
        columns=[
            "Sequence_y",
            "Targets_y",
            "Sequence Length_y",
            "Molecular Weight",
            "Log MIC Value_y",
            "MIC Group_y",
        ],
        errors="ignore",  # In case some columns are not present
    )
    merged = merged.rename(
        columns={
            "Sequence_x": "Sequence",
            "Targets_x": "Targets",
            "Sequence Length_x": "Sequence Length",
            "Log MIC Value_x": "Log MIC Value",
            "MIC Group_x": "MIC Group",
        }
    )

    return merged


def compute_pixel_correlation(
    df: pd.DataFrame,
    feature: str,
    mic_column: str = "Predicted Log MIC Value",
    resolution: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute correlation between each pixel and MIC value.
    Also return a boolean mask marking pixels with zero variance.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataframe containing pixel features and MIC values.
    feature : str
        Feature group name (e.g., "CGR(resolution=16)" or "PROP(...)").
    mic_column : str
        Column name of MIC values.
    resolution : int
        Grid resolution.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - Correlation matrix (resolution x resolution)
        - Mask matrix: True for pixels with zero variance (to be grayed out)
    """
    mic_values = df[mic_column].values
    columns = [
        f"{feature} | {i}-{j}"
        for i in range(1, resolution + 1)
        for j in range(1, resolution + 1)
    ]

    correlations = []
    mask = []
    for col in columns:
        pixel_values = df[col].values
        if np.std(pixel_values) == 0:  # type: ignore
            correlations.append(np.nan)  # Placeholder value
            mask.append(True)  # Mark as invalid
        else:
            r = np.corrcoef(pixel_values, mic_values)[0, 1]  # type: ignore
            correlations.append(0.0 if np.isnan(r) else r)
            mask.append(False)

    corr_matrix = np.array(correlations).reshape(resolution, resolution)
    mask_matrix = np.array(mask).reshape(resolution, resolution)
    return corr_matrix, mask_matrix


def plot_correlation_heatmap(
    matrix: np.ndarray,
    output_path: str = None,  # type: ignore
    cmap: Union[Colormap, str] = "Greys",
    transparent: bool = False,
    mask: np.ndarray = None,  # type: ignore
    annot_text: bool = True,
):
    """
    Visualize and save Pearson correlation heatmap with annotations and optional mask.

    Parameters
    ----------
    matrix : np.ndarray
        2D correlation matrix to visualize.
    output_path : str, optional
        File path to save the heatmap image. If None, the plot is only shown.
    cmap : str
        Colormap for the heatmap (e.g., "coolwarm", "viridis").
    transparent : bool
        Whether to use transparent background when saving the figure.
    mask : np.ndarray, optional
        Boolean mask where True marks pixels to hide (e.g., invalid correlation).
    annot_text : bool
        Whether to show numeric correlation values in each cell.
    """
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.set_facecolor("none" if transparent else "white")
    plt.gcf().set_facecolor("none" if transparent else "white")

    matrix_masked = matrix.copy()
    if mask is not None:
        matrix_masked[mask] = np.nan
    cmap_obj = cm.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#F5F5F5")

    heatmap = sns.heatmap(
        matrix_masked,
        cmap=cmap_obj,
        center=0,
        vmin=-1,
        vmax=1,
        annot=annot_text,
        fmt=".2f",
        square=True,
        cbar_kws={"label": "Pearson Correlation", "shrink": 0.8, "aspect": 20},
        annot_kws=(
            {
                "size": 8,
                "fontproperties": TIMES_NEW_ROMAN,
                "color": "black",
            }
            if annot_text
            else None
        ),
        linewidths=0.8,
        linecolor="lightgray",
    )

    ax.tick_params(length=0)
    ax.set_xticklabels(
        ax.get_xticklabels(), fontproperties=TIMES_NEW_ROMAN, fontsize=15, color="black"
    )
    ax.set_yticklabels(
        ax.get_yticklabels(), fontproperties=TIMES_NEW_ROMAN, fontsize=15, color="black"
    )

    cbar = heatmap.collections[0].colorbar
    for tick_label in cbar.ax.get_yticklabels():  # type: ignore
        tick_label.set_fontproperties(TIMES_NEW_ROMAN)
        tick_label.set_fontsize(13)
        tick_label.set_color("black")

    cbar.ax.set_ylabel(  # type: ignore
        cbar.ax.get_ylabel(),  # type: ignore
        fontproperties=TIMES_NEW_ROMAN_BD,
        fontsize=20,
        color="black",
        labelpad=10,
    )

    plt.xlabel(
        "Width (Pixel)",
        fontsize=22,
        labelpad=8,
        fontproperties=TIMES_NEW_ROMAN_BD,
        color="black",
    )
    plt.ylabel(
        "Height (Pixel)",
        fontsize=22,
        labelpad=8,
        fontproperties=TIMES_NEW_ROMAN_BD,
        color="black",
    )
    ax.set_aspect("equal")

    fig = plt.gcf()
    fig.canvas.draw()

    if transparent:
        for spine in ax.spines.values():
            spine.set_visible(False)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
            transparent=transparent,
        )
    else:
        plt.show()

    plt.close()


def plot_feature_distribution(
    df: pd.DataFrame,
    feature: str,
    output_path: str = None,  # type: ignore
    cmap: Union[Colormap, str] = "Blues",
    transparent: bool = False,
    center_at_zero: bool = False,
    mode: str = "mean",
    value_format: str = "float",
    annot_text: bool = True,
    colorbar_label: Optional[str] = None,
) -> None:
    """
    Plot the heatmap of a specific 2D feature group across all samples.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing flattened 2D features (e.g., CGR pixels).
    feature : str
        Name of the feature group (e.g., "CGR(resolution=16)", "PROP(...)").
    output_path : str, optional
        If set, save the figure to the given path. Otherwise, display it.
    cmap : str
        Colormap to apply (e.g., "Blues", "coolwarm").
    transparent : bool
        Whether to use a transparent background.
    center_at_zero : bool
        Whether to center the colormap at zero (for diverging distributions).
    mode : str
        Use "mean" or "sum" for heatmap intensity aggregation.
    value_format : str
        Format of values shown in cells ("float" or "int").
    annot_text : bool
        Whether to annotate values in each cell.
    """
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.set_facecolor("none" if transparent else "white")
    plt.gcf().set_facecolor("none" if transparent else "white")

    feature_columns = [
        f"{feature} | {i}-{j}" for i in range(1, 17) for j in range(1, 17)
    ]

    feature_data = df[feature_columns]
    values = (
        feature_data.mean().values.reshape(16, 16)  # type: ignore
        if mode == "mean"
        else feature_data.sum().values.reshape(16, 16)  # type: ignore
    )

    if value_format in ("int", "d"):
        values = np.round(values).astype(int)
        fmt = "d"
    else:
        fmt = ".2f"

    if center_at_zero:
        max_abs = max(abs(values.min()), abs(values.max()))
        vmin, vmax, center = -max_abs, max_abs, 0
    else:
        vmin = vmax = center = None

    # Determine label
    if colorbar_label is None:
        colorbar_label = (
            "Mean Feature Intensity" if mode == "mean" else "Sum Feature Intensity"
        )

    heatmap = sns.heatmap(
        values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        square=True,
        cbar=True,
        linewidths=0.8,
        linecolor="lightgray",
        annot=annot_text,
        fmt=fmt,
        annot_kws=(
            {
                "size": 8,
                "fontproperties": TIMES_NEW_ROMAN,
                "color": "black",
            }
            if annot_text
            else None
        ),
        xticklabels=np.arange(16),  # type: ignore
        yticklabels=np.arange(16),  # type: ignore
        cbar_kws={"label": colorbar_label, "shrink": 0.8, "aspect": 20},
    )

    ax.tick_params(length=0)
    ax.set_xticklabels(
        ax.get_xticklabels(), fontproperties=TIMES_NEW_ROMAN, fontsize=15, color="black"
    )
    ax.set_yticklabels(
        ax.get_yticklabels(), fontproperties=TIMES_NEW_ROMAN, fontsize=15, color="black"
    )

    cbar = heatmap.collections[0].colorbar
    for tick_label in cbar.ax.get_yticklabels():  # type: ignore
        tick_label.set_fontproperties(TIMES_NEW_ROMAN)
        tick_label.set_fontsize(13)
        tick_label.set_color("black")

    cbar.ax.set_ylabel(  # type: ignore
        cbar.ax.get_ylabel(),  # type: ignore
        fontproperties=TIMES_NEW_ROMAN_BD,
        fontsize=20,
        color="black",
        labelpad=10,
    )

    plt.xlabel(
        "Width (Pixel)",
        fontsize=22,
        labelpad=8,
        fontproperties=TIMES_NEW_ROMAN_BD,
        color="black",
    )
    plt.ylabel(
        "Height (Pixel)",
        fontsize=22,
        labelpad=8,
        fontproperties=TIMES_NEW_ROMAN_BD,
        color="black",
    )
    ax.set_aspect("equal")

    fig = plt.gcf()
    fig.canvas.draw()

    if transparent:
        for spine in ax.spines.values():
            spine.set_visible(False)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
            transparent=transparent,
        )
    else:
        plt.show()

    plt.close()


def plot_difference_distribution(
    df_mic_low: pd.DataFrame,
    df_mic_high: pd.DataFrame,
    feature: str,
    output_path: str = None,  # type: ignore
    cmap: Union[Colormap, str] = "coolwarm",
    transparent: bool = False,
    center_at_zero: bool = True,
    mode: str = "mean",
    value_format: str = "float",
    annot_text: bool = True,
    colorbar_label: Optional[str] = None,
    mask_zero: bool = True,
) -> None:
    """
    Plot a heatmap showing the difference in feature intensity between MIC-low and MIC-high groups.

    Parameters
    ----------
    df_mic_low : pd.DataFrame
        DataFrame for low MIC group samples.
    df_mic_high : pd.DataFrame
        DataFrame for high MIC group samples.
    feature : str
        Feature group name (e.g., "CGR(resolution=16)").
    output_path : str, optional
        Path to save the output figure. If None, display interactively.
    cmap : str
        Colormap to use for the heatmap.
    transparent : bool
        Whether to use transparent background.
    center_at_zero : bool
        Whether to center color scale at zero (for diverging difference).
    mode : str
        Aggregation mode: "mean" or "sum".
    value_format : str
        Display format for values: "float" or "int".
    annot_text : bool
        Whether to annotate cell values.
    colorbar_label : str, optional
        Custom label for the colorbar. Auto-determined if not provided.
    mask_zero : bool
        Whether to mask zero values (set to NaN) for neutral gray display.
    """
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.set_facecolor("none" if transparent else "white")
    plt.gcf().set_facecolor("none" if transparent else "white")

    feature_columns = [
        f"{feature} | {i}-{j}" for i in range(1, 17) for j in range(1, 17)
    ]

    diff_series = (
        df_mic_low[feature_columns].mean() - df_mic_high[feature_columns].mean()
        if mode == "mean"
        else df_mic_low[feature_columns].sum() - df_mic_high[feature_columns].sum()
    )
    diff_matrix = diff_series.values.reshape(16, 16)  # type: ignore

    diff_masked = diff_matrix.astype(float)

    if mask_zero:
        diff_matrix[diff_matrix == 0] = np.nan

    if value_format in ("int", "d"):
        display_matrix = diff_masked  # 保留 float，但用 ".0f" 顯示為整數
        fmt = ".0f"
    else:
        display_matrix = diff_masked
        fmt = ".2f"

    cmap_obj = cm.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#F5F5F5")

    if center_at_zero:
        max_abs = max(abs(diff_matrix.min()), abs(diff_matrix.max()))
        vmin, vmax, center = -max_abs, max_abs, 0
    else:
        vmin = vmax = center = None

    if colorbar_label is None:
        colorbar_label = "Mean Difference" if mode == "mean" else "Sum Difference"

    heatmap = sns.heatmap(
        display_matrix,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        center=center,
        square=True,
        cbar=True,
        linewidths=0.8,
        linecolor="lightgray",
        annot=annot_text,
        fmt=fmt,
        annot_kws=(
            {
                "size": 8,
                "fontproperties": TIMES_NEW_ROMAN,
                "color": "black",
            }
            if annot_text
            else None
        ),
        xticklabels=np.arange(16),  # type: ignore
        yticklabels=np.arange(16),  # type: ignore
        cbar_kws={"label": colorbar_label, "shrink": 0.8, "aspect": 20},
    )

    ax.tick_params(length=0)
    ax.set_xticklabels(
        ax.get_xticklabels(), fontproperties=TIMES_NEW_ROMAN, fontsize=15, color="black"
    )
    ax.set_yticklabels(
        ax.get_yticklabels(), fontproperties=TIMES_NEW_ROMAN, fontsize=15, color="black"
    )

    cbar = heatmap.collections[0].colorbar
    for tick_label in cbar.ax.get_yticklabels():  # type: ignore
        tick_label.set_fontproperties(TIMES_NEW_ROMAN)
        tick_label.set_fontsize(13)
        tick_label.set_color("black")

    cbar.ax.set_ylabel(  # type: ignore
        cbar.ax.get_ylabel(),  # type: ignore
        fontproperties=TIMES_NEW_ROMAN_BD,
        fontsize=20,
        color="black",
        labelpad=10,
    )

    plt.xlabel(
        "Width (Pixel)",
        fontsize=22,
        labelpad=8,
        fontproperties=TIMES_NEW_ROMAN_BD,
        color="black",
    )
    plt.ylabel(
        "Height (Pixel)",
        fontsize=22,
        labelpad=8,
        fontproperties=TIMES_NEW_ROMAN_BD,
        color="black",
    )
    ax.set_aspect("equal")

    fig = plt.gcf()
    fig.canvas.draw()

    if transparent:
        for spine in ax.spines.values():
            spine.set_visible(False)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
            transparent=transparent,
        )
    else:
        plt.show()

    plt.close()


# ============================== Main Function ==============================
if __name__ == "__main__":

    # Heatmap bar
    custom_cmap_positive = LinearSegmentedColormap.from_list(
        "my_white_red", ["#FFFFFF", "#FF0000"]
    )
    custom_cmap_negative = LinearSegmentedColormap.from_list(
        "my_white_blue",
        [
            "#0000FF",
            "#FFFFFF",
        ],
    )
    custom_cmap = LinearSegmentedColormap.from_list(
        "my_white_red", ["#0000FF", "#FFFFFF", "#FF0000"]
    )

    # Mapping of full strain names to their corresponding suffixes
    strains = {
        "Escherichia coli": "EC",
        "Pseudomonas aeruginosa": "PA",
        "Staphylococcus aureus": "SA",
    }

    # Loop over each strain type
    for strain_index, (strain, suffix) in enumerate(strains.items(), start=1):

        # Load data for the target strain
        predict_path = os.path.join(
            BASE_PATH, f"experiments/{suffix}/predictions/ania_test.csv"
        )
        feature_path = os.path.join(
            BASE_PATH, f"data/processed/split/{suffix}_test.csv"
        )
        merged_df = merge_prediction_and_features(predict_path, feature_path, top_n=100)

        # Compute correlation matrix
        corr_matrix, mask_matrix = compute_pixel_correlation(
            df=merged_df,
            feature="CGR(resolution=16)",
            mic_column="Predicted Log MIC Value",
            resolution=16,
        )
        plot_correlation_heatmap(
            matrix=corr_matrix,
            mask=mask_matrix,
            output_path=os.path.join(
                BASE_PATH,
                f"outputs/results/{suffix}/explainability/correlation/cgr_predicted_heatmap.png",
            ),
            cmap=custom_cmap,
            transparent=False,
            annot_text=False,
        )

        merged_df = merge_prediction_and_features(predict_path, feature_path, top_n=100)
        plot_feature_distribution(
            df=merged_df,
            feature="GradCAM(i1)",
            output_path=os.path.join(
                BASE_PATH,
                f"outputs/results/{suffix}/explainability/feature/GradCAM_i1.png",
            ),
            cmap="viridis",
            transparent=False,
            center_at_zero=False,
            mode="mean",
            value_format="float",
            annot_text=False,
            colorbar_label="Grad-CAM Score",
        )
        plot_feature_distribution(
            df=merged_df,
            feature="GradCAM(i2)",
            output_path=os.path.join(
                BASE_PATH,
                f"outputs/results/{suffix}/explainability/feature/GradCAM_i2.png",
            ),
            cmap="viridis",
            transparent=False,
            center_at_zero=False,
            mode="mean",
            value_format="float",
            annot_text=False,
            colorbar_label="Grad-CAM Score",
        )

        # 活性低
        predict_path = os.path.join(
            BASE_PATH, f"experiments/{suffix}/predictions/ania_high.csv"
        )
        feature_path = os.path.join(
            BASE_PATH, f"data/processed/group/{suffix}_high.csv"
        )
        merged_df_mic_high = merge_prediction_and_features(
            predict_path, feature_path, top_n=None  # type: ignore
        )

        # 活性高
        predict_path = os.path.join(
            BASE_PATH, f"experiments/{suffix}/predictions/ania_low.csv"
        )
        feature_path = os.path.join(BASE_PATH, f"data/processed/group/{suffix}_low.csv")
        merged_df_mic_low = merge_prediction_and_features(
            predict_path, feature_path, top_n=None  # type: ignore
        )

        plot_difference_distribution(
            df_mic_low=merged_df_mic_low,
            df_mic_high=merged_df_mic_high,
            feature="CGR(resolution=16)",
            output_path=os.path.join(
                BASE_PATH,
                f"outputs/results/{suffix}/explainability/feature/CGR_diff_heatmap.png",
            ),
            cmap=custom_cmap,
            transparent=False,
            center_at_zero=True,
            mode="sum",
            value_format="d",
            annot_text=False,
            colorbar_label="CGR Frequency Difference (High - Low)",
            mask_zero=False,
        )

        aaindex_info = {
            "ARGP820101": {
                "name": "Hydrophobicity index",
                "name_zh": "疏水性指數",
                "source": "Argos et al., 1982",
            },
            "CHAM830107": {
                "name": "Charge transfer capability parameter",
                "name_zh": "電荷轉移能力參數",
                "source": "Charton & Charton, 1983",
            },
            "FAUJ880103": {
                "name": "Normalized van der Waals volume",
                "name_zh": "範德瓦體積（標準化）",
                "source": "Fauchere et al., 1988",
            },
            "GRAR740102": {
                "name": "Polarity",
                "name_zh": "極性",
                "source": "Grantham, 1974",
            },
            "JANJ780101": {
                "name": "Average accessible surface area",
                "name_zh": "平均可接觸表面積",
                "source": "Janin et al., 1978",
            },
            "KYTJ820101": {
                "name": "Hydropathy index",
                "name_zh": "親疏水性指數",
                "source": "Kyte & Doolittle, 1982",
            },
            "NAKH920104": {
                "name": "AA composition of EXT2 in single-span proteins",
                "name_zh": "單跨膜蛋白中 EXT2 區域的胺基酸組成",
                "source": "Nakashima & Nishikawa, 1992",
            },
            "ROSM880102": {
                "name": "Side chain hydropathy (solvation-corrected)",
                "name_zh": "側鏈疏水性（溶劑校正）",
                "source": "Roseman, 1988",
            },
            "WERD780104": {
                "name": "Free energy change (ε(i) to α(Rh))",
                "name_zh": "構型轉變自由能（ε(i) → α(Rh)）",
                "source": "Wertz & Scheraga, 1978",
            },
            "ZIMJ680101": {
                "name": "Hydrophobicity",
                "name_zh": "疏水性",
                "source": "Zimmerman et al., 1968",
            },
        }

        for aaindex_id, aaindex_meta in aaindex_info.items():
            feature_name = f"PROP({aaindex_id})"
            output_path = os.path.join(
                BASE_PATH,
                f"outputs/results/{suffix}/explainability/feature/PROP_{aaindex_id}_diff_heatmap.png",
            )

            plot_difference_distribution(
                df_mic_low=merged_df_mic_low,
                df_mic_high=merged_df_mic_high,
                feature=feature_name,
                output_path=output_path,
                cmap=custom_cmap,
                transparent=False,
                center_at_zero=True,
                mode="mean",
                value_format="float",
                annot_text=False,
                colorbar_label=f"{aaindex_meta['name']} Difference (High - Low)",
                mask_zero=False,
            )
