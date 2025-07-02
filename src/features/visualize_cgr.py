# pylint: disable=line-too-long, import-error, wrong-import-position, too-many-locals
"""
This module provides visualization utilities for Chaos Game Representation (CGR)
and Frequency Chaos Game Representation (FCGR) of antimicrobial peptide (AMP) sequences.

Main functionalities:
- `visualize_cgr_trajectory()`: Visualizes the CGR path overlaid on amino acid base structure.
- `visualize_fcgr_heatmap()`: Plots the FCGR matrix as a heatmap for a given sequence.

Both functions support white or transparent background, Times New Roman font styling,
and customization of resolution and color maps.
"""

# ============================== Standard Library Imports ==============================
import os
import sys
from collections import Counter
from typing import List, Union

# ============================== Third-Party Library Imports ==============================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import seaborn as sns
from matplotlib import transforms
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

# ============================== Project-Specific Imports ==============================
# CGR Encoding Utility Functions
from src.features.cgr_encoding import compute_cgr, map_kmers


# ============================== CGR Visualization Functions ==============================
def visualize_cgr_points_with_margins(
    cgr_result: ro.vectors.ListVector,
    output_file: str = None,  # type: ignore
    transparent: bool = False,
    image_size: int = 512,
    point_size: int = 3,
    margin_ratio: float = 0.05,  # 5% 留白
):
    """
    Visualize CGR points with fixed-size canvas and small surrounding margin.

    Parameters
    ----------
    cgr_result : ro.vectors.ListVector
        R object from kaos::cgr(), containing CGR coordinates.
    output_file : str, optional
        Path to save the output image.
    transparent : bool
        If True, background is transparent.
    image_size : int
        Output image size in pixels (width = height).
    point_size : int
        Size of each point in the scatter plot.
    margin_ratio : float
        Fraction of margin around the plot (e.g., 0.05 = 5%).
    """
    traj_x = np.array(cgr_result.rx2("x"))
    traj_y = -np.array(cgr_result.rx2("y"))  # Flip Y-axis

    dpi = 100
    fig_size_inch = image_size / dpi
    fig = plt.figure(figsize=(fig_size_inch, fig_size_inch), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # type: ignore # Full canvas

    ax.set_facecolor("none" if transparent else "white")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 自動計算資料邊界 + margin
    x_min, x_max = traj_x.min(), traj_x.max()
    y_min, y_max = traj_y.min(), traj_y.max()

    x_margin = (x_max - x_min) * margin_ratio
    y_margin = (y_max - y_min) * margin_ratio

    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    # 畫點
    ax.scatter(traj_x, traj_y, s=point_size, color="black", alpha=0.9)

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(
            output_file,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.0,
            transparent=transparent,
        )
    plt.close()


def visualize_cgr_trajectory(
    cgr_result: ro.vectors.ListVector,
    output_file: str = None,  # type: ignore
    resolution: int = 16,
    transparent: bool = False,
):
    """
    Visualize CGR trajectory with amino acid base labels and connecting edges.

    Parameters
    ----------
    cgr_result : ro.vectors.ListVector
        R object returned from `kaos::cgr()`, containing base and trajectory.
    output_file : str, optional
        Path to save the output image. If None, the plot is only shown.
    resolution : int
        Grid resolution for the CGR coordinate system.
    transparent : bool
        Whether to render background as transparent.
    """
    custom_colors = {
        "Y": "#82B0D2",
        "A": "#82B0D2",
        "C": "#82B0D2",
        "D": "#82B0D2",
        "E": "#82B0D2",
        "F": "#FA7F6F",
        "G": "#FA7F6F",
        "H": "#FA7F6F",
        "I": "#FA7F6F",
        "K": "#FA7F6F",
        "L": "#8ECFC9",
        "M": "#8ECFC9",
        "N": "#8ECFC9",
        "P": "#8ECFC9",
        "Q": "#8ECFC9",
        "R": "#FF69B4",
        "S": "#FF69B4",
        "T": "#FF69B4",
        "V": "#FF69B4",
        "W": "#FF69B4",
    }

    base_x = np.array(cgr_result.rx2("base").rx2("x"))
    base_y = -np.array(cgr_result.rx2("base").rx2("y"))  # Flip Y-axis
    base_labels = np.array(ro.r["rownames"](cgr_result.rx2("base")))  # type: ignore

    plt.figure(figsize=(4, 4))
    ax = plt.gca()
    ax.set_facecolor("none" if transparent else "white")
    plt.gcf().set_facecolor("none" if transparent else "white")

    # Draw base edges
    shrink_ratio = 0.70
    for i in range(len(base_x)):
        x1, y1 = base_x[i], base_y[i]
        x2, y2 = base_x[(i + 1) % len(base_x)], base_y[(i + 1) % len(base_y)]
        plt.plot(
            [x1 + (x2 - x1) * shrink_ratio, x2 + (x1 - x2) * shrink_ratio],
            [y1 + (y2 - y1) * shrink_ratio, y2 + (y1 - y2) * shrink_ratio],
            color="black",
            linewidth=1.5,
            alpha=0.6,
            zorder=0,
        )

    # Grid
    ticks = np.linspace(-1, 1, resolution + 1)
    for t in ticks:
        plt.plot([t, t], [-1, 1], color="black", linewidth=0.6)
        plt.plot([-1, 1], [t, t], color="black", linewidth=0.6)

    # Axis arrows
    ax.annotate(
        "",
        xy=(1.2, 0),
        xytext=(-1.2, 0),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
    )
    ax.annotate(
        "",
        xy=(0, 1.2),
        xytext=(0, -1.2),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
    )

    # Base labels
    for label, x, y in zip(base_labels, base_x, base_y):
        color = custom_colors.get(label, "#000000")
        plt.text(
            x,
            y,
            label,
            fontsize=12,
            color=color,
            ha="center",
            va="center",
            fontproperties=TIMES_NEW_ROMAN_BD,
        )

    # Trajectory
    traj_x = np.array(cgr_result.rx2("x"))
    traj_y = -np.array(cgr_result.rx2("y"))
    plt.annotate(
        "",
        xy=(traj_x[0], traj_y[0]),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="red", lw=1, alpha=0.6),
    )
    for i in range(1, len(traj_x)):
        plt.annotate(
            "",
            xy=(traj_x[i], traj_y[i]),
            xytext=(traj_x[i - 1], traj_y[i - 1]),
            arrowprops=dict(arrowstyle="->", color="red", lw=1, alpha=0.6),
        )

    plt.scatter(traj_x, traj_y, color="black", s=15, alpha=0.9, zorder=3)

    # Final touches
    plt.xticks([]), plt.yticks([])  # type: ignore
    plt.xlim(-1.3, 1.3), plt.ylim(-1.3, 1.3)  # type: ignore
    ax.set_aspect("equal")
    plt.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
            transparent=transparent,
        )
    plt.close()


def visualize_fcgr_heatmap(
    cgr_result: ro.vectors.ListVector,
    cmap: Union[Colormap, str] = "Greys",
    output_file: str = None,  # type: ignore
    resolution: int = 16,
    transparent: bool = False,
):
    """
    Visualize FCGR heatmap for a single AMP sequence.

    Parameters
    ----------
    cgr_result : ro.vectors.ListVector
        R object returned from `kaos::cgr()`, containing base and matrix.
    cmap : Union[Colormap, str]
        Matplotlib colormap or custom colormap to apply (e.g., "hot", "Reds").
    output_file : str, optional
        Path to save the heatmap image. If None, the plot is only shown.
    resolution : int
        Resolution for the CGR matrix (e.g., 16 = 16x16).
    transparent : bool
        Whether to use transparent background.
    """
    x = np.array(cgr_result.rx2("x"))
    y = np.array(cgr_result.rx2("y"))

    def map_cgr_coords_to_fcgr_pixels(x_coords, y_coords, res):
        x_pixel = np.ceil((x_coords + 1) * res / 2).astype(int) - 1
        y_pixel = np.ceil((y_coords + 1) * res / 2).astype(int) - 1
        return y_pixel, x_pixel

    row_idx, col_idx = map_cgr_coords_to_fcgr_pixels(x, y, resolution)
    matrix = np.zeros((resolution, resolution), dtype=int)
    for r, c in zip(row_idx, col_idx):
        matrix[r, c] += 1

    # Plot
    plt.style.use("ggplot")
    plt.figure(figsize=(6, 5.5))
    ax = plt.gca()
    ax.set_facecolor("none" if transparent else "white")
    plt.gcf().set_facecolor("none" if transparent else "white")

    heatmap = sns.heatmap(
        matrix,
        cmap=cmap,
        square=True,
        cbar=True,
        linewidths=0.8,
        linecolor="lightgray",
        xticklabels=np.arange(resolution),  # type: ignore
        yticklabels=np.arange(resolution),  # type: ignore
        cbar_kws={"shrink": 0.7, "aspect": 20},
    )

    ax.tick_params(length=0)
    ax.set_xticklabels(
        ax.get_xticklabels(), fontproperties=TIMES_NEW_ROMAN, fontsize=11, color="black"
    )
    ax.set_yticklabels(
        ax.get_yticklabels(), fontproperties=TIMES_NEW_ROMAN, fontsize=11, color="black"
    )

    cbar = heatmap.collections[0].colorbar
    for label in cbar.ax.get_yticklabels():  # type: ignore
        label.set_fontproperties(TIMES_NEW_ROMAN)
        label.set_color("black")

    plt.xlabel(
        "Width (Pixel)",
        fontsize=11,
        labelpad=10,
        fontproperties=TIMES_NEW_ROMAN_BD,
        color="black",
    )
    plt.ylabel(
        "Height (Pixel)",
        fontsize=11,
        labelpad=10,
        fontproperties=TIMES_NEW_ROMAN_BD,
        color="black",
    )
    ax.set_aspect("equal")

    if transparent:
        for spine in ax.spines.values():
            spine.set_visible(False)

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
            transparent=transparent,
        )
    plt.close()


def extract_most_frequent_kmers(pixel_to_kmers: dict) -> dict:
    """
    For each pixel, keep only the most frequent k-mer.

    Parameters
    ----------
    pixel_to_kmers : dict
        Original mapping (row, col) -> list of k-mers.

    Returns
    -------
    dict
        Reduced mapping where each (row, col) maps to [most frequent k-mer].
    """
    new_map = {}
    for pixel, kmer_list in pixel_to_kmers.items():
        if not kmer_list:
            continue
        counter = Counter(kmer_list)
        most_common_kmer, _ = counter.most_common(1)[0]
        new_map[pixel] = [most_common_kmer]
    return new_map


def visualize_fcgr_with_kmer_labels(
    pixel_to_kmers: dict,
    output_file: str = None,  # type: ignore
    resolution: int = 16,
    transparent: bool = False,
    text_fontsize: int = 6,
):
    """
    Visualize FCGR pixels with multicolor k-mers, each letter centered using actual width.

    Parameters
    ----------
    pixel_to_kmers : dict
        Mapping (row, col) -> list of k-mers (str).
    output_file : str
        Path to save the figure.
    resolution : int
        FCGR grid resolution.
    transparent : bool
        Whether background is transparent.
    text_fontsize : int
        Font size for labels.
    """
    # Amino acid color map
    custom_colors = {
        "Y": "#82B0D2",
        "A": "#82B0D2",
        "C": "#82B0D2",
        "D": "#82B0D2",
        "E": "#82B0D2",
        "F": "#FA7F6F",
        "G": "#FA7F6F",
        "H": "#FA7F6F",
        "I": "#FA7F6F",
        "K": "#FA7F6F",
        "L": "#8ECFC9",
        "M": "#8ECFC9",
        "N": "#8ECFC9",
        "P": "#8ECFC9",
        "Q": "#8ECFC9",
        "R": "#FF69B4",
        "S": "#FF69B4",
        "T": "#FF69B4",
        "V": "#FF69B4",
        "W": "#FF69B4",
    }

    matrix = np.zeros((resolution, resolution), dtype=int)
    plt.style.use("ggplot")
    plt.figure(figsize=(6, 5.5))
    ax = plt.gca()
    ax.set_facecolor("none" if transparent else "white")
    plt.gcf().set_facecolor("none" if transparent else "white")

    # Draw empty grid
    sns.heatmap(
        matrix,
        cmap="Greys",
        square=True,
        cbar=False,
        linewidths=2.0,
        linecolor="black",
        xticklabels=np.arange(resolution),  # type: ignore
        yticklabels=np.arange(resolution),  # type: ignore
    )

    ax.tick_params(length=0)
    ax.set_xticklabels(
        ax.get_xticklabels(), fontproperties=TIMES_NEW_ROMAN, fontsize=13, color="black"
    )
    ax.set_yticklabels(
        ax.get_yticklabels(), fontproperties=TIMES_NEW_ROMAN, fontsize=13, color="black"
    )
    plt.xlabel(
        "Width (Pixel)",
        fontsize=15,
        labelpad=8,
        fontproperties=TIMES_NEW_ROMAN_BD,
        color="black",
    )
    plt.ylabel(
        "Height (Pixel)",
        fontsize=15,
        labelpad=8,
        fontproperties=TIMES_NEW_ROMAN_BD,
        color="black",
    )
    ax.set_aspect("equal")

    fig = plt.gcf()
    fig.canvas.draw()

    # Get figure-to-data coordinate transform ratio
    ax_width_px = ax.get_window_extent().width
    ax_data_width = resolution
    px_to_data_ratio = ax_data_width / ax_width_px

    for (r, c), kmers in pixel_to_kmers.items():
        x_center = c + 0.5
        y_center = r + 0.5
        for i, kmer in enumerate(kmers):
            y_offset = y_center + ((len(kmers) - 1) / 2 - i) * 0.3

            # Measure letter widths in pixels
            letter_widths_px = []
            for letter in kmer:
                t = ax.text(
                    0, 0, letter, fontsize=text_fontsize, fontproperties=TIMES_NEW_ROMAN
                )
                fig.canvas.draw()
                bbox = t.get_window_extent(renderer=fig.canvas.get_renderer())  # type: ignore
                letter_widths_px.append(bbox.width)
                t.remove()

            # Convert to data units
            letter_widths_data = [w * px_to_data_ratio for w in letter_widths_px]
            total_width_data = sum(letter_widths_data)
            start_x = x_center - total_width_data / 2

            # Draw letters
            current_x = start_x
            for letter, w in zip(kmer, letter_widths_data):
                color = custom_colors.get(letter, "#000000")
                ax.text(
                    current_x,
                    y_offset,
                    letter,
                    fontsize=text_fontsize,
                    ha="left",
                    va="center",
                    fontproperties=TIMES_NEW_ROMAN,
                    color=color,
                )
                current_x += w

    if transparent:
        for spine in ax.spines.values():
            spine.set_visible(False)

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
            transparent=transparent,
        )
    else:
        plt.show()

    plt.close()


def visualize_cgr_multiple_sequences(
    cgr_results: List[ro.vectors.ListVector],
    output_file: str = None,  # type: ignore
    transparent: bool = False,
    image_size: int = 512,
    point_size: int = 3,
    margin_ratio: float = 0.05,
):
    """
    Visualize multiple CGR point sequences overlaid on the same canvas with margins.

    Parameters
    ----------
    cgr_results : list of ro.vectors.ListVector
        List of CGR results from kaos::cgr(), each containing CGR coordinates for a sequence.
    output_file : str, optional
        Path to save the output image.
    transparent : bool
        If True, background is transparent.
    image_size : int
        Output image size in pixels (width = height).
    point_size : int
        Size of each point in the scatter plot.
    margin_ratio : float
        Fraction of margin around the plot (e.g., 0.05 = 5%).
    """
    dpi = 100
    fig_size_inch = image_size / dpi
    fig = plt.figure(figsize=(fig_size_inch, fig_size_inch), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # Full canvas

    ax.set_facecolor("none" if transparent else "white")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 收集所有序列的點座標以決定邊界
    all_x, all_y = [], []
    for cgr_result in cgr_results:
        x = np.array(cgr_result.rx2("x"))
        y = -np.array(cgr_result.rx2("y"))  # Flip Y-axis
        all_x.extend(x)
        all_y.extend(y)

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_margin = (x_max - x_min) * margin_ratio
    y_margin = (y_max - y_min) * margin_ratio

    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    # 繪製每個序列的點
    for cgr_result in cgr_results:
        traj_x = np.array(cgr_result.rx2("x"))
        traj_y = -np.array(cgr_result.rx2("y"))
        ax.scatter(traj_x, traj_y, s=point_size, color="black", alpha=0.9)

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(
            output_file,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.0,
            transparent=transparent,
        )
        plt.close()
    else:
        plt.show()


# Test script to visualize a single sequence
if __name__ == "__main__":

    # Test sequence
    # test_sequence = "FLPI"
    # test_sequence = "FLPIVGKLLSGLSGLS"
    # test_sequence = "GNTWE"
    test_sequence = "VLNEALLR"

    # Heatmap bar
    custom_cmap = LinearSegmentedColormap.from_list(
        "my_red_white", ["#FFFFFF", "#8B0000"]
    )

    # Compute CGR features for the test sequence
    sequences = [test_sequence]
    cgr_vectors, cgr_results = compute_cgr(
        sequences=sequences,
        resolution=8,
    )
    visualize_cgr_trajectory(
        cgr_result=cgr_results[0],
        output_file=os.path.join(BASE_PATH, "outputs/materials/cgr_white.png"),
        resolution=8,
        transparent=True,
    )
    visualize_fcgr_heatmap(
        cgr_results[0],
        resolution=8,
        cmap=custom_cmap,
        transparent=True,
        output_file=os.path.join(BASE_PATH, "outputs/materials/fcgr_white.png"),
    )

    # workflow
    test_sequence = "ATYDGKCYKKDNICKYKAQSGKTAICKCYVKVCPRDGAKCEFDSYKGKCYC"
    sequences = [test_sequence]
    cgr_vectors, cgr_results = compute_cgr(
        sequences=sequences,
        resolution=8,
    )
    visualize_cgr_points_with_margins(
        cgr_result=cgr_results[0],
        output_file=os.path.join(BASE_PATH, "outputs/materials/workflow_cgr1.png"),
        transparent=False,
        point_size=30,
        image_size=512,
        margin_ratio=0.03,
    )
    test_sequence = "AATAKKGAKKADAPAKPKKATKPKSPKKAAKKAGAKKGVKRAGKKGAKKTTKAKK"
    sequences = [test_sequence]
    cgr_vectors, cgr_results = compute_cgr(
        sequences=sequences,
        resolution=8,
    )
    visualize_cgr_points_with_margins(
        cgr_result=cgr_results[0],
        output_file=os.path.join(BASE_PATH, "outputs/materials/workflow_cgr2.png"),
        transparent=False,
        point_size=30,
        image_size=512,
        margin_ratio=0.03,
    )
    test_sequence = "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTESDYKDDDDKCQDSERTFY"
    sequences = [test_sequence]
    cgr_vectors, cgr_results = compute_cgr(
        sequences=sequences,
        resolution=8,
    )
    pixel_maps = map_kmers(sequences, cgr_results, resolution=8, k=3)
    visualize_cgr_points_with_margins(
        cgr_result=cgr_results[0],
        output_file=os.path.join(BASE_PATH, "outputs/materials/workflow_cgr3.png"),
        transparent=False,
        point_size=30,
        image_size=512,
        margin_ratio=0.03,
    )
    visualize_fcgr_with_kmer_labels(
        pixel_to_kmers=extract_most_frequent_kmers(pixel_maps[0]),
        output_file=os.path.join(BASE_PATH, "outputs/materials/heatmap_with_kmers.png"),
        resolution=8,
        text_fontsize=14,
        transparent=True,
    )

    # supplementary
    TOP_N = 50
    USE_LARGEST = True
    for num in range(0, TOP_N + 1, 5):
        if num == 0:
            continue
        for strain in ["SA", "EC", "PA"]:
            for level in ["high", "low"]:
                predict_path = os.path.join(
                    BASE_PATH, f"experiments/{strain}/predictions/ania_{level}.csv"
                )
                df = pd.read_csv(predict_path)

                # 檢查欄位是否存在
                if "Predicted Log MIC Value" not in df.columns:
                    raise ValueError(
                        f"'Predicted Log MIC Value' not found in {predict_path}"
                    )

                # 依照指定方向排序，並取前 100 筆
                df_sorted = df.sort_values(
                    by="Predicted Log MIC Value", ascending=not USE_LARGEST
                )
                df_top = df_sorted.head(num)

                sequence_list = df_top["Sequence"].dropna().astype(str).tolist()

                cgr_vectors, cgr_results = compute_cgr(
                    sequences=sequence_list,
                    resolution=8,
                )
                visualize_cgr_multiple_sequences(
                    cgr_results=cgr_results,
                    output_file=f"outputs/supplementary/{strain}_{level}_top{num}_cgr.png",
                    point_size=30,
                    image_size=512,
                    margin_ratio=0.03,
                )
