    # pylint: disable=line-too-long, import-error, wrong-import-position, too-many-locals
    """ """
    # ============================== Standard Library Imports ==============================
    import logging
    import os
    import sys
    from collections import defaultdict
    from pydoc import text

    # ============================== Third-Party Library Imports ==============================
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import rpy2.robjects as ro
    import seaborn as sns
    from matplotlib.font_manager import FontProperties
    from rpy2.robjects.packages import importr
    from rpy2.robjects.vectors import StrVector

    # 設定 Times New Roman 字體（你可根據你的系統路徑調整）
    TIMES_NEW_ROMAN = FontProperties(
        fname="/usr/share/fonts/truetype/msttcorefonts/times.ttf"
    )
    TIMES_NEW_ROMAN_BD = FontProperties(
        fname="/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf"
    )

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
            cgr_matrix = np.array(ro.r("matrix")(cgr_result.rx2("matrix")))  # type: ignore

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


    def merge_pixel_kmer_maps(pixel_kmer_maps: list[dict]) -> dict:
        """
        Merge pixel-to-kmer mappings from multiple sequences into a combined frequency map.

        Parameters
        ----------
        pixel_kmer_maps : list of dict
            A list where each element is a dict mapping (row, col) to list of k-mers.

        Returns
        -------
        dict
            A dictionary mapping (row, col) to another dictionary of {kmer: count}.
        """
        merged = defaultdict(lambda: defaultdict(int))  # (row, col) -> {kmer: count}

        for seq_map in pixel_kmer_maps:
            for pixel, kmer_list in seq_map.items():
                for kmer in kmer_list:
                    merged[pixel][kmer] += 1

        # Convert defaultdicts to normal dicts for cleaner downstream usage
        merged = {pixel: dict(kmer_counts) for pixel, kmer_counts in merged.items()}
        return merged


    def load_aaindex_table(csv_path: str) -> pd.DataFrame:
        """
        Load AAindex-like property table from CSV file.

        Parameters
        ----------
        csv_path : str
            Path to the AAindex property CSV file.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by amino acid single-letter code.
        """
        df = pd.read_csv(csv_path)
        df.set_index("AminoAcid", inplace=True)
        return df


    def compute_property_matrix(
        grid_kmer_dict: dict, aaindex_df: pd.DataFrame, prop_id: str, resolution: int
    ) -> np.ndarray:
        """
        Compute a property matrix based on amino acid property values,
        using sum of AA property values in all kmers, weighted by kmer frequency.

        Parameters
        ----------
        grid_kmer_dict : dict
            Dictionary mapping (row, col) to a dict of {kmer: count}.
        aaindex_df : pd.DataFrame
            AAindex property table (indexed by amino acid letters).
        prop_id : str
            AAindex ID to extract (e.g., "ZIMJ680101").
        resolution : int
            Resolution of the CGR grid.

        Returns
        -------
        np.ndarray
            A matrix where each cell contains the weighted average AA property.
        """
        prop_matrix = np.zeros((resolution, resolution))

        for (row, col), kmer_dict in grid_kmer_dict.items():
            total_score = 0.0
            total_kmer_count = 0

            for kmer, count in kmer_dict.items():
                try:
                    aa_values = [
                        aaindex_df.at[aa, prop_id] for aa in kmer if aa in aaindex_df.index
                    ]
                    if len(aa_values) == len(kmer):
                        total_score += sum(aa_values) * count
                        total_kmer_count += count
                except KeyError:
                    continue

            if total_kmer_count > 0:
                prop_matrix[row, col] = total_score / total_kmer_count
                # prop_matrix[row, col] = total_score

        return prop_matrix


    def compute_property_matrix_dominant_kmer(
        grid_kmer_dict: dict, aaindex_df: pd.DataFrame, prop_id: str, resolution: int
    ) -> np.ndarray:
        """
        Compute a property matrix using the property value of the most frequent k-mer
        in each CGR grid cell.

        Parameters
        ----------
        grid_kmer_dict : dict
            Dictionary mapping (row, col) to {kmer: count}.
        aaindex_df : pd.DataFrame
            AAindex property table (indexed by amino acid letters).
        prop_id : str
            AAindex ID to extract (e.g., "ZIMJ680101").
        resolution : int
            CGR grid resolution (e.g., 16 for 16x16).

        Returns
        -------
        np.ndarray
            Property matrix using the top-frequency k-mer's property per grid.
        """
        prop_matrix = np.zeros((resolution, resolution))

        for (row, col), kmer_dict in grid_kmer_dict.items():
            # 找出出現次數最多的 k-mer
            most_common_kmer = max(kmer_dict.items(), key=lambda x: x[1])[0]

            try:
                aa_values = [
                    aaindex_df.at[aa, prop_id]
                    for aa in most_common_kmer
                    if aa in aaindex_df.index
                ]
                if len(aa_values) == len(most_common_kmer):
                    # 可以選擇 sum or average，這邊預設取平均
                    prop_matrix[row, col] = sum(aa_values) / len(aa_values)
                    # prop_matrix[row, col] = sum(aa_values)
            except KeyError:
                continue  # 遇到未知氨基酸時跳過

        return prop_matrix


    def plot_property_heatmap(
        matrix: np.ndarray,
        cmap: str = "coolwarm",
        title: str = "Property Heatmap",
        output_file: str = None,  # type: ignore
        transparent: bool = False,
    ):
        """
        Plot a heatmap with values annotated on each cell.

        Parameters
        ----------
        matrix : np.ndarray
            2D property matrix to visualize.
        cmap : str
            Colormap to use (e.g., "viridis", "coolwarm").
        title : str
            Title of the heatmap.
        output_file : str, optional
            Path to save the image. If None, it will just show.
        transparent : bool
            Whether to use transparent background.
        """
        plt.figure(figsize=(6, 5.5))
        ax = sns.heatmap(
            matrix,
            cmap=cmap,
            annot=True,
            fmt=".2f",
            square=True,
            cbar=True,
            linewidths=0.5,
            linecolor="gray",
            annot_kws={"fontsize": 9, "fontproperties": TIMES_NEW_ROMAN},
            xticklabels=np.arange(matrix.shape[1]),  # type: ignore
            yticklabels=np.arange(matrix.shape[0]),  # type: ignore
            cbar_kws={"shrink": 0.8},
        )

        ax.set_title(title, fontsize=14, fontproperties=TIMES_NEW_ROMAN_BD)
        ax.set_xticklabels(ax.get_xticklabels(), fontproperties=TIMES_NEW_ROMAN)
        ax.set_yticklabels(ax.get_yticklabels(), fontproperties=TIMES_NEW_ROMAN)

        plt.xlabel("Column", fontproperties=TIMES_NEW_ROMAN_BD)
        plt.ylabel("Row", fontproperties=TIMES_NEW_ROMAN_BD)

        if transparent:
            ax.set_facecolor("none")
            plt.gcf().set_facecolor("none")

        plt.tight_layout()
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight", transparent=transparent)
            print(f"Heatmap saved to: {output_file}")
        else:
            plt.show()

        plt.close()


    def load_fasta_sequences(fasta_path: str) -> list[str]:
        """
        Load amino acid sequences from a FASTA file using plain Python.

        Parameters
        ----------
        fasta_path : str
            Path to the FASTA file.

        Returns
        -------
        list of str
            List of amino acid sequences as uppercase strings.
        """
        sequences = []
        current_seq = []

        with open(fasta_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # skip empty lines
                if line.startswith(">"):
                    if current_seq:
                        sequences.append("".join(current_seq).upper())
                        current_seq = []
                else:
                    current_seq.append(line)
            if current_seq:  # append last sequence
                sequences.append("".join(current_seq).upper())

        # 過濾非天然氨基酸
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        sequences = [seq for seq in sequences if all(c in valid_aa for c in seq)]

        return sequences


    if __name__ == "__main__":

        for strain in ["EC", "PA", "SA"]:
            fasta_path = os.path.join(BASE_PATH, f"data/processed/all/{strain}.fasta")
            sequences = load_fasta_sequences(fasta_path)
            print(f"[INFO] Loaded {len(sequences)} valid sequences from FASTA.")

            resolution = 16
            k = 3

            cgr_vectors, cgr_results = compute_cgr(sequences, resolution=resolution)
            pixel_kmer_maps = map_kmers(sequences, cgr_results, resolution=resolution, k=k)
            merged_result = merge_pixel_kmer_maps(pixel_kmer_maps)

            aaindex_df = load_aaindex_table("configs/AAindex_properties.csv")

            AAINDEX_DESCRIPTIONS = {
                "ARGP820101": "Hydrophobicity (Argos et al., 1982)",
                "CHAM830107": "Side chain volume (Chothia, 1976)",
                "FAUJ880103": "Normalized van der Waals volume",
                "GRAR740102": "Polarity (Grantham, 1974)",
                "JANJ780101": "Isoelectric point (Janin, 1978)",
                "KYTJ820101": "Hydrophilicity (Kyte-Doolittle scale)",
                "NAKH920104": "Accessible surface area (Nakashima et al., 1992)",
                "ROSM880102": "Steric parameter",
                "WERD780104": "Electron-ion interaction potential",
                "ZIMJ680101": "Hydrophobicity (Zimmerman et al., 1968)",
            }

            for _, (id, text) in enumerate(AAINDEX_DESCRIPTIONS.items(), start=1):
                prop_matrix_avg = compute_property_matrix(
                    grid_kmer_dict=merged_result,
                    aaindex_df=aaindex_df,
                    prop_id=id,
                    resolution=resolution,
                )
                prop_matrix_max = compute_property_matrix_dominant_kmer(
                    grid_kmer_dict=merged_result,
                    aaindex_df=aaindex_df,
                    prop_id=id,
                    resolution=resolution,
                )
                os.makedirs(os.path.join("heatmap/all", strain), exist_ok=True)
                plot_property_heatmap(
                    matrix=prop_matrix_avg,
                    cmap="coolwarm",
                    title=f"{text}",
                    output_file=f"heatmap/all/{strain}/avg_{id}.png",
                    transparent=False,
                )
                plot_property_heatmap(
                    matrix=prop_matrix_max,
                    cmap="coolwarm",
                    title=f"{text}",
                    output_file=f"heatmap/all/{strain}/max_{id}.png",
                    transparent=False,
                )

            for group in ["high", "medium", "low"]:
                fasta_path = os.path.join(
                    BASE_PATH, f"data/processed/group/{strain}_{group}.fasta"
                )
                sequences = load_fasta_sequences(fasta_path)
                print(f"[INFO] Loaded {len(sequences)} valid sequences from FASTA.")

                resolution = 16
                k = 3

                cgr_vectors, cgr_results = compute_cgr(sequences, resolution=resolution)
                pixel_kmer_maps = map_kmers(
                    sequences, cgr_results, resolution=resolution, k=k
                )
                merged_result = merge_pixel_kmer_maps(pixel_kmer_maps)

                aaindex_df = load_aaindex_table("configs/AAindex_properties.csv")

                AAINDEX_DESCRIPTIONS = {
                    "ARGP820101": "Hydrophobicity (Argos et al., 1982)",
                    "CHAM830107": "Side chain volume (Chothia, 1976)",
                    "FAUJ880103": "Normalized van der Waals volume",
                    "GRAR740102": "Polarity (Grantham, 1974)",
                    "JANJ780101": "Isoelectric point (Janin, 1978)",
                    "KYTJ820101": "Hydrophilicity (Kyte-Doolittle scale)",
                    "NAKH920104": "Accessible surface area (Nakashima et al., 1992)",
                    "ROSM880102": "Steric parameter",
                    "WERD780104": "Electron-ion interaction potential",
                    "ZIMJ680101": "Hydrophobicity (Zimmerman et al., 1968)",
                }

                for _, (id, text) in enumerate(AAINDEX_DESCRIPTIONS.items(), start=1):
                    prop_matrix_avg = compute_property_matrix(
                        grid_kmer_dict=merged_result,
                        aaindex_df=aaindex_df,
                        prop_id=id,
                        resolution=resolution,
                    )
                    prop_matrix_max = compute_property_matrix_dominant_kmer(
                        grid_kmer_dict=merged_result,
                        aaindex_df=aaindex_df,
                        prop_id=id,
                        resolution=resolution,
                    )
                    os.makedirs(os.path.join("heatmap/group", strain), exist_ok=True)
                    plot_property_heatmap(
                        matrix=prop_matrix_avg,
                        cmap="coolwarm",
                        title=f"{text}",
                        output_file=f"heatmap/group/{strain}/avg_{group}_{id}.png",
                        transparent=False,
                    )
                    plot_property_heatmap(
                        matrix=prop_matrix_max,
                        cmap="coolwarm",
                        title=f"{text}",
                        output_file=f"heatmap/group/{strain}/max_{group}_{id}.png",
                        transparent=False,
                    )