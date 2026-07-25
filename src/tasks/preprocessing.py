import abc

import pandas as pd


def _safe_len(x) -> int:
    try:
        return len(x)
    except Exception:
        return 0


class Dataset(abc.ABC):

    def get_dataset(self) -> pd.DataFrame:
        raise NotImplementedError()

    def get_name(self) -> str:
        raise NotImplementedError()

    def get_sdb_columns(self) -> list[str]:
        raise NotImplementedError()

    def get_annotation_column(self) -> str:
        raise NotImplementedError()

    def get_comment_key_column(self) -> str:
        raise NotImplementedError()

    # ------------------------------------------------------------------
    # Descriptive statistics
    # ------------------------------------------------------------------

    def get_annotations_per_comment(self) -> pd.Series:
        """
        Returns a Series with one entry per comment: the number of
        annotations recorded for that comment. Assumes the annotation
        column holds a list of per-annotator values for each comment (i.e.
        the dataset has already been grouped by comment).
        """
        df = self.get_dataset()
        col = self.get_annotation_column()
        return df[col].apply(_safe_len)

    def get_subgroup_counts(self) -> dict[str, pd.Series]:
        """
        For each SDB (self-disclosed background) column, flattens the
        per-comment lists of annotator characteristics across the entire
        dataset and returns value counts -- i.e. how many annotations came
        from annotators in each subgroup (e.g. Gender: {Male: 1234, ...}).
        """
        df = self.get_dataset()
        counts = {}
        for col in self.get_sdb_columns():
            counts[col] = df[col].explode().value_counts()
        return counts

    # ------------------------------------------------------------------
    # Printing / reporting
    # ------------------------------------------------------------------

    def print_subgroup_counts(self) -> None:
        """
        Prints, for this dataset, the count of annotations belonging to
        each subgroup within every personal characteristic (SDB) dimension.
        """
        print(f"=== Subgroup counts for {self.get_name()} ===")
        for col, series in self.get_subgroup_counts().items():
            print(f"\n--- {col} ---")
            print(series.to_string())
        print()

    @staticmethod
    def _fmt_num(x: float) -> str:
        """Formats a float to 4 decimal places, stripping trailing zeros
        (and a trailing decimal point) so whole numbers print cleanly."""
        return f"{x:.4f}".rstrip("0").rstrip(".")

    @staticmethod
    def print_annotation_count_table(datasets: list["Dataset"]) -> None:
        """
        Prints a LaTeX table* -- matching the descriptive-statistics format
        used in the paper -- summarizing the distribution of the number of
        annotations per comment, across the given list of datasets. E.g.:

            Dataset.print_annotation_count_table([ds_350, ds_990, kumar, sap])
        """
        rows = []
        for ds in datasets:
            desc = ds.get_annotations_per_comment().describe()
            rows.append({"dataset": ds.get_name(), **desc.to_dict()})
        stats_df = pd.DataFrame(rows).set_index("dataset")

        print("\\begin{table*}[t]")
        print("\t\\centering")
        print(
            "\t\\caption{Descriptive statistics for the number of "
            "annotations per comment, grouped by dataset.}"
        )
        print("\t\\label{tab:num-annot}")
        print("\t\\begin{tabular}{lrrrrrrrr}")
        print("\t\t\\toprule")
        print(
            "\t\t& count & mean & std & min & 25\\% & 50\\% & 75\\% & "
            "max \\\\"
        )
        print("\t\t\\midrule")
        for name, row in stats_df.iterrows():
            print(
                f"\t\t{name} & {int(row['count'])} & "
                f"{Dataset._fmt_num(row['mean'])} & "
                f"{Dataset._fmt_num(row['std'])} & "
                f"{Dataset._fmt_num(row['min'])} & "
                f"{Dataset._fmt_num(row['25%'])} & "
                f"{Dataset._fmt_num(row['50%'])} & "
                f"{Dataset._fmt_num(row['75%'])} & "
                f"{Dataset._fmt_num(row['max'])} \\\\"
            )
        print("\t\t\\bottomrule")
        print("\t\\end{tabular}")
        print("\\end{table*}")

    @staticmethod
    def print_descriptive_statistics(datasets: list["Dataset"]) -> None:
        """
        General-purpose report for a list of datasets: prints the combined
        LaTeX annotations-per-comment table, followed by per-dataset
        subgroup counts for every personal characteristic (SDB) dimension.
        """
        Dataset.print_annotation_count_table(datasets)
        print()
        for ds in datasets:
            ds.print_subgroup_counts()


class SubsampledView:
    """
    Thin wrapper around a Dataset that overrides get_dataset() to return a
    subsampled DataFrame, delegating every other method/attribute (column
    accessors, get_name, etc.) to the wrapped dataset. This lets
    subsample_dataset work on any Dataset subclass, not just dataset-specific
    ones such as DicesDataset.
    """

    def __init__(self, base_dataset: Dataset, df: pd.DataFrame):
        self._base = base_dataset
        self._df = df

    def get_dataset(self) -> pd.DataFrame:
        return self._df

    def __getattr__(self, name):
        return getattr(self._base, name)
