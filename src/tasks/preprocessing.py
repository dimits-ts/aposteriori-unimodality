import abc

import pandas as pd


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
