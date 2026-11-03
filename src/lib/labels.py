"""
Display labels for exported figures and LaTeX tables.

Everything here is applied *at export time only*: the analysis, the caches
and all file names keep using the internal names (dataset keys such as
"sap", Dataset.get_name() values such as "Sap et al. 2022", and model
pseudonyms such as "gemma7b"). The export functions translate those names
into display labels just before drawing a figure or writing a table, so
changing a label here never invalidates a cache or renames a file.

Unknown names pass through unchanged, so new datasets/models still export
(with their internal name) without an entry here.
"""

import re
from dataclasses import dataclass

import pandas as pd

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

# Dataset key -> (full label, short label). The short label is only used in
# tables that have to be narrow to fit (e.g. the significance tables).
_DATASETS = {
    "dices-350": ("DICES-350", "DICES-350"),
    "dices-990": ("DICES-990", "DICES-990"),
    "sap": ("Sap", "Sap"),
    "kumar": ("Kumar", "Kumar"),
    "popquorn": ("POPQUORN", "POP"),
}

# Other internal spellings that reach export functions -> dataset key:
# Dataset.get_name() values and the human file-name stems.
_DATASET_ALIASES = {
    "DICES-350": "dices-350",
    "DICES-990": "dices-990",
    "Sap et al. 2022": "sap",
    "Kumar et al. 2021": "kumar",
    "POPQUORN": "popquorn",
}


def dataset_label(name: str, short: bool = False) -> str:
    """
    Display label of a dataset, given its key ("sap"), its
    Dataset.get_name() value ("Sap et al. 2022") or a file-name stem
    ("dices-350"). With `short`, the narrow-table label (e.g. "POP").
    Unknown names are returned unchanged.
    """
    key = _DATASET_ALIASES.get(name, name)
    if key not in _DATASETS:
        return name
    full, short_label = _DATASETS[key]
    return short_label if short else full


def short_dataset_label(name: str) -> str:
    """dataset_label(name, short=True), for use with Series.map."""
    return dataset_label(name, short=True)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# Model pseudonym (as in the annotation file names) -> display label.
# Note: gemma7b is deliberately shown as "Gemma 5B".
MODEL_LABELS = {
    "llama8b": "Llama 8B",
    "olmo7b": "OLMo 7B",
    "olmo32b": "OLMo 32B",
    "qwen7b": "Qwen 7B",
    "qwen32b": "Qwen 32B",
    "mistral7b": "Mistral 7B",
    "mistral24b": "Mistral 24B",
    "gemma7b": "Gemma 5B",
    "gemma31b": "Gemma 31B",
}

# Model family -> display name, for pseudonyms not in MODEL_LABELS.
_MODEL_FAMILIES = {
    "llama": "Llama",
    "olmo": "OLMo",
    "qwen": "Qwen",
    "mistral": "Mistral",
    "gemma": "Gemma",
}
_PSEUDO_PATTERN = re.compile(r"^([a-z]+)(\d+(?:\.\d+)?)([bm])$")


def model_label(pseudo: str) -> str:
    """
    Display label of a model pseudonym, e.g. "qwen32b" -> "Qwen 32B".
    Pseudonyms not in MODEL_LABELS but of the form <family><size>b are
    formatted the same way; anything else is returned unchanged.
    """
    if pseudo in MODEL_LABELS:
        return MODEL_LABELS[pseudo]
    match = _PSEUDO_PATTERN.match(str(pseudo))
    if match is None:
        return pseudo
    family, size, unit = match.groups()
    family = _MODEL_FAMILIES.get(family, family.capitalize())
    return f"{family} {size}{unit.upper()}"


# ---------------------------------------------------------------------------
# Acronyms (mirror the paper's \DeclareAcronym entries)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Acronym:
    """
    `key` is the acronym package key, used as \\ac{key} in LaTeX tables
    and captions. `short` is the plain short form, used in matplotlib
    figures, whose LaTeX run does not load the paper's acronym package.
    """

    key: str
    short: str

    @property
    def tex(self) -> str:
        return rf"\ac{{{self.key}}}"

    @property
    def tex_plural(self) -> str:
        return rf"\acp{{{self.key}}}"

    def __str__(self) -> str:
        return self.short


IAA = Acronym("iaa", "IAA")
SDD = Acronym("sdd", "SDD")
AU = Acronym("au", "AU")
ASD = Acronym("atsysdis", "ASD")
USD = Acronym("unpol", "USD")
NDFU = Acronym("ndfu", "nDFU")
GMM = Acronym("gmm", "GMM")


# ---------------------------------------------------------------------------
# DataFrame helper for table exports
# ---------------------------------------------------------------------------

# Column / index-level name -> labeller, for relabel_frame.
_MODEL_LEVELS = ("Model", "Source")
_DATASET_LEVELS = ("Dataset",)


def relabel_frame(df, short: bool = False):
    """
    Copy of `df` in which the values of every column and index level named
    "Dataset" are replaced by dataset labels (short ones with `short`) and
    those named "Model" or "Source" by model labels. Anything else,
    including unknown names, is left as is.
    """

    def labeller(name):
        if name in _DATASET_LEVELS:
            return short_dataset_label if short else dataset_label
        if name in _MODEL_LEVELS:
            return model_label
        return None

    df = df.copy()
    for col in df.columns:
        fn = labeller(col)
        if fn is not None:
            df[col] = df[col].map(
                lambda v, fn=fn: fn(v) if isinstance(v, str) else v
            )

    index = df.index
    if isinstance(index, pd.MultiIndex):
        fns = [labeller(n) for n in index.names]
        if any(fns):
            df.index = pd.MultiIndex.from_tuples(
                [
                    tuple(
                        fn(v) if fn is not None and isinstance(v, str) else v
                        for fn, v in zip(fns, row)
                    )
                    for row in index
                ],
                names=index.names,
            )
    else:
        fn = labeller(index.name)
        if fn is not None:
            df.index = index.map(lambda v: fn(v) if isinstance(v, str) else v)
    return df
