import abc
from pathlib import Path
from typing import Callable

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

    def get_text_column(self) -> str:
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


class LazyDatasetLoader:
    """
    Singleton class for any dataset that uses lazy loading.
    """

    def __init__(self, factory: Callable[[], Dataset]) -> None:
        self._factory = factory
        self._dataset: Dataset | None = None

    def get(self) -> Dataset:
        if self._dataset is None:
            self._dataset = self._factory()
        return self._dataset


class DicesDataset(Dataset):
    def __init__(self, dataset_path: Path, variant: str):
        self.variant = variant
        self.df = self._base_df(dataset_path)

    def get_name(self) -> str:
        return "DICES-" + self.variant

    def get_dataset(self) -> pd.DataFrame:
        return self.df

    def get_sdb_columns(self) -> list[str]:
        return ["Gender", "Race", "Age", "Education"]

    def get_comment_key_column(self) -> str:
        return "item_id"

    def get_text_column(self) -> str:
        return "text"

    def get_annotation_column(self) -> str:
        return "is_harmful"

    def _base_df(self, dataset_path: Path) -> pd.DataFrame:
        df = pd.read_csv(dataset_path)

        if self.variant == "350":
            target_label = "Q3_bias_targeting_inherited_attributes"
        elif self.variant == "990":
            target_label = "Q3_bias_incites_hatred"
        else:
            raise ValueError(f"Variant must be 990 and 350 not {self.variant}")

        df = df.loc[
            :,
            [
                "rater_gender",
                "rater_age",
                "rater_race",
                "rater_education",
                target_label,
                "item_id",
                "context",
                "response",
            ],
        ]
        # "context" is the conversation so far and "response" is the final
        # chatbot turn being rated; concatenate into a single text field.
        df["text"] = (
            df["context"].fillna("") + "\n" + df["response"].fillna("")
        )
        df = df.drop(columns=["context", "response"])
        df[target_label] = (
            df[target_label].map({"No": -1, "Unsure": 0, "Yes": 1}).astype(int)
        )

        df = df.replace(
            {
                "College degree or higher": "College +",
                "High school or below": "High school -",
            }
        )
        df = df.replace(
            {
                "Asian/Asian subcontinent": "Asian",
                "Black/African American": "African Am.",
                "LatinX, Latino, Hispanic or Spanish Origin": "Latino",
                "Self-describe (below)": "Other",
            }
        )
        # add numbers for proper ordering during export
        df = df.replace(
            {
                "gen x+": "3) Gen. X+",
                "millenial": "2) Millennial",
                "gen z": "1) Gen. Z",
            }
        )

        agg = {
            col: list for col in df.columns if col not in ("item_id", "text")
        }
        agg["text"] = "first"
        df = df.groupby("item_id").agg(agg).reset_index()
        df = df.rename(
            columns={
                "rater_gender": "Gender",
                "rater_age": "Age",
                "rater_race": "Race",
                "rater_education": "Education",
                target_label: "is_harmful",
            }
        )
        return df


class KumarDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        num_samples: int | None = None,
        seed: int = 42,
    ):
        self.df = self._remove_invalid_ann_counts(
            KumarDataset._base_df(dataset_path, num_samples, seed)
        )

    def get_name(self) -> str:
        return "Kumar et al. 2021"

    def get_dataset(self) -> pd.DataFrame:
        return self.df

    def get_sdb_columns(self) -> list[str]:
        return [
            "Gender",
            "Ethnicity",
            "Age",
            "Education",
            "Sexual Orientation",
            "Is Transgender",
            "Political Affiliation",
            "Is Parent",
            "Technology Impact",
            "Toxicity Problem",
            "Religion Important",
            "Seen Toxicity",
            "Has Been Targeted",
        ]

    def get_comment_key_column(self) -> str:
        return "comment"

    def get_annotation_column(self) -> str:
        return "Toxicity"

    def get_text_column(self) -> str:
        return "comment"

    @staticmethod
    def _base_df(
        dataset_path: Path, num_samples: int | None, seed: int = 42
    ) -> pd.DataFrame:
        df = pd.read_json(dataset_path, lines=True)
        df = df.explode(column="ratings")
        df = df.dropna()

        ratings_df = pd.json_normalize(df.ratings)
        df = pd.concat([df.reset_index(), ratings_df.reset_index()], axis=1)
        df = df.drop(columns=["ratings", "index"])
        # shorten names
        df = df.replace(
            {
                (
                    "High school graduate (high school diploma or equivalent "
                    "including GED)"
                ): "High School graduate",
                "Associate degree in college (2-year)": "Associate degree",
                "Bachelor's degree in college (4-year)": "Bachelor's degree",
                "Less than high school degree": "No high school",
                "Professional degree (JD, MD)": "Professional degree",
                "Some college but no degree": "College, no degree",
            }
        )
        # define ranking from most to least qualified
        ranking = [
            "Doctoral degree",
            "Professional degree",
            "Master's degree",
            "Bachelor's degree",
            "Associate degree",
            "College, no degree",
            "High School graduate",
            "No high school",
        ]
        ranking.reverse()

        # create a mapping with ordinal prefix: 1), 2), 3)...
        ordinal_map = {
            name: f"{i+1}) {name}" for i, name in enumerate(ranking)
        }

        # apply the new labels
        df["education"] = df["education"].replace(ordinal_map)

        df = df.replace(
            {
                "Very important": "4) Very",
                "Somewhat important": "3) Somewhat",
                "Not too important": "2) Not very",
                "Not important": "1) No",
            }
        )
        df = df.replace(
            {
                "Very frequently a problem": "5) Very Frequently",
                "Frequently a problem": "4) Frequently",
                "Occasionally a problem": "3) Occasionally",
                "Rarely a problem": "2) Rarely",
                "Not a problem": "1) Never",
            }
        )
        df = df.replace(
            {
                "Very positive": "5) Very positive",
                "Somewhat positive": "4) Somewhat positive",
                # wtf?
                "Neutral \u00e2\u0080\u0093 neither positive nor negative": "3) Neutral",
                "Somewhat negative": "2) Somewhat negative",
                "Very negative": "1) Very negative",
            }
        )

        age_ranking = [
            "18 - 24",
            "25 - 34",
            "35 - 44",
            "45 - 54",
            "55 - 64",
            "65 or older",
        ]
        age_ordinal_map = {
            name: f"{i+1}) {name}" for i, name in enumerate(age_ranking)
        }
        df.age_range = df.age_range.replace(age_ordinal_map)

        df = df.loc[
            :,
            [
                "comment",
                "toxic_score",
                "gender",
                "race",
                "personally_seen_toxic_content",
                "personally_been_target",
                "identify_as_transgender",
                "toxic_comments_problem",
                "education",
                "age_range",
                "lgbtq_status",
                "political_affilation",  # sic
                "is_parent",
                "religion_important",
                "technology_impact",
            ],
        ]
        df.race = df.race.apply(KumarDataset._simplify_ethnicity)
        df = df.groupby("comment").agg(list)

        if num_samples is not None:
            print(
                f"Selecting {num_samples} out of {len(df)} total comments "
                f"(seed={seed})."
            )
            df = df.sample(num_samples, random_state=seed)

        df = df.reset_index()

        df = df.rename(
            columns={
                "personally_seen_toxic_content": "Seen Toxicity",
                "personally_been_target": "Has Been Targeted",
                "identify_as_transgender": "Is Transgender",
                "toxic_comments_problem": "Toxicity Problem",
                "education": "Education",
                "age_range": "Age",
                "lgbtq_status": "Sexual Orientation",
                "political_affilation": "Political Affiliation",
                "is_parent": "Is Parent",
                "religion_important": "Religion Important",
                "toxic_score": "Toxicity",
                "gender": "Gender",
                "race": "Ethnicity",
                "technology_impact": "Technology Impact",
            }
        )
        return df

    @staticmethod
    def _simplify_ethnicity(x):
        if isinstance(x, list):
            # If your field is a list (after aggregation)
            x = x[0]

        if pd.isna(x):
            return "Unknown"

        if "," in x:
            return "Multiracial"

        mapping = {
            "Asian": "Asian",
            "Black or African American": "Black",
            "Hispanic": "Hispanic",
            "White": "White",
            "Other": "Other",
            "Prefer not to say": "Unknown",
        }
        return mapping.get(x, "Other")

    @staticmethod
    def _remove_invalid_ann_counts(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        # --- There is a single comment with 650 annotators ---
        df["annotator_count"] = df["Toxicity"].apply(_safe_len)

        over_10_mask = df["annotator_count"] > 10

        if over_10_mask.any():
            over_10_df = pd.DataFrame(
                {
                    "comment": df.index[over_10_mask],
                    "annotator_count": df.loc[over_10_mask, "annotator_count"],
                }
            ).sort_values("annotator_count", ascending=False)
            print(f"#Comments with >10 annotators:{len(over_10_df)}")

        df = df.loc[~over_10_mask].drop(columns=["annotator_count"])
        return df


class SapDataset(Dataset):
    def __init__(self, dataset_path: Path):
        self.df = SapDataset._base_df(dataset_path)

    def get_name(self) -> str:
        return "Sap et al. 2022"

    def get_dataset(self) -> pd.DataFrame:
        return self.df

    def get_sdb_columns(self) -> list[str]:
        return [
            "Age",
            "Ethnicity",
            "Gender",
        ]

    def get_comment_key_column(self) -> str:
        return "tweet"

    def get_annotation_column(self) -> str:
        return "Racism"

    def get_text_column(self) -> str:
        return "tweet"

    @staticmethod
    def _base_df(dataset_path: Path) -> pd.DataFrame:
        df = pd.read_pickle(dataset_path)
        df = df.loc[
            :,
            [
                "tweet",
                "racism",
                "annotatorAge",
                "annotatorRace",
                "annotatorGender",
            ],
        ]
        df.annotatorAge = df.annotatorAge.apply(SapDataset._map_generation)
        df.annotatorRace = df.annotatorRace.apply(
            lambda x: None if ("na" in x) else x
        )

        df.annotatorGender = df.annotatorGender.apply(
            lambda x: None if ("na" in x) else x
        )
        df = df.dropna()

        df = df.rename(
            columns={
                "racism": "Racism",
                "annotatorAge": "Age",
                "annotatorRace": "Ethnicity",
                "annotatorGender": "Gender",
            }
        )
        return df

    @staticmethod
    def _map_generation(age_list):
        if age_list is None or not isinstance(age_list, (list, tuple)):
            return None

        gens = []
        for a in age_list:
            if pd.isna(a):
                continue
            age = int(a)

            # reference year: 2022
            if age < 26:
                gens.append("3) Gen. Z")
            elif age < 41:
                gens.append("2) Millennial")
            else:
                gens.append("1) Gen. X+")

        return gens if len(gens) > 0 else None


class PopquornDataset(Dataset):
    """
    POPQUORN (Pei & Jurgens, 2023):
    https://github.com/Jiaxin-Pei/potato-prolific-dataset

    Only the two rating tasks are supported (the question-answering and
    email-rewriting tasks have free-text outputs, not per-annotator labels):

    - ``offensiveness``: 1-5 offensiveness rating
    - ``politeness_rating``: 1-5 politeness rating

    Each task is exposed as its own dataset, analogous to the DICES variants.
    Annotations carrying a "Prefer not to disclose" value in any SDB column
    are dropped, mirroring how SapDataset drops "na" values.
    """

    UNDISCLOSED = "Prefer not to disclose"

    def __init__(self, dataset_path: Path):
        self.df = self._base_df(dataset_path)

    def get_name(self) -> str:
        return "POPQUORN"

    def get_dataset(self) -> pd.DataFrame:
        return self.df

    def get_sdb_columns(self) -> list[str]:
        return ["Gender", "Race", "Age", "Education", "Occupation"]

    def get_comment_key_column(self) -> str:
        return "instance_id"

    def get_text_column(self) -> str:
        return "text"

    def get_annotation_column(self) -> str:
        return "Rating"

    def _base_df(self, dataset_path: Path) -> pd.DataFrame:
        label = "offensiveness"
        df = pd.read_csv(dataset_path)
        df = df.loc[
            :,
            [
                "instance_id",
                "text",
                label,
                "gender",
                "race",
                "age",
                "education",
                "occupation",
            ],
        ]
        df = df.rename(
            columns={
                label: "Rating",
                "gender": "Gender",
                "race": "Race",
                "age": "Age",
                "education": "Education",
                "occupation": "Occupation",
            }
        )

        sdb_cols = self.get_sdb_columns()
        df[sdb_cols] = df[sdb_cols].replace(self.UNDISCLOSED, None)
        df = df.dropna()
        df["Rating"] = df["Rating"].astype(int)

        df["Race"] = df["Race"].replace(
            {
                "Black or African American": "Black",
                "Hispanic or Latino": "Hispanic",
                "Native Hawaiian or Pacific Islander": "Pacific Islander",
                # typo in the politeness task; same group as offensiveness'
                "American India or Alaska Native": "Native American",
            }
        )

        # Ordinal prefixes (zero-padded so they sort correctly) keep the
        # ordering in exports; "Other" is left unprefixed and sorts last.
        age_ranking = [
            "18-24",
            "25-29",
            "30-34",
            "35-39",
            "40-44",
            "45-49",
            "50-54",
            "54-59",  # sic, as in the source data
            "60-64",
            ">65",
        ]
        df["Age"] = df["Age"].replace(
            {a: f"{i + 1:02d}) {a}" for i, a in enumerate(age_ranking)}
        )

        education_ranking = [
            "Less than a high school diploma",
            "High school diploma or equivalent",
            "College degree",
            "Graduate degree",
        ]
        short_education = {
            "Less than a high school diploma": "No high school",
            "High school diploma or equivalent": "High school",
        }
        df["Education"] = df["Education"].replace(
            {
                e: f"{i + 1}) {short_education.get(e, e)}"
                for i, e in enumerate(education_ranking)
            }
        )

        agg = {col: list for col in ["Rating", *sdb_cols]}
        agg["text"] = "first"
        df = df.groupby("instance_id").agg(agg).reset_index()
        return df
