import dataclasses
import random
import json
import re
from pathlib import Path

import transformers
import pandas as pd
from tqdm.auto import tqdm

from . import kumar

SEX_OPTIONS = ["male", "female", "non-binary"]

SEXUAL_ORIENTATION_OPTIONS = [
    "heterosexual",
    "homosexual",
    "bisexual",
    "asexual",
]

EDUCATION_LEVEL_OPTIONS = [
    "no formal education",
    "primary education",
    "secondary education",
    "vocational training",
    "bachelor's degree",
    "master's degree",
    "doctoral degree",
]

POLITICAL_AFFILIATION_OPTIONS = [
    "left-wing",
    "centrist",
    "right-wing",
    "apolitical",
]


@dataclasses.dataclass(frozen=True)
class Persona:
    """
    A dataclass holding information about the synthetic persona of a LLM actor.
    Includes sociodemographic traits and personality traits.
    """

    age: int = -1
    sex: str = ""
    sexual_orientation: str = ""
    education_level: str = ""
    political_affiliation: str = ""

    def to_dict(self):
        return dataclasses.asdict(self)

    def to_json_file(self, output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def __str__(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ---------------------------
#  Persona Generation
# ---------------------------


def generate_persona_list(n: int = 10):
    """Generate a list of n random Persona objects."""
    return [_generate_random_persona() for _ in range(n)]


def annotate(
    pipeline, personas: list[Persona], instructions: str, texts: list[str]
) -> dict[Persona, dict[str, float]]:
    results = {}
    for persona in tqdm(personas, desc="Annotators"):
        prompt = _annotation_prompt(persona=persona, instructions=instructions)
        annotations = {}
        for text in tqdm(texts, desc="Comments", leave=False):
            annotation = _annotate_texts(
                pipeline=pipeline, prompt=prompt, text=text
            )
            annotations[text] = annotation

        results[persona] = annotations

    return results


def annotations_to_df(results: dict) -> pd.DataFrame:
    """
    Convert the annotation results dictionary into a DataFrame.

    Each row corresponds to one (text, persona) pair.
    Persona attributes (age, sex, etc.) become separate columns.
    """
    rows = []

    for persona, annotations in results.items():
        persona_dict = persona.to_dict()
        for text, score in annotations.items():
            row = {"text": text, "annotation": score, **persona_dict}
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def get_texts(kumar_path: Path, num_comments: int) -> list[str]:
    ds = kumar.KumarDataset(
        dataset_path=kumar_path, num_samples=num_comments
    )
    texts = ds.df["comment"]
    return texts.tolist()


def _choice(options):
    """Helper function for weighted random choice."""
    return random.choices(options, k=1)[0]


def _generate_random_persona() -> Persona:
    """Generate a Persona instance with weighted sociodemographic traits."""
    age = random.randint(18, 80)
    sex = _choice(SEX_OPTIONS)
    sexual_orientation = _choice(SEXUAL_ORIENTATION_OPTIONS)
    education_level = _choice(EDUCATION_LEVEL_OPTIONS)
    political_affiliation = _choice(POLITICAL_AFFILIATION_OPTIONS)

    return Persona(
        age=age,
        sex=sex,
        sexual_orientation=sexual_orientation,
        education_level=education_level,
        political_affiliation=political_affiliation,
    )


def _annotation_prompt(persona: Persona, instructions: str) -> str:
    prompt = instructions.format(**persona.to_dict())
    return prompt


def _annotate_texts(pipeline, prompt: str, text: str) -> float:

    # Construct chat input if model supports chat format
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
    ]

    try:
        # Generate response
        response = pipeline(messages)
        response = _get_answer(message=response)
        response = _parse_response(response)
        return response

    except Exception as e:
        print(f"[ERROR] Failed to annotate text: {text[:50]!r} | {e}")
        return -1


def _get_answer(message) -> str:
    return message[0]["generated_text"][-1]["content"]


def _parse_response(response: str) -> float:
    # Extract a numeric rating (float between 1 and 5)
    match = re.search(r"([1-5](?:\.\d+)?)", response)
    if match:
        score = int(match.group(1))
        if 1.0 <= score <= 5.0:
            return score
    print(f"Invalid model response: {response}")
    return -1


# ---------------------------
#  Example Usage
# ---------------------------

if __name__ == "__main__":
    random.seed(42)
    personas = generate_persona_list(100)

    pipe = transformers.pipeline(
        "text-generation",
        model="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
        max_new_tokens=4,
        device_map="auto"
    )

    texts = get_texts(Path("data/kumar.json"), num_comments=80)

    with open("data/annotation/prompt.txt", "r") as file:
        instructions = file.read()

    results = annotate(
        pipeline=pipe,
        personas=personas,
        instructions=instructions,
        texts=texts,
    )

    results_df = annotations_to_df(results)
    # if this breaks i WILL cry.
    results_df.to_csv("data/annotation_70b.csv", index=False)
