import dataclasses
import random
import json
import re
from pathlib import Path

import transformers
from tqdm.auto import tqdm

from . import real_life_kumar


@dataclasses.dataclass
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


SEX_OPTIONS = ["male", "female", "non-binary"]
SEX_WEIGHTS = [0.48, 0.50, 0.02]

SEXUAL_ORIENTATION_OPTIONS = [
    "heterosexual",
    "homosexual",
    "bisexual",
    "asexual",
]
SEXUAL_ORIENTATION_WEIGHTS = [0.85, 0.07, 0.06, 0.02]

EDUCATION_LEVEL_OPTIONS = [
    "no formal education",
    "primary education",
    "secondary education",
    "vocational training",
    "bachelor's degree",
    "master's degree",
    "doctoral degree",
]
EDUCATION_LEVEL_WEIGHTS = [0.02, 0.10, 0.40, 0.15, 0.20, 0.10, 0.03]

POLITICAL_AFFILIATION_OPTIONS = [
    "left-wing",
    "centrist",
    "right-wing",
    "apolitical",
]
POLITICAL_AFFILIATION_WEIGHTS = [0.25, 0.35, 0.25, 0.15]


# ---------------------------
#  Persona Generation
# ---------------------------


def _weighted_choice(options, weights):
    """Helper function for weighted random choice."""
    return random.choices(options, weights=weights, k=1)[0]


def _generate_random_persona() -> Persona:
    """Generate a Persona instance with weighted sociodemographic traits."""
    age = random.randint(18, 80)
    sex = _weighted_choice(SEX_OPTIONS, SEX_WEIGHTS)
    sexual_orientation = _weighted_choice(
        SEXUAL_ORIENTATION_OPTIONS, SEXUAL_ORIENTATION_WEIGHTS
    )
    education_level = _weighted_choice(
        EDUCATION_LEVEL_OPTIONS, EDUCATION_LEVEL_WEIGHTS
    )
    political_affiliation = _weighted_choice(
        POLITICAL_AFFILIATION_OPTIONS, POLITICAL_AFFILIATION_WEIGHTS
    )

    return Persona(
        age=age,
        sex=sex,
        sexual_orientation=sexual_orientation,
        education_level=education_level,
        political_affiliation=political_affiliation,
    )


def generate_persona_list(n: int = 10):
    """Generate a list of n random Persona objects."""
    return [_generate_random_persona() for _ in range(n)]


def save_personas_to_json(personas, output_path: Path):
    """Save a list of personas to a JSON file."""
    data = [p.to_dict() for p in personas]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(personas)} personas to {output_path}")


def get_texts(kumar_path: Path, num_comments: int) -> list[str]:
    ds = real_life_kumar.KumarDataset(dataset_path=kumar_path)
    texts = ds.df["tweet"]
    return texts.tolist()


def annotate(
    pipeline, personas: list[Persona], instructions: str, texts: list[str]
) -> dict[Persona, dict[str, float]]:
    results = {}
    for persona in tqdm(personas, desc="Annotators"):
        prompt = _annotation_prompt(persona=persona, instructions=instructions)
        annotations = {}
        for text in tqdm(texts, desc="Comments"):
            annotation = _annotate_texts(
                pipeline=pipeline, prompt=prompt, text=text
            )
            annotations[text] = annotation

        results[persona] = annotations

    return results


def _annotation_prompt(persona: Persona, instructions: str) -> str:
    persona_summary = (
        f"You are a {persona.age}-year-old {persona.sex} "
        f"with {persona.education_level}, who identifies as "
        f"{persona.sexual_orientation} "
        f"and leans {persona.political_affiliation} politically."
    )

    # Final prompt combines persona and instructions, separated clearly
    prompt = (
        f"{persona_summary}\n\n"
        f"Follow the instructions carefully:\n"
        f"{instructions}"
    )

    return prompt


def _annotate_texts(pipeline, prompt: str, text: str) -> float:

    # Construct chat input if model supports chat format
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
    ]

    try:
        # Generate response
        response = pipeline(messages)[0]["generated_text"]
        response = _parse_response(response)
        return response

    except Exception as e:
        return -1
        print(f"[ERROR] Failed to annotate text: {text[:50]!r} | {e}")


def _parse_response(response: str) -> float:
    # Extract a numeric rating (float between 1 and 5)
    match = re.search(r"([1-5](?:\.\d+)?)", response)
    if match:
        score = float(match.group(1))
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
        model="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        max_new_tokens=10,
    )

    texts = get_texts("data/kumar.json")

    with open("data/annotation/prompt.txt", "r") as file:
        instructions = file.read()

    results = annotate(
        pipeline=pipe,
        personas=personas,
        instructions=instructions,
        texts=texts,
    )
    # Optionally save to file
    output_file = Path("data/annotation.json")
    json.dump(results, output_file)  # if this breaks i WILL cry.
