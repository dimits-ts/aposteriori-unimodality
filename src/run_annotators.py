import dataclasses
import random
import json
from pathlib import Path


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


# ---------------------------
#  Sample Value Lists + Weights
# ---------------------------

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


def weighted_choice(options, weights):
    """Helper function for weighted random choice."""
    return random.choices(options, weights=weights, k=1)[0]


def generate_random_persona() -> Persona:
    """Generate a Persona instance with weighted sociodemographic traits."""
    age = random.randint(18, 80)
    sex = weighted_choice(SEX_OPTIONS, SEX_WEIGHTS)
    sexual_orientation = weighted_choice(
        SEXUAL_ORIENTATION_OPTIONS, SEXUAL_ORIENTATION_WEIGHTS
    )
    education_level = weighted_choice(
        EDUCATION_LEVEL_OPTIONS, EDUCATION_LEVEL_WEIGHTS
    )
    political_affiliation = weighted_choice(
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
    return [generate_random_persona() for _ in range(n)]


def save_personas_to_json(personas, output_path: Path):
    """Save a list of personas to a JSON file."""
    data = [p.to_dict() for p in personas]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(personas)} personas to {output_path}")


# ---------------------------
#  Example Usage
# ---------------------------

if __name__ == "__main__":
    random.seed(42)
    personas = generate_persona_list(100)

    # Print them
    for p in personas:
        print(p, "\n")

    # Optionally save to file
    output_file = Path("data/random_personas.json")
    save_personas_to_json(personas, output_file)
