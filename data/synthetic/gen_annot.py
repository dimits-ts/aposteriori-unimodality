from pathlib import Path

import numpy as np
import syndisco


NUM_ANNOTATORS = 100
CONTEXT = "You are a human annotator."
INSTRUCTIONS = """
You will be given a series of comments made on an online discussion.
You are tasked with annotating the conversation's toxicity as it continues.
Annotate according to your background, experiences and political beliefs.

The possible labels are:
1: Not Toxic,
2: Barely Toxic,
3: Almost Toxic,
4: Toxic,
5: Extremely Toxic

Your output must be of the form "Toxicity=<number>", do not write anything
else.

Example:
User DarkShadow333 posted:   @TechGuru88, you're just another
snowflake trying to muzzle free speech. I don't need to present my
arguments with kid gloves. If you can't handle the heat, stay out of
the kitchen. And @moderator, who made you the thought police? This is
a free country, and I'll say what I want. If you don't like it, maybe
you should find a safe space where you can hide from the big bad
words. 😈💥

Output: Toxicity=4

Annotate the following conversation without engaging with it:
"""
INPUT_DIR = Path("./input")
OUTPUT_DIR = Path("./output")


def main():
    rng = np.random.default_rng(42)
    ages = rng.uniform(low=15, high=90, size=NUM_ANNOTATORS)
    education = rng.choice(
        ["high-school", "university", "none"], size=NUM_ANNOTATORS
    )
    occupation = rng.choice(
        ["unemployed", "blue-collar", "white-collar"], size=NUM_ANNOTATORS
    )
    sexes = rng.choice(
        ["male", "female", "non-binary"],
        size=NUM_ANNOTATORS,
        p=[0.45, 0.45, 0.1],
    )
    sex_orient = rng.choice(
        ["straight", "homosexual", "bisexual", "other"], p=[0.7, 0.1, 0.1, 0.1]
    )
    group = ["black", "white", "asian", "other"]
    characteristics = [
        "right-wing conservative",
        "left-wing liberal",
        "apolitical",
    ]

    model = syndisco.backend.model.TransformersModel(
        model_path="unsloth/Llama-3.3-70B-Instruct-bnb-4bit", name="llama3.3"
    )

    annotators = []
    for i in range(NUM_ANNOTATORS):
        persona = syndisco.backend.persona.LLMPersona(
            username="",
            age=ages[i],
            sex=sexes[i],
            sexual_orientation=sex_orient[i],
            demographic_group=group[i],
            current_employment=occupation[i],
            education_level=education[i],
            special_instructions="",
            personality_characteristics=characteristics[i],
        )
        actor = syndisco.backend.actors.LLMActor(
            model=model,
            name="annotator",
            attributes=persona.to_attribute_list(),
            context=CONTEXT,
            instructions=INSTRUCTIONS,
            actor_type=syndisco.backend.actors.ActorType.ANNOTATOR,
        )
        annotators.append(actor)

    experiment = syndisco.experiments.AnnotationExperiment(
        annotators=annotators
    )
    experiment.begin(discussions_dir=INPUT_DIR, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
