from pathlib import Path

import numpy as np
import pandas as pd
import syndisco.model
import syndisco.actors
import syndisco.experiments
import syndisco.logging_util
import syndisco.postprocessing


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
INPUT_DIR = Path(
    "/media/SSD_4TB_2/dtsirmpas/projects/aposteriori-unimodality/data/synthetic/input/"
)
OUTPUT_DIR = Path(
    "/media/SSD_4TB_2/dtsirmpas/projects/aposteriori-unimodality/data/synthetic/output/"
)
DATASET_OUTPUT_PATH = Path(
    "/media/SSD_4TB_2/dtsirmpas/projects/aposteriori-unimodality/data/100_annotators.csv"
)
LOGS_DIR = Path(
    "/media/SSD_4TB_2/dtsirmpas/projects/aposteriori-unimodality/data/synthetic/logs/"
)


def main():
    rng = np.random.default_rng(42)
    ages = np.floor(rng.uniform(low=15, high=90, size=NUM_ANNOTATORS))
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
        ["straight", "homosexual", "bisexual", "other"],
        size=NUM_ANNOTATORS,
        p=[0.7, 0.1, 0.1, 0.1],
    )
    group = rng.choice(
        ["black", "white", "asian", "other"], size=NUM_ANNOTATORS
    )
    characteristics = rng.choice(
        [
            "right-wing conservative",
            "left-wing liberal",
            "apolitical",
        ],
        size=NUM_ANNOTATORS,
    )

    model = syndisco.model.TransformersModel(
        model_path="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
        name="llama3.3",
        max_out_tokens=20,
    )

    syndisco.logging_util.logging_setup(
        print_to_terminal=True,
        write_to_file=True,
        logs_dir=LOGS_DIR,
        level="info",
        use_colors=True,
        log_warnings=True,
    )

    annotators = []
    for i in range(NUM_ANNOTATORS):
        persona = syndisco.actors.Persona(
            username="annotator",
            age=ages[i],
            sex=sexes[i],
            sexual_orientation=sex_orient[i],
            demographic_group=group[i],
            current_employment=occupation[i],
            education_level=education[i],
            special_instructions="",
            personality_characteristics=characteristics[i],
        )
        actor = syndisco.actors.Actor(
            model=model,
            persona=persona,
            context=CONTEXT,
            instructions=INSTRUCTIONS,
            actor_type=syndisco.actors.ActorType.ANNOTATOR,
        )
        annotators.append(actor)

    experiment = syndisco.experiments.AnnotationExperiment(
        annotators=annotators
    )
    experiment.begin(
        discussions_dir=INPUT_DIR, output_dir=OUTPUT_DIR, verbose=False
    )
    annotations_df = syndisco.postprocessing.import_annotations(
        annot_dir=OUTPUT_DIR
    )
    annotations_df.to_csv(DATASET_OUTPUT_PATH)


if __name__ == "__main__":
    main()
