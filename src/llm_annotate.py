import argparse
import os
import itertools
from pathlib import Path
from typing import Optional

# Toggle to True if VRAM is under durress
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")

import numpy as np
import pandas as pd
import torch
import transformers
from tqdm.auto import tqdm
from transformers import pipeline

import tasks.preprocessing
from dices import DicesDataset
from kumar import KumarDataset
from sap import SapDataset

SEED = 42
N_PERSONAS_PER_COMMENT = 6
MAX_NEW_TOKENS = 3
MAX_CTX_TOKENS = 512

SAMPLES_PER_DATASET = {
    "dices-350": 300,
    "dices-990": 300,
    "kumar": 1000,
    "sap": 300,
}

DATASET_LOADERS = {
    "dices-350": lambda p: DicesDataset(dataset_path=p, variant="350"),
    "dices-990": lambda p: DicesDataset(dataset_path=p, variant="990"),
    "kumar": lambda p: KumarDataset(
        dataset_path=p, num_samples=SAMPLES_PER_DATASET["kumar"]
    ),
    "sap": lambda p: SapDataset(dataset_path=p),
}


def load_dataset(
    dataset_key: str, dataset_path: Path
) -> tasks.preprocessing.Dataset:
    return DATASET_LOADERS[dataset_key](dataset_path)


def load_generator(model_name: str):
    generator = pipeline(
        "text-generation",
        model=model_name,
        device_map="auto",
    )
    if generator.tokenizer.pad_token is None:
        generator.tokenizer.pad_token = generator.tokenizer.eos_token
    return generator


def get_subgroup_value_pools(
    ds: tasks.preprocessing.Dataset,
) -> dict[str, list]:
    """Distinct observed values per SDB column, used as the sampling pool
    for random persona characteristics."""
    pools = {}
    for col, counts in ds.get_subgroup_counts().items():
        pools[col] = counts.index.tolist()
    return pools


def truncate_text(tokenizer, text: str, max_tokens: int) -> str:
    """Truncates to the last max_tokens tokens, so every prompt fed to the
    model has a bounded, consistent sequence length."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return tokenizer.decode(ids[-max_tokens:])


def sample_texts(
    ds: tasks.preprocessing.Dataset,
    n: int,
    rng: np.random.Generator,
) -> list[tuple[str, str]]:
    key_col = ds.get_comment_key_column()
    keys = ds.get_dataset()[key_col].tolist()

    text_col = ds.get_text_column()
    texts = ds.get_dataset()[text_col].tolist()

    n = min(n, len(keys))
    idx = rng.choice(len(keys), size=n, replace=False)

    return [(keys[i], texts[i]) for i in idx]


def sample_personas(value_pools, n, rng):
    columns = list(value_pools.keys())
    seen = set()
    personas = []

    while len(personas) < n:
        candidate = tuple(
            value_pools[col][rng.integers(len(value_pools[col]))]
            for col in columns
        )
        if candidate not in seen:
            seen.add(candidate)
            personas.append(dict(zip(columns, candidate)))

    return personas


def format_persona(persona: dict[str, str]) -> str:
    return "; ".join(f"{k}: {v}" for k, v in persona.items())


def build_messages(
    template: str,
    persona: dict[str, str],
    text: str,
) -> list[dict]:
    system_content = template.format(persona=format_persona(persona))

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": text},
    ]


def generate_annotation(generator, messages: list[dict]) -> str:
    # no need to supply cuda device due to accelerate
    with torch.inference_mode():
        output = generator(
            messages,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
        )

    reply = output[0]["generated_text"]

    # Chat-formatted input makes the pipeline return the full conversation
    # (as a list of role/content dicts); pull out the assistant's reply.
    if isinstance(reply, list):
        reply = reply[-1]["content"]

    return reply.strip()


def annotate_comment(
    generator,
    template: str,
    text_id: str,
    text: str,
    value_pools: dict[str, list],
    model_name: str,
    prompt_name: str,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Sample N_PERSONAS_PER_COMMENT distinct personas for this comment,
    then generate one annotation for each persona.
    """
    rows = []

    personas = sample_personas(
        value_pools=value_pools,
        n=N_PERSONAS_PER_COMMENT,
        rng=rng,
    )

    for persona in personas:
        messages = build_messages(
            template=template,
            persona=persona,
            text=text,
        )

        annotation = generate_annotation(
            generator,
            messages,
        )

        rows.append(
            {
                "model": model_name,
                "instruction_prompt": prompt_name,
                "text_id": text_id,
                "text": text,
                **persona,
                "annotation": annotation,
            }
        )

    return rows


def main(
    dataset_key: str,
    dataset_path: Path,
    instruction_prompt_path: Path,
    model_name: str,
    output_path: Path,
    sample_fraction: Optional[float] = None,
):
    transformers.set_seed(SEED)
    rng = np.random.default_rng(SEED)

    ds = load_dataset(dataset_key, dataset_path)
    template = instruction_prompt_path.read_text()
    value_pools = get_subgroup_value_pools(ds)
    generator = load_generator(model_name)

    base_n = SAMPLES_PER_DATASET[dataset_key]
    if sample_fraction is not None:
        n_samples = max(1, int(round(base_n * sample_fraction)))
    else:
        n_samples = base_n

    packets = sample_texts(
        ds,
        n_samples,
        rng,
    )

    rows = []

    for packet in tqdm(
        packets,
        desc=f"Annotating {ds.get_name()}",
    ):
        text_id, text = packet

        text = truncate_text(
            generator.tokenizer,
            text,
            MAX_CTX_TOKENS,
        )

        rows.extend(
            annotate_comment(
                generator=generator,
                template=template,
                text_id=text_id,
                text=text,
                value_pools=value_pools,
                model_name=model_name,
                prompt_name=instruction_prompt_path.name,
                rng=rng,
            )
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Annotate dataset comments with LLM personas that have random "
            "sociodemographic characteristics."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASET_LOADERS.keys()),
        help="Which dataset to sample comments from.",
    )

    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to the raw dataset file for the chosen --dataset.",
    )

    parser.add_argument(
        "--instruction-prompt-path",
        required=True,
        help=(
            "Path to a text file containing the system-prompt template, "
            "with a {persona} placeholder. Used as the system message; "
            "the comment text is sent separately as the user message."
        ),
    )

    parser.add_argument(
        "--model-name",
        required=True,
        help="Hugging Face transformers model name or path.",
    )

    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to write the resulting annotations CSV to.",
    )

    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help=(
            "If set, sample this fraction of the dataset's normal "
            "SAMPLES_PER_DATASET count (not the raw dataset size), e.g. "
            "0.1 on dices-350 (300 samples normally) yields 30 samples. "
            "Used for sensitivity ablations so repeat / paraphrase runs "
            "stay cheap. Uses the same SEED as a normal run in all cases, "
            "so the sampled comments are identical across every repeat "
            "run and every paraphrase variant for a given dataset, "
            "keeping their outputs directly comparable."
        ),
    )

    args = parser.parse_args()

    main(
        dataset_key=args.dataset,
        dataset_path=Path(args.dataset_path),
        instruction_prompt_path=Path(args.instruction_prompt_path),
        model_name=args.model_name,
        output_path=Path(args.output_path),
        sample_fraction=args.sample_fraction,
    )