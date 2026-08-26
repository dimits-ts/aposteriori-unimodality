import argparse
from pathlib import Path

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
N_PERSONAS = 5
MAX_NEW_TOKENS = 128

# Number of comments to sample per dataset (dataset-specific).
SAMPLES_PER_DATASET = {
    "dices-350": 15,
    "dices-990": 15,
    "kumar": 30,
    "sap": 30,
}

DATASET_LOADERS = {
    "dices-350": lambda p: DicesDataset(dataset_path=p, variant="350"),
    "dices-990": lambda p: DicesDataset(dataset_path=p, variant="990"),
    "kumar": lambda p: KumarDataset(dataset_path=p),
    "sap": lambda p: SapDataset(dataset_path=p),
}


def load_dataset(dataset_key: str, dataset_path: Path) -> tasks.preprocessing.Dataset:
    return DATASET_LOADERS[dataset_key](dataset_path)


def load_generator(model_name: str):
    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline("text-generation", model=model_name, device=device)
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


def sample_texts(
    ds: tasks.preprocessing.Dataset, n: int, rng: np.random.Generator
) -> list:
    col = ds.get_comment_key_column()
    values = ds.get_dataset()[col].tolist()
    n = min(n, len(values))
    idx = rng.choice(len(values), size=n, replace=False)
    return [values[i] for i in idx]


def sample_persona(
    value_pools: dict[str, list], rng: np.random.Generator
) -> dict[str, str]:
    persona = {}
    for col, values in value_pools.items():
        persona[col] = values[rng.integers(len(values))]
    return persona


def format_persona(persona: dict[str, str]) -> str:
    return "; ".join(f"{k}: {v}" for k, v in persona.items())


def build_prompt(template: str, persona: dict[str, str], text: str) -> str:
    return template.format(persona=format_persona(persona), text=text)


def generate_annotation(generator, prompt: str) -> str:
    output = generator(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        return_full_text=False,
    )
    return output[0]["generated_text"].strip()


def annotate_comment(
    generator,
    template: str,
    text: str,
    value_pools: dict[str, list],
    model_name: str,
    prompt_name: str,
    rng: np.random.Generator,
) -> list[dict]:
    rows = []
    for _ in range(N_PERSONAS):
        persona = sample_persona(value_pools, rng)
        prompt = build_prompt(template, persona, text)
        annotation = generate_annotation(generator, prompt)
        rows.append(
            {
                "model": model_name,
                "instruction_prompt": prompt_name,
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
):
    transformers.set_seed(SEED)
    rng = np.random.default_rng(SEED)

    ds = load_dataset(dataset_key, dataset_path)
    template = instruction_prompt_path.read_text()
    value_pools = get_subgroup_value_pools(ds)
    generator = load_generator(model_name)

    texts = sample_texts(ds, SAMPLES_PER_DATASET[dataset_key], rng)

    rows = []
    for text in tqdm(texts, desc=f"Annotating {ds.get_name()}"):
        rows.extend(
            annotate_comment(
                generator=generator,
                template=template,
                text=text,
                value_pools=value_pools,
                model_name=model_name,
                prompt_name=instruction_prompt_path.name,
                rng=rng,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


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
            "Path to a text file containing the instruction prompt "
            "template, with {persona} and {text} placeholders."
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
    args = parser.parse_args()
    main(
        dataset_key=args.dataset,
        dataset_path=Path(args.dataset_path),
        instruction_prompt_path=Path(args.instruction_prompt_path),
        model_name=args.model_name,
        output_path=Path(args.output_path),
    )