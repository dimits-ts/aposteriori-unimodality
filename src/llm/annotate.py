import argparse
import logging
import os
import sys
import math
import copy
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import torch
import transformers
from tqdm.auto import tqdm
from transformers import AutoTokenizer, DynamicCache, pipeline

from ..lib.preprocessing import (
    SapDataset,
    KumarDataset,
    Dataset,
    DicesDataset,
    PopquornDataset,
)

logger = logging.getLogger("annotate")

LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
SEED = 42
MAX_NEW_TOKENS = 3
MAX_CTX_TOKENS = 512


DATASET_LOADERS = {
    "kumar": lambda p: KumarDataset(dataset_path=p, num_samples=1000),
    "sap": lambda p: SapDataset(dataset_path=p),
    "dices-350": lambda p: DicesDataset(dataset_path=p, variant="350"),
    "dices-990": lambda p: DicesDataset(dataset_path=p, variant="990"),
    "popquorn": lambda p: PopquornDataset(dataset_path=p),
}
PERSONA_SUFFIX = (
    "\n\nAnnotate as a person with these characteristics: {persona}"
)


def main(
    dataset_key: str,
    dataset_path: Path,
    instruction_prompt_path: Path,
    model_name: str,
    output_path: Path,
    batch_size: int,
    num_annotators: int,
    sample_fraction: float | None = None,
    no_persona: bool = False,
):
    # Toggle to True if VRAM is under durress
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False"
    )

    transformers.set_seed(SEED)
    rng = np.random.default_rng(SEED)

    logger.info(
        "Starting run: dataset=%s, prompt=%s, model=%s, num_annotators=%d, "
        "batch_size=%d, sample_fraction=%s, no_persona=%s",
        dataset_key,
        instruction_prompt_path,
        model_name,
        num_annotators,
        batch_size,
        sample_fraction,
        no_persona,
    )

    ds = load_dataset(dataset_key, dataset_path)
    template = instruction_prompt_path.read_text()
    # in case I forget
    if "{persona}" in template:
        raise ValueError(
            "Template still contains {persona}. For prefix caching the persona "
            "must come last, so remove it from the template; it is appended to "
            "the user message via PERSONA_SUFFIX."
        )

    # In the no-persona condition every annotator receives the identical
    # prompt; variation comes only from sampling (do_sample=True).
    value_pools = None if no_persona else get_subgroup_value_pools(ds)
    if value_pools is not None:
        _warn_if_persona_capped(value_pools, num_annotators)

    logger.info("Loading model %s", model_name)
    generator = load_generator(model_name)

    base_n = len(ds.get_dataset())
    if sample_fraction is not None:
        n_samples = max(1, int(round(base_n * sample_fraction)))
    else:
        n_samples = base_n

    packets = sample_texts(
        ds,
        n_samples,
        rng,
    )
    logger.info(
        "Annotating %d of %d comments from %s",
        len(packets),
        base_n,
        ds.get_name(),
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
                batch_size=batch_size,
                num_annotators=num_annotators,
                no_persona=no_persona,
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
    logger.info("Wrote %d annotations to %s", len(rows), output_path)


def setup_logging(log_file: Path | None) -> None:
    """Log to stderr and, if given, append to log_file. Warnings from
    transformers / huggingface_hub and Python warnings go through the same
    handlers, so they reach the log file too."""
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(log_file, mode="a", encoding="utf-8")
        )

    # WARNING at the root keeps chatty INFO loggers (e.g. HTTP clients) out.
    logging.basicConfig(
        level=logging.WARNING, format=LOG_FORMAT, handlers=handlers, force=True
    )
    logger.setLevel(logging.INFO)
    logging.captureWarnings(True)
    _route_library_logs()


def _route_library_logs() -> None:
    """The HF libraries attach their own stderr handlers; drop them and let
    records propagate to the root handlers instead (avoids duplicates)."""
    transformers.logging.disable_default_handler()
    transformers.logging.enable_propagation()

    hub_logger = logging.getLogger("huggingface_hub")
    hub_logger.handlers.clear()
    hub_logger.propagate = True


def _warn_if_persona_capped(
    value_pools: dict[str, list], num_annotators: int
) -> None:
    total = _count_possible_personas(value_pools)
    if total < num_annotators:
        logger.warning(
            "Only %d distinct personas are possible; each comment gets %d "
            "annotators instead of the requested %d.",
            total,
            total,
            num_annotators,
        )


def load_dataset(dataset_key: str, dataset_path: Path) -> Dataset:
    return DATASET_LOADERS[dataset_key](dataset_path)


def load_generator(model_name: str):
    # pipeline() has no tokenizer_kwargs argument, so loader flags must be
    # passed by loading the tokenizer ourselves. fix_mistral_regex is a no-op
    # for tokenizers that aren't affected.
    kwargs = (
        {"fix_mistral_regex": True} if "mistral" in model_name.lower() else {}
    )
    tok = AutoTokenizer.from_pretrained(model_name, **kwargs)
    generator = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tok,
        device_map="auto",
    )
    # The model's generation_config sets max_length, which clashes with the
    # max_new_tokens passed to generate() and warns on every call.
    generator.model.generation_config.max_length = None
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return generator


def get_subgroup_value_pools(
    ds: Dataset,
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
    ds: Dataset,
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


def _count_possible_personas(value_pools: dict[str, list]) -> int:
    return math.prod(len(v) for v in value_pools.values())


def _sample_personas_exhaustive(
    value_pools: dict[str, list], columns, n: int, rng: np.random.Generator
):
    all_combos = list(product(*(value_pools[c] for c in columns)))
    idx = rng.choice(len(all_combos), size=n, replace=False)
    return [dict(zip(columns, all_combos[i])) for i in idx]


def _sample_personas_rejection(
    value_pools: dict[str, list], columns, n: int, rng: np.random.Generator
):
    seen, personas = set(), []
    while len(personas) < n:
        candidate = tuple(
            value_pools[col][rng.integers(len(value_pools[col]))]
            for col in columns
        )
        if candidate not in seen:
            seen.add(candidate)
            personas.append(dict(zip(columns, candidate)))
    return personas


def sample_personas(value_pools, n, rng):
    """Sample n distinct personas. If n exceeds the number of possible
    combinations, returns all of them (in random order)."""
    columns = list(value_pools.keys())
    total = _count_possible_personas(value_pools)
    n = min(n, total)

    # if more than 1000 combinations, use the rejection sampling approach
    if total <= 1000:
        return _sample_personas_exhaustive(
            value_pools=value_pools, columns=columns, n=n, rng=rng
        )
    else:
        return _sample_personas_rejection(
            value_pools=value_pools, columns=columns, n=n, rng=rng
        )


def format_persona(persona: dict[str, str]) -> str:
    return "; ".join(f"{k}: {v}" for k, v in persona.items())


def build_messages(
    template: str,
    persona: dict[str, str] | None,
    text: str,
) -> list[dict]:
    """If persona is None (no-persona condition), the user message is just
    the comment text, with no PERSONA_SUFFIX appended."""
    content = text
    if persona is not None:
        content += PERSONA_SUFFIX.format(persona=format_persona(persona))
    return [
        {"role": "system", "content": template},
        {"role": "user", "content": content},
    ]


def _tokenize_prompts(
    tok, batch_of_messages: list[list[dict]]
) -> list[list[int]]:
    """Render each conversation with the chat template and tokenize it.
    Working at token level guarantees the cached tokens match exactly."""
    prompts = [
        tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        for messages in batch_of_messages
    ]
    return [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]


def _common_prefix_length(all_ids: list[list[int]]) -> int:
    """Length of the longest shared token prefix, leaving at least one
    token per prompt so every suffix is non-empty."""
    min_len = min(len(ids) for ids in all_ids)
    n = 0
    while n < min_len - 1 and all(ids[n] == all_ids[0][n] for ids in all_ids):
        n += 1
    assert (
        n > 0
    ), "No shared prefix found; check the chat template / prompt layout."
    return n


def _prefill_prefix(model, prefix_ids: torch.Tensor) -> DynamicCache:
    """Run the shared prefix through the model once and return its KV cache."""
    cache = DynamicCache()
    with torch.inference_mode():
        model(
            input_ids=prefix_ids.to(model.device),
            past_key_values=cache,
            use_cache=True,
        )
    return cache


def _left_pad_suffixes(
    suffixes: list[list[int]], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad suffix token lists into an (ids, mask) tensor pair."""
    b = len(suffixes)
    max_s = max(len(s) for s in suffixes)
    ids = torch.full((b, max_s), pad_id, dtype=torch.long)
    mask = torch.zeros((b, max_s), dtype=torch.long)
    for i, s in enumerate(suffixes):
        ids[i, max_s - len(s) :] = torch.tensor(s)
        mask[i, max_s - len(s) :] = 1
    return ids, mask


def _build_batch_inputs(
    prefix_ids: torch.Tensor,
    suffixes: list[list[int]],
    pad_id: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate prefix + padded suffixes into full input_ids and attention
    mask. The padding sits between prefix and suffix, which is fine because
    the attention mask hides it and generate() derives position_ids from it."""
    b = len(suffixes)
    prefix_len = prefix_ids.shape[1]
    suffix_ids, suffix_mask = _left_pad_suffixes(suffixes, pad_id)

    input_ids = torch.cat([prefix_ids.expand(b, -1), suffix_ids], dim=1)
    attention_mask = torch.cat(
        [torch.ones(b, prefix_len, dtype=torch.long), suffix_mask], dim=1
    )
    return input_ids.to(device), attention_mask.to(device)


def _expand_cache(prefix_cache: DynamicCache, batch_size: int) -> DynamicCache:
    """generate() mutates the cache, so work on a copy expanded to batch size."""
    cache = copy.deepcopy(prefix_cache)
    cache.batch_repeat_interleave(batch_size)
    return cache


def _generate_batch(
    generator,
    prefix_ids: torch.Tensor,
    prefix_cache: DynamicCache,
    suffixes: list[list[int]],
) -> list[str]:
    """Generate one annotation per suffix on top of the cached prefix."""
    tok, model = generator.tokenizer, generator.model
    pad_id = tok.pad_token_id

    input_ids, attention_mask = _build_batch_inputs(
        prefix_ids, suffixes, pad_id, model.device
    )
    cache = _expand_cache(prefix_cache, len(suffixes))

    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            pad_token_id=pad_id,
        )

    new_tokens = out[:, input_ids.shape[1] :]
    return [
        t.strip()
        for t in tok.batch_decode(new_tokens, skip_special_tokens=True)
    ]


def generate_annotations(
    generator,
    batch_of_messages: list[list[dict]],
    batch_size: int,
) -> list[str]:
    """Prefill the shared prefix (instructions + comment) once, then run the
    per-persona suffixes in batches on top of a copy of that cache."""
    all_ids = _tokenize_prompts(generator.tokenizer, batch_of_messages)
    prefix_len = _common_prefix_length(all_ids)

    prefix_ids = torch.tensor([all_ids[0][:prefix_len]])
    prefix_cache = _prefill_prefix(generator.model, prefix_ids)

    chunks = [
        all_ids[i : i + batch_size] for i in range(0, len(all_ids), batch_size)
    ]
    results = []
    for chunk in chunks:
        suffixes = [ids[prefix_len:] for ids in chunk]
        results.extend(
            _generate_batch(generator, prefix_ids, prefix_cache, suffixes)
        )
    return results


def annotate_comment(
    generator,
    template: str,
    text_id: str,
    text: str,
    value_pools: dict[str, list],
    model_name: str,
    prompt_name: str,
    rng: np.random.Generator,
    batch_size: int,
    num_annotators: int,
    no_persona: bool = False,
) -> list[dict]:
    """
    Sample up to n distinct personas for this comment,
    then generate one annotation per persona in batches.

    If no_persona is set, num_annotators identical persona-less prompts are
    used instead, so annotations differ only through sampling.
    """
    if no_persona:
        personas = [None] * num_annotators
    else:
        personas = sample_personas(
            value_pools=value_pools,
            n=num_annotators,
            rng=rng,
        )

    batch = [
        build_messages(template=template, persona=p, text=text)
        for p in personas
    ]
    annotations = generate_annotations(generator, batch, batch_size)

    rows = []
    for i, (persona, annotation) in enumerate(zip(personas, annotations)):
        row = {
            "model": model_name,
            "instruction_prompt": prompt_name,
            "text_id": text_id,
            "text": text,
        }
        if persona is None:
            # No SDB columns exist; an index keeps annotators distinguishable.
            row["annotator_id"] = i
        else:
            row.update(persona)
        row["annotation"] = annotation
        rows.append(row)
    return rows


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
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Number of persona prompts per forward pass.",
    )
    parser.add_argument(
        "--num-annotators",
        type=int,
        required=True,
        help="Number of persona prompts per forward pass.",
    )
    parser.add_argument(
        "--no-persona",
        action="store_true",
        help=(
            "Run the no-persona baseline: no sociodemographic persona is "
            "appended to the prompt, and each comment is annotated "
            "--num-annotators times with the identical prompt (variation "
            "comes only from sampling). Output has an annotator_id column "
            "instead of SDB columns. Comments sampled are the same as in "
            "persona runs with the same --sample-fraction."
        ),
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help=(
            "If set, append log messages (including library warnings and "
            "errors) to this file, in addition to stderr. The progress bar "
            "is shown on the terminal only."
        ),
    )

    args = parser.parse_args()
    setup_logging(Path(args.log_file) if args.log_file else None)

    try:
        main(
            dataset_key=args.dataset,
            dataset_path=Path(args.dataset_path),
            instruction_prompt_path=Path(args.instruction_prompt_path),
            model_name=args.model_name,
            output_path=Path(args.output_path),
            sample_fraction=args.sample_fraction,
            batch_size=args.batch_size,
            num_annotators=args.num_annotators,
            no_persona=args.no_persona,
        )
    except Exception:
        # Log the traceback so failures show up in the log file.
        logger.exception("Annotation run failed")
        sys.exit(1)
