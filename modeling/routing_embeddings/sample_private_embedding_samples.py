"""
Sample n random examples from one or more HuggingFace dataset repos and mix them together.
Optionally format each sampled row into a "doc" column using a per-dataset template.
You can also format from a prebuilt chat message column via a tokenizer chat template.

Usage (single dataset):
    python sample_hf_dataset.py \
        --repos stanfordnlp/imdb \
        --n 100

Usage (multiple datasets, mixed):
    python sample_hf_dataset.py \
        --repos stanfordnlp/imdb rajpurkar/squad \
        --n 50 50 \
        --splits test validation \
        --doc-templates "{text} [label={label}]" "Q: {question} A: {answers[text][0]}"

Usage (via JSON config file):
    python sample_hf_dataset.py --config-file my_config.json

--- JSON config schema ---
{
  "seed": 42,
  "output": "out.jsonl",
  "streaming": false,
  "datasets": [
    {
      "repo": "stanfordnlp/imdb",
      "n": 50,
      "split": "test",
      "config": null,
            "doc_template": "{text} [label={label}]",
            "chat_messages_column": null,
            "chat_template_tokenizer": null
    },
    {
      "repo": "rajpurkar/squad",
      "n": 50,
      "split": "validation",
      "config": null,
            "doc_template": "Q: {question}\nContext: {context}\nA: {answers[text][0]}",
            "chat_messages_column": null,
            "chat_template_tokenizer": null
    }
  ]
}
"""

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from datasets import load_dataset
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

def _resolve_path(obj, path: str):
    """
    Resolve a dotted / indexed path like "answers[text][0]" against a dict.
    Supports:  key, key.subkey, key[subkey], key[index]
    """
    tokens = re.split(r'\.|\[|\]', path)
    tokens = [t for t in tokens if t != '']
    for token in tokens:
        if isinstance(obj, dict):
            obj = obj[token]
        elif isinstance(obj, (list, tuple)):
            obj = obj[int(token)]
        else:
            raise KeyError(f"Cannot index into {type(obj)} with key '{token}'")
    return obj


def render_template(template: str, row: dict) -> str:
    """
    Render a template string against a dataset row dict.

    Supports:
      {column}              — simple column lookup
      {column.subkey}       — nested dict access
      {column[subkey][0]}   — nested dict + list index
    """
    def replacer(match):
        path = match.group(1)
        try:
            value = _resolve_path(row, path)
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(
                f"Template field '{path}' not found in row. "
                f"Available keys: {list(row.keys())}"
            ) from e
        return str(value)

    return re.sub(r'\{([^}]+)\}', replacer, template)


# ---------------------------------------------------------------------------
# Per-dataset config
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    repo: str
    n: int
    split: str = "train"
    config: str | None = None
    doc_template: str | None = None
    chat_messages_column: str | None = None
    chat_template_tokenizer: str | None = None
    streaming: bool = False

    @staticmethod
    def from_dict(d: dict) -> "DatasetConfig":
        return DatasetConfig(
            repo=d["repo"],
            n=d["n"],
            split=d.get("split", "train"),
            config=d.get("config", None),
            doc_template=d.get("doc_template", None),
            chat_messages_column=d.get("chat_messages_column", None),
            chat_template_tokenizer=d.get("chat_template_tokenizer", None),
            streaming=d.get("streaming", False),
        )


def _render_chat_template(
    row: dict,
    chat_messages_column: str,
    tokenizer_name: str,
    tokenizer_cache: dict[str, AutoTokenizer],
) -> str:
    """Render chat messages through tokenizer.apply_chat_template(tokenize=False)."""
    if tokenizer_name not in tokenizer_cache:
        tokenizer_cache[tokenizer_name] = AutoTokenizer.from_pretrained(tokenizer_name)

    try:
        messages = _resolve_path(row, chat_messages_column)
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(
            f"Chat messages field '{chat_messages_column}' not found in row. "
            f"Available keys: {list(row.keys())}"
        ) from e

    if not isinstance(messages, list):
        raise ValueError(
            f"Chat messages field '{chat_messages_column}' must be a list, "
            f"got {type(messages)}"
        )

    tokenizer = tokenizer_cache[tokenizer_name]
    return tokenizer.apply_chat_template(messages, tokenize=False)


# ---------------------------------------------------------------------------
# Core sampling logic
# ---------------------------------------------------------------------------

def sample_one(
    cfg: DatasetConfig,
    seed: int | None,
    tokenizer_cache: dict[str, AutoTokenizer],
) -> list[dict]:
    """Load a single dataset and return cfg.n random samples as plain dicts."""
    label = f"'{cfg.repo}'" + (f" ({cfg.config})" if cfg.config else "")
    print(f"  Loading {label} split='{cfg.split}' n={cfg.n} ...")

    if cfg.streaming:
        ds = load_dataset(cfg.repo, cfg.config, split=cfg.split, streaming=True)
        buffer_size = max(cfg.n * 10, 10_000)
        ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
        samples = [dict(example) for _, example in zip(range(cfg.n), ds)]
    else:
        ds = load_dataset(cfg.repo, cfg.config, split=cfg.split)
        total = len(ds)
        if cfg.n > total:
            print(f"  Warning: requested {cfg.n} but '{cfg.repo}' only has {total}. Using all.")
            cfg.n = total
        rng = random.Random(seed)
        indices = rng.sample(range(total), cfg.n)
        samples = [dict(ds[i]) for i in sorted(indices)]

    # Build output records: {id, doc}
    dataset_slug = re.sub(r'[^a-z0-9]+', '_', cfg.repo.lower()).strip('_')
    records = []
    for idx, row in enumerate(samples):
        if cfg.chat_messages_column is not None:
            if cfg.chat_template_tokenizer is None:
                raise ValueError(
                    f"Dataset '{cfg.repo}' set chat_messages_column but no "
                    "chat_template_tokenizer was provided."
                )
            doc = _render_chat_template(
                row,
                cfg.chat_messages_column,
                cfg.chat_template_tokenizer,
                tokenizer_cache,
            )
        elif cfg.doc_template is not None:
            doc = render_template(cfg.doc_template, row)
        else:
            doc = ""
        records.append({
            "id": f"{dataset_slug}_{idx}",
            "doc": doc,
        })

    print(f"  Sampled {len(records)} row(s) from {label}.")
    return records


def sample_and_mix(
    dataset_configs: list[DatasetConfig],
    seed: int | None = None,
    shuffle_output: bool = True,
) -> list[dict]:
    """Sample from each dataset config and mix (concatenate + optionally shuffle) results."""
    all_samples: list[dict] = []
    tokenizer_cache: dict[str, AutoTokenizer] = {}
    for cfg in dataset_configs:
        all_samples.extend(sample_one(cfg, seed=seed, tokenizer_cache=tokenizer_cache))

    if shuffle_output and len(dataset_configs) > 1:
        rng = random.Random(seed)
        rng.shuffle(all_samples)

    return all_samples


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def build_configs_from_args(args) -> list[DatasetConfig]:
    repos = args.repos
    ns = args.n
    splits = args.splits or ["train"] * len(repos)
    configs = args.configs or [None] * len(repos)
    templates = args.doc_templates or [None] * len(repos)
    message_columns = args.chat_messages_columns or [None] * len(repos)
    chat_template_tokenizers = args.chat_template_tokenizers or [None] * len(repos)

    # Broadcast single value to all datasets
    if len(ns) == 1:        ns = ns * len(repos)
    if len(splits) == 1:    splits = splits * len(repos)
    if len(configs) == 1:   configs = configs * len(repos)
    if len(templates) == 1: templates = templates * len(repos)
    if len(message_columns) == 1: message_columns = message_columns * len(repos)
    if len(chat_template_tokenizers) == 1:
        chat_template_tokenizers = chat_template_tokenizers * len(repos)

    lengths = {
        "repos": len(repos), "n": len(ns), "splits": len(splits),
        "configs": len(configs), "doc_templates": len(templates),
        "chat_messages_columns": len(message_columns),
        "chat_template_tokenizers": len(chat_template_tokenizers),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"All list arguments must have the same length (or length 1). Got: {lengths}"
        )

    for i, (repo, msg_col, chat_tok) in enumerate(
        zip(repos, message_columns, chat_template_tokenizers)
    ):
        if msg_col is not None and chat_tok is None:
            raise ValueError(
                f"Dataset index {i} ('{repo}') set --chat-messages-columns but "
                "is missing --chat-template-tokenizers."
            )
        if msg_col is None and chat_tok is not None:
            raise ValueError(
                f"Dataset index {i} ('{repo}') set --chat-template-tokenizers but "
                "is missing --chat-messages-columns."
            )

    return [
        DatasetConfig(
            repo=repo, n=n, split=split, config=cfg,
            doc_template=tmpl,
            chat_messages_column=msg_col,
            chat_template_tokenizer=chat_tok,
            streaming=args.streaming,
        )
        for repo, n, split, cfg, tmpl, msg_col, chat_tok in zip(
            repos,
            ns,
            splits,
            configs,
            templates,
            message_columns,
            chat_template_tokenizers,
        )
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sample and mix rows from one or more HuggingFace datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- source spec (mutually exclusive: CLI flags vs JSON config file) ---
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--repos", nargs="+", metavar="REPO",
                     help="One or more HuggingFace repo IDs")
    src.add_argument("--config-file", metavar="FILE",
                     help="JSON file describing datasets (see schema in module docstring)")

    # --- per-dataset options (parallel lists, or single value broadcast) ---
    parser.add_argument("--n", nargs="+", type=int, metavar="N",
                        help="Number of samples per dataset (single value broadcasts to all)")
    parser.add_argument("--splits", nargs="+", metavar="SPLIT",
                        help="Split per dataset (default: train)")
    parser.add_argument("--configs", nargs="+", metavar="CONFIG",
                        help="HF config/subset name per dataset")
    parser.add_argument("--doc-templates", nargs="+", metavar="TMPL",
                        help='Format template per dataset, e.g. "{title}: {body}"')
    parser.add_argument("--chat-messages-columns", nargs="+", metavar="COL",
                        help=(
                            "Column/path per dataset containing chat messages list (e.g. messages). "
                            "If set, uses tokenizer chat templating instead of --doc-templates."
                        ))
    parser.add_argument("--chat-template-tokenizers", nargs="+", metavar="TOKENIZER",
                        help=(
                            "Tokenizer name/path per dataset used for apply_chat_template, "
                            "e.g. Qwen/Qwen2.5-1.5B-Instruct"
                        ))

    # --- global options ---
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming mode for all datasets")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Keep samples grouped by source instead of interleaving")
    parser.add_argument("--output", default=None,
                        help='Save mixed samples to this JSONL file (one {"id","doc"} per line)')
    parser.add_argument("--preview", action="store_true",
                        help="Print a preview of the first 3 samples")

    args = parser.parse_args()

    # --- build dataset configs ---
    if args.config_file:
        with open(args.config_file, encoding="utf-8") as f:
            cfg_data = json.load(f)
        dataset_configs = [DatasetConfig.from_dict(d) for d in cfg_data["datasets"]]
        seed   = cfg_data.get("seed", args.seed)
        output = cfg_data.get("output", args.output)
        if cfg_data.get("streaming", args.streaming):
            for dc in dataset_configs:
                dc.streaming = True
    else:
        if not args.n:
            parser.error("--n is required when using --repos")
        dataset_configs = build_configs_from_args(args)
        seed   = args.seed
        output = args.output

    print(f"\nMixing {len(dataset_configs)} dataset(s) ...")
    samples = sample_and_mix(
        dataset_configs,
        seed=seed,
        shuffle_output=not args.no_shuffle,
    )
    print(f"\nTotal samples: {len(samples)}")

    if output:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            for record in samples:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved {len(samples)} lines to '{output}'.")

    if args.preview:
        print("\n--- Preview (first 3 samples) ---")
        for record in samples[:3]:
            print(json.dumps(record, ensure_ascii=False))

    return samples


if __name__ == "__main__":
    main()