"""
Sample n random examples from a HuggingFace dataset repo.

Usage:
    python sample_hf_dataset.py --repo <repo_id> --n <num_samples> [options]

Examples:
    python sample_hf_dataset.py --repo stanfordnlp/imdb --n 100
    python sample_hf_dataset.py --repo stanfordnlp/imdb --n 50 --split test --seed 42
    python sample_hf_dataset.py --repo allenai/c4 --n 20 --split train --config en --output samples.json
"""

import argparse
import json
import random

from datasets import load_dataset


def sample_dataset(
    repo: str,
    n: int,
    split: str = "train",
    config: str | None = None,
    seed: int | None = None,
    output: str | None = None,
    streaming: bool = False,
) -> list[dict]:
    """
    Load a HuggingFace dataset and return n random samples.

    Args:
        repo:      HuggingFace dataset repo ID (e.g. 'stanfordnlp/imdb')
        n:         Number of samples to draw
        split:     Dataset split to use (default: 'train')
        config:    Dataset config/subset name, if required
        seed:      Random seed for reproducibility
        output:    If provided, save samples to this JSON file
        streaming: Use streaming mode (useful for very large datasets)

    Returns:
        List of sampled examples as dicts
    """
    print(f"Loading '{repo}' (split='{split}'{f', config={config!r}' if config else ''}) ...")

    if streaming:
        # Streaming: shuffle a buffer and take n items — avoids downloading the full dataset
        ds = load_dataset(repo, config, split=split, streaming=True)
        buffer_size = max(n * 10, 10_000)
        ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
        samples = [example for _, example in zip(range(n), ds)]
    else:
        ds = load_dataset(repo, config, split=split)
        total = len(ds)
        if n > total:
            print(f"Warning: requested {n} samples but dataset only has {total}. Returning all.")
            n = total

        rng = random.Random(seed)
        indices = rng.sample(range(total), n)
        samples = [dict(ds[i]) for i in sorted(indices)]

    print(f"Sampled {len(samples)} example(s).")

    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, default=str)
        print(f"Saved samples to '{output}'.")

    return samples


def main():
    parser = argparse.ArgumentParser(description="Sample n random rows from a HuggingFace dataset.")
    parser.add_argument("--repo",      required=True,          help="HuggingFace dataset repo ID")
    parser.add_argument("--n",         required=True, type=int, help="Number of samples to draw")
    parser.add_argument("--split",     default="train",         help="Dataset split (default: train)")
    parser.add_argument("--config",    default=None,            help="Dataset config/subset name")
    parser.add_argument("--seed",      default=None, type=int,  help="Random seed")
    parser.add_argument("--output",    default=None,            help="Save samples to this JSON file")
    parser.add_argument("--streaming", action="store_true",     help="Use streaming mode (for huge datasets)")
    parser.add_argument("--preview",   action="store_true",     help="Print a preview of the first 3 samples")
    args = parser.parse_args()

    samples = sample_dataset(
        repo=args.repo,
        n=args.n,
        split=args.split,
        config=args.config,
        seed=args.seed,
        output=args.output,
        streaming=args.streaming,
    )

    if args.preview:
        print("\n--- Preview (first 3 samples) ---")
        for i, sample in enumerate(samples[:3]):
            print(f"\n[{i}] {json.dumps(sample, indent=2, default=str)}")

    return samples


if __name__ == "__main__":
    main()