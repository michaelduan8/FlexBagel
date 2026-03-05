#!/usr/bin/env python3
"""
Embed texts from a JSONL file using GritLM-7B and compute the average embedding.

Example:
  python embed_gritlm_avg.py \
    --input data.jsonl \
    --text-field text \
    --model GritLM/GritLM-7B \
    --batch-size 8 \
    --max-length 512 \
    --out-avg avg.npy \
    --out-embs embs.npy
"""

import argparse
import json
import os
import psutil

import numpy as np
import torch

from transformers import AutoModel, AutoTokenizer
from typing import List, Optional, Tuple
from tqdm import tqdm
from vllm import LLM, PoolingParams


def read_jsonl_texts(path: str, text_field: str) -> List[str]:
    texts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {ln}: {e}") from e
            if text_field not in obj:
                raise KeyError(
                    f"Missing field '{text_field}' on line {ln}. "
                    f"Available keys: {list(obj.keys())}"
                )
            val = obj[text_field]
            if val is None:
                continue
            if not isinstance(val, str):
                val = str(val)
            val = val.strip()
            if val:
                texts.append(val)
    if not texts:
        raise ValueError(f"No non-empty texts found in {path} using field '{text_field}'.")
    return texts


@torch.no_grad()
def embed_hf(
    texts: List[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    device: str,
    dtype: str,
) -> np.ndarray:
    # dtype selection
    if dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp32":
        torch_dtype = torch.float32
    else:
        raise ValueError("dtype must be one of: fp16, bf16, fp32")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=None,  # we'll move manually
        trust_remote_code=True,  # many embedding models (including GritLM) may use this
    )
    model.eval()
    model.to(device)

    # If tokenizer has no pad token, set it (common for some LLM tokenizers)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_embs: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model(**enc)

        # Mean-pool last hidden state with attention mask
        last_hidden = out.last_hidden_state  # (B, T, H)
        mask = enc["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)  # (B, T, 1)

        summed = (last_hidden * mask).sum(dim=1)  # (B, H)
        counts = mask.sum(dim=1).clamp(min=1e-6)  # (B, 1)
        embs = summed / counts  # (B, H)

        # L2 normalize (common for retrieval embeddings; you can disable if you want)
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)

        all_embs.append(embs.detach().cpu().to(torch.float32).numpy())

    return np.concatenate(all_embs, axis=0)


def embed_vllm(embed_requests, cache_dir, model="Qwen/Qwen3-Embedding-0.6B", batch_size=64, instruction="{er}", mrl=None):
    # TODO: only supports vllm embedding right now
    os.makedirs(cache_dir, exist_ok=True)
    embed_file = os.path.join(cache_dir, "embeddings.npy")
    extra_args = {}
    if mrl:
        extra_args["hf_overrides"] = {"is_matryoshka": True, "matryoshka_dimensions": [mrl]}

    embedder = LLM(
        model=model,
        task="embed",
        enforce_eager=True,
        **extra_args
    )

    # Get virtual memory information
    mem_info = psutil.virtual_memory()

    # Calculate total and available RAM in GB
    total_ram_gb = mem_info.total / (1024**3)
    available_ram_gb = mem_info.available / (1024**3)
    used_ram_gb = mem_info.used / (1024**3)

    print(f"Total RAM: {total_ram_gb:.2f} GB")
    print(f"Used RAM: {used_ram_gb:.2f} GB")
    print(f"Available RAM: {available_ram_gb:.2f} GB")

    # TODO: make this toggle-able
    if True:
        # prepend clustering tag for nomic clustering
        print(embed_requests[0])
        embed_requests = [instruction.format(er=er) for er in tqdm(embed_requests)]

    embeddings_memmap = None
    if os.path.exists(embed_file):
        # Get embedding size from output
        test_embedding = embedder.embed(embed_requests[:1])
        embedding_size = len(test_embedding[0].outputs.embedding)
        output_shape = (len(embed_requests), embedding_size)
        embeddings_memmap = np.memmap(embed_file, dtype='float32', mode='r', shape=output_shape)
    else:
        print("generating from scratch")
        # Process passages in batches
        for i in tqdm(range(0, len(embed_requests), batch_size)):
            batch = embed_requests[i:i + batch_size]
            
            batch_out = embedder.embed(batch)
            batch_embed = np.array([o.outputs.embedding for o in batch_out])

            if embeddings_memmap is None:
                # Instantiate embeddings map with length of corpus and shape of embedding
                embed_shape = batch_embed.shape
                print(f"Embed shape: {batch_embed.shape}")
                output_shape = (len(embed_requests), embed_shape[1])
                print(f"Output shape: {output_shape}")
                embeddings_memmap = np.memmap(embed_file, dtype='float32', mode='w+', shape=output_shape)

            # Move embeddings to CPU and convert to numpy
            embeddings_memmap[i:i + len(batch)] = batch_embed
            embeddings_memmap.flush()

            assert np.all(np.equal(batch_embed[0], embeddings_memmap[i]))

    embedder.llm_engine.engine_core.shutdown()
    del embedder
    
    return embeddings_memmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input JSONL.")
    ap.add_argument("--text-field", default="text", help="JSON field that contains text.")
    ap.add_argument("--model", default="GritLM/GritLM-7B", help="HF model name/path.")
    ap.add_argument("--mode", default="vllm", help="What framework to run embedding inference in")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp32")
    ap.add_argument("--mrl", default=None, type=int, help="Optional dimension size if MRL supported.")
    ap.add_argument("--out-embs", default=None, help="Optional .npy path to save per-item embeddings.")
    ap.add_argument("--out-avg", default=None, help="Optional .npy path to save average embedding.")
    ap.add_argument("--no-normalize-avg", action="store_true", help="Do not L2-normalize the average vector.")
    args = ap.parse_args()

    texts = read_jsonl_texts(args.input, args.text_field)

    if args.mode == "vllm":
        instruction = ""
        embs = embed_vllm(
            embed_requests=texts,
            # TODO: A little hacky, need to clean up
            cache_dir=os.path.dirname(args.out_embs),
            model=args.model,
            batch_size=args.batch_size,
            instruction=instruction,
            mrl=args.mrl,
        )
    elif args.mode == "hf":
        embs = embed_hf(
            texts=texts,
            model_name=args.model,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
            dtype=args.dtype,
        )
    else:
        raise ValueError(f"Unsupported embedding mode: {args.mode}")

    avg = embs.mean(axis=0)  # average across samples
    if not args.no_normalize_avg:
        norm = np.linalg.norm(avg) + 1e-12
        avg = avg / norm

    print(f"Loaded {len(texts)} texts from: {args.input}")
    print(f"Embeddings shape: {embs.shape} (N, hidden)")
    print(f"Average embedding shape: {avg.shape} (hidden,)")

    if args.out_embs:
        os.makedirs(os.path.dirname(args.out_embs) or ".", exist_ok=True)
        np.save(args.out_embs, embs)
        print(f"Saved embeddings to: {args.out_embs}")

    if args.out_avg:
        os.makedirs(os.path.dirname(args.out_avg) or ".", exist_ok=True)
        np.save(args.out_avg, avg)
        print(f"Saved average embedding to: {args.out_avg}")


if __name__ == "__main__":
    main()