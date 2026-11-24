"""Measure how fast training can read frame-list samples from a Modal volume (e.g. RadGenome CT slab videos).

Reads and decodes every frame of N random rows of a JSONL with a pool of W processes (like dataloader workers),
twice: a cold pass and a warm pass over the same rows (Modal volumes cache what was read).

    modal run scripts/modal_mm_tuning/bench_frame_reads.py --jsonl /mnt/radgenome/radgenome_train_video_modal.jsonl
"""
import os

import modal

VOLUME = os.environ.get("BENCH_VOLUME", "radgenome")
MOUNT = f"/mnt/{VOLUME}"

app = modal.App("bench-frame-reads")
image = modal.Image.debian_slim().pip_install("pillow").env({"BENCH_VOLUME": VOLUME})
vol = modal.Volume.from_name(VOLUME, create_if_missing=False)


def _read_row(paths):
    from PIL import Image
    for p in paths:
        with Image.open(p) as im:
            im.convert("RGB").load()
    return len(paths)


@app.function(image=image, volumes={MOUNT: vol}, cpu=24, memory=32768, timeout=3600)
def bench(jsonl: str, rows: int = 400, workers: int = 24, seed: int = 0):
    import json, random, time
    from multiprocessing import Pool

    with open(jsonl) as f:
        lines = f.readlines()
    picked = random.Random(seed).sample(lines, min(rows, len(lines)))
    samples = []
    for line in picked:
        r = json.loads(line)
        samples.append([p for v in r.get("videos") or [] for p in v] + list(r.get("images") or []))
    frames = sum(len(s) for s in samples)
    for label in ("cold", "warm"):
        t = time.time()
        with Pool(workers) as pool:
            done = sum(pool.imap_unordered(_read_row, samples, chunksize=1))
        dt = time.time() - t
        print(f"{label}: {len(samples)} samples / {done} frames in {dt:.1f}s -> {len(samples)/dt:.1f} samples/s, "
              f"{done/dt:.0f} frames/s ({workers} workers)")
    return frames


@app.local_entrypoint()
def main(jsonl: str, rows: int = 400, workers: int = 24):
    bench.remote(jsonl, rows, workers)
