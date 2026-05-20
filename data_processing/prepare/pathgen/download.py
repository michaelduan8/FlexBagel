"""
GDC Batch WSI Download & Patch Extraction Pipeline
====================================================
Usage:
    python gdc_batch_pipeline.py \
        --metadata PathGen-1.6M.json \
        --wsi-dir /scratch/wsi \
        --output-dir ./output \
        --batch-size 500 \
        --max-workers 8 \
        --gdc-token gdc-token.txt   # optional, for controlled-access data

The metadata JSON must be a list of dicts, each with at least:
    - "file_id"  : GDC file UUID  (used to download the WSI)
    - "wsi_id"   : slide filename stem (used to locate the .svs on disk)
    - "position" : [x, y] coordinates for the patch

Progress is written to pipeline_progress.json after every patch so the
pipeline can resume exactly where it left off after any interruption.
Failed items are logged to pipeline_failures.json for later inspection.
"""

import os
import sys
import json
import shutil
import logging
import argparse
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path
from typing import List, Dict, Set, Any

from PIL import Image
import openslide
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging — file only; tqdm owns stdout
# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

_file_handler = logging.FileHandler("pipeline.log")
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
)
log.addHandler(_file_handler)


def _tprint(msg: str) -> None:
    """Print above the active tqdm bar without corrupting it."""
    tqdm.write(msg)


# ---------------------------------------------------------------------------
# Progress / failure state
# ---------------------------------------------------------------------------
PROGRESS_FILE = "pipeline_progress.json"
FAILURES_FILE = "pipeline_failures.json"


def load_progress() -> Dict[str, Any]:
    """
    Schema
    ------
    {
      "completed_batches": ["batch_0000", ...],
      "processed_patches": ["WSI_id_x_y", ...],   # per-patch resume
      "total_processed": 0,
      "total_skipped":   0,
      "total_failed":    0,
      "batch_stats": {
          "batch_0000": {"processed": N, "skipped": N, "failed": N,
                         "elapsed_s": N}
      }
    }
    """
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {
        "completed_batches": [],
        "processed_patches": [],
        "total_processed": 0,
        "total_skipped": 0,
        "total_failed": 0,
        "batch_stats": {},
    }


def _atomic_save(path: str, obj: Any) -> None:
    """Write JSON via a temp file so a crash never corrupts the state."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def save_progress(progress: Dict[str, Any]) -> None:
    _atomic_save(PROGRESS_FILE, progress)


def load_failures() -> List[Dict[str, Any]]:
    if os.path.exists(FAILURES_FILE):
        with open(FAILURES_FILE) as f:
            return json.load(f)
    return []


def append_failure(record: Dict[str, Any]) -> None:
    failures = load_failures()
    failures.append(record)
    _atomic_save(FAILURES_FILE, failures)


# ---------------------------------------------------------------------------
# GDC download helpers
# ---------------------------------------------------------------------------

def build_manifest(file_ids: List[str], manifest_path: str) -> None:
    with open(manifest_path, "w") as f:
        f.write("id\tfilename\tmd5\tsize\tstate\n")
        for fid in file_ids:
            f.write(f"{fid}\t\t\t\t\n")
    log.info("Manifest: %s  (%d files)", manifest_path, len(file_ids))


def download_batch(
    file_ids: List[str],
    wsi_dir: str,
    gdc_token: str | None,
    n_processes: int = 8,
) -> bool:
    """Run gdc-client and return True on success."""
    os.makedirs(wsi_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="gdc_manifest_"
    ) as tmp:
        manifest_path = tmp.name
    build_manifest(file_ids, manifest_path)

    cmd = [
        "/home1/duanm/gdc-client", "download",
        "--manifest",     manifest_path,
        "--dir",          wsi_dir,
        "--n-processes",   str(n_processes),
    ]
    if gdc_token:
        cmd += ["--token-file", gdc_token]

    _tprint(f"  ↓  {' '.join(cmd)}")
    log.info("Running: %s", " ".join(cmd))

    # inherit stdout/stderr so gdc-client's own progress is visible
    result = subprocess.run(cmd)
    os.unlink(manifest_path)

    if result.returncode != 0:
        log.error("gdc-client exited %d", result.returncode)
        return False

    _flatten_gdc_download(wsi_dir)
    return True


def _flatten_gdc_download(wsi_dir: str) -> None:
    """
    gdc-client saves  <wsi_dir>/<file_id>/<filename>.svs
    Flatten to        <wsi_dir>/<filename>.svs
    """
    WSI_EXTS = {".svs", ".ndpi", ".tif", ".tiff", ".scn", ".mrxs", ".vms"}
    for entry in Path(wsi_dir).iterdir():
        if entry.is_dir():
            for wsi_file in entry.iterdir():
                if wsi_file.suffix.lower() in WSI_EXTS:
                    dest = Path(wsi_dir) / wsi_file.name
                    if not dest.exists():
                        wsi_file.rename(dest)
                        log.debug("Moved %s → %s", wsi_file, dest)
            try:
                entry.rmdir()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

PATCH_SIZE = (672, 672)
WSI_EXTS   = [".svs", ".ndpi", ".tif", ".tiff", ".scn", ".mrxs", ".vms"]


def extract_patch_from_wsi(
    wsi_path: str, position: List[str], patch_size: tuple
) -> "Image.Image | None":
    try:
        wsi = openslide.OpenSlide(wsi_path)
        x, y = map(int, position)
        patch = wsi.read_region((x, y), 0, patch_size)
        wsi.close()
        return patch
    except Exception as exc:
        log.error("Extraction failed %s @ %s: %s", wsi_path, position, exc)
        return None


def process_batch_items(
    items: List[Dict[str, Any]],
    wsi_dir: str,
    output_dir: str,
    already_done: Set[str],
    progress: Dict[str, Any],
    batch_label: str,
    max_workers: int = 4,
) -> Dict[str, int]:
    """
    Extract patches for every item, skipping already_done patches.
    Patches are processed in parallel across `max_workers` threads.
    A single lock serialises all writes to `progress` and the failures file
    so progress is flushed to disk safely after every successful patch.
    """
    os.makedirs(output_dir, exist_ok=True)
    counts = {"processed": 0, "skipped": 0, "failed": 0}
    lock = threading.Lock()

    # ------------------------------------------------------------------
    # Per-item worker (runs in a thread-pool thread)
    # ------------------------------------------------------------------
    def _process_one(item: Dict[str, Any]) -> tuple[str, Any, Any]:
        """
        Returns (status, failure_record_or_None, output_patch_or_None).
        All updates to shared state are done by the caller under `lock`.
        """
        wsi_id   = item["wsi_id"]
        position = item["position"]
        output_patch = os.path.join(
            output_dir, wsi_id, f"{position[0]}_{position[1]}.png"
        )
        os.makedirs(os.path.dirname(output_patch), exist_ok=True)

        # ── already done (mid-batch resume) ──────────────────────────
        if output_patch in already_done:
            return "skipped", None, None

        # ── locate WSI ────────────────────────────────────────────────
        wsi_path = next(
            (
                os.path.join(wsi_dir, f"{wsi_id}{ext}")
                for ext in WSI_EXTS
                if os.path.exists(os.path.join(wsi_dir, f"{wsi_id}{ext}"))
            ),
            None,
        )
        if wsi_path is None:
            log.warning("WSI not found: %s", wsi_id)
            return "failed", {
                "batch": batch_label, "wsi_id": wsi_id,
                "position": position, "reason": "wsi_not_found",
            }, None

        # ── extract (OpenSlide releases the GIL here) ─────────────────
        patch = extract_patch_from_wsi(wsi_path, position, PATCH_SIZE)
        if patch is None:
            return "failed", {
                "batch": batch_label, "wsi_id": wsi_id,
                "position": position, "reason": "extraction_error",
            }, None

        # ── save patch (pure I/O, no shared state touched yet) ────────
        patch.save(output_patch)
        log.info("Saved %s", output_patch)
        return "processed", None, output_patch

    # ------------------------------------------------------------------
    # Dispatch + collect
    # ------------------------------------------------------------------
    with tqdm(
        total=len(items),
        desc=f"  Patches {batch_label}",
        unit="patch",
        dynamic_ncols=True,
        leave=True,
    ) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {
                executor.submit(_process_one, item): item for item in items
            }

            for future in as_completed(future_to_item):
                try:
                    status, failure, output_patch = future.result()
                except Exception as exc:
                    # Unexpected exception in the worker itself
                    item = future_to_item[future]
                    log.error("Unhandled worker error for %s: %s", item, exc)
                    status, failure, output_patch = "failed", {
                        "batch": batch_label,
                        "wsi_id": item.get("wsi_id"),
                        "position": item.get("position"),
                        "reason": f"unhandled_exception: {exc}",
                    }, None

                # ── all shared-state updates serialised by lock ───────
                with lock:
                    counts[status] += 1

                    if status == "processed":
                        progress["total_processed"] += 1
                        progress["processed_patches"].append(output_patch)
                        save_progress(progress)   # atomic write, safe here

                    elif status == "failed" and failure:
                        append_failure(failure)   # also does atomic write

                pbar.update(1)
                pbar.set_postfix(
                    ok=counts["processed"],
                    skip=counts["skipped"],
                    fail=counts["failed"],
                    refresh=False,
                )

    return counts


# ---------------------------------------------------------------------------
# WSI cleanup
# ---------------------------------------------------------------------------

def delete_batch_wsis(wsi_dir: str) -> None:
    wsi_ext_set = {e for e in WSI_EXTS}
    removed = 0
    for p in Path(wsi_dir).iterdir():
        if p.is_file() and p.suffix.lower() in wsi_ext_set:
            p.unlink()
            removed += 1
        elif p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    _tprint(f"  🗑  Deleted {removed} WSI file(s) from {wsi_dir}")
    log.info("Deleted %d WSI file(s)", removed)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _fmt(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def _print_startup_summary(
    total_records: int,
    total_file_ids: int,
    n_batches: int,
    n_done: int,
    batch_size: int,
    max_workers: int,
) -> None:
    w = 60
    lines = [
        "", "═" * w,
        "  GDC Batch WSI Pipeline",
        "═" * w,
        f"  Total metadata records : {total_records:,}",
        f"  Unique file IDs        : {total_file_ids:,}",
        f"  Batch size             : {batch_size}",
        f"  Total batches          : {n_batches}",
        f"  Already completed      : {n_done}",
        f"  Remaining              : {n_batches - n_done}",
        f"  Patch workers          : {max_workers}",
        "═" * w, "",
    ]
    for line in lines:
        _tprint(line)


def _print_batch_summary(
    batch_label: str,
    batch_idx: int,
    n_batches: int,
    counts: Dict[str, int],
    elapsed: float,
    progress: Dict[str, Any],
) -> None:
    lines = [
        "",
        f"  ✔  {batch_label}  ({batch_idx + 1}/{n_batches})",
        f"     Processed : {counts['processed']:,}",
        f"     Skipped   : {counts['skipped']:,}",
        f"     Failed    : {counts['failed']:,}",
        f"     Elapsed   : {_fmt(elapsed)}",
        f"     ─ Cumulative ─",
        f"     Processed : {progress['total_processed']:,}",
        f"     Failed    : {progress['total_failed']:,}",
        "",
    ]
    for line in lines:
        _tprint(line)
    log.info(
        "%s | ok=%d skip=%d fail=%d elapsed=%.1fs",
        batch_label, counts["processed"], counts["skipped"],
        counts["failed"], elapsed,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    metadata_path: str,
    wsi_dir: str,
    output_dir: str,
    batch_size: int,
    batch_pause_s: int,
    gdc_token: str | None,
    n_processes: int,
    max_workers: int,
    dry_run: bool,
) -> None:

    # ── load metadata ─────────────────────────────────────────────────────
    _tprint(f"\nLoading metadata from {metadata_path} …")
    with open(metadata_path) as f:
        data: List[Dict[str, Any]] = json.load(f)

    file_id_to_items: Dict[str, List[Dict[str, Any]]] = {}
    for item in data:
        file_id_to_items.setdefault(item["file_id"], []).append(item)

    unique_file_ids = list(file_id_to_items.keys())
    batches = [
        unique_file_ids[i : i + batch_size]
        for i in range(0, len(unique_file_ids), batch_size)
    ]

    # ── load prior progress ───────────────────────────────────────────────
    progress          = load_progress()
    completed_batches = set(progress["completed_batches"])
    done_patches: Set[str] = set(progress["processed_patches"])

    _print_startup_summary(
        total_records=len(data),
        total_file_ids=len(unique_file_ids),
        n_batches=len(batches),
        n_done=len(completed_batches),
        batch_size=batch_size,
        max_workers=max_workers,
    )

    pipeline_start = time.time()

    # outer bar: one tick per batch
    with tqdm(
        total=len(batches),
        initial=len(completed_batches),
        desc="Overall batches",
        unit="batch",
        dynamic_ncols=True,
        colour="green",
    ) as batch_bar:

        for batch_idx, batch_file_ids in enumerate(batches):
            batch_label = f"batch_{batch_idx:04d}"

            if batch_label in completed_batches:
                continue  # already counted in `initial`

            batch_bar.set_description(f"Batches  [{batch_label}]")
            _tprint(
                f"\n── {batch_label}  "
                f"({batch_idx + 1}/{len(batches)})  "
                f"{len(batch_file_ids)} files ──"
            )

            if dry_run:
                _tprint(f"  [DRY RUN] Would download + process {batch_label}")
                batch_bar.update(1)
                continue

            batch_start = time.time()

            # 1. Download ──────────────────────────────────────────────────
            _tprint(f"  ↓  Downloading {len(batch_file_ids)} WSIs …")
            ok = download_batch(
                file_ids=batch_file_ids,
                wsi_dir=wsi_dir,
                gdc_token=gdc_token,
                n_processes=n_processes,
            )
            if not ok:
                _tprint(f"  ✗  Download failed for {batch_label} – skipping")
                append_failure({
                    "batch": batch_label,
                    "reason": "download_failed",
                    "file_ids": batch_file_ids,
                })
                batch_bar.update(1)
                continue

            # 2. Process ───────────────────────────────────────────────────
            batch_items: List[Dict[str, Any]] = []
            for fid in batch_file_ids:
                batch_items.extend(file_id_to_items[fid])

            counts = process_batch_items(
                items=batch_items,
                wsi_dir=wsi_dir,
                output_dir=output_dir,
                already_done=done_patches,
                progress=progress,
                batch_label=batch_label,
                max_workers=max_workers,
            )
            # absorb newly done patches into the master set
            done_patches.update(progress["processed_patches"])

            # 3. Delete WSIs ───────────────────────────────────────────────
            delete_batch_wsis(wsi_dir)

            # 4. Persist batch completion ───────────────────────────────────
            elapsed = time.time() - batch_start
            progress["completed_batches"].append(batch_label)
            progress["total_skipped"] += counts["skipped"]
            progress["total_failed"]  += counts["failed"]
            progress["batch_stats"][batch_label] = {
                "processed": counts["processed"],
                "skipped":   counts["skipped"],
                "failed":    counts["failed"],
                "elapsed_s": round(elapsed, 1),
            }
            save_progress(progress)

            _print_batch_summary(
                batch_label=batch_label,
                batch_idx=batch_idx,
                n_batches=len(batches),
                counts=counts,
                elapsed=elapsed,
                progress=progress,
            )
            batch_bar.update(1)

            if batch_pause_s > 0 and batch_idx < len(batches) - 1:
                _tprint(f"  ⏸  Sleeping {batch_pause_s}s before next batch...")
                time.sleep(batch_pause_s)

    # ── final summary ─────────────────────────────────────────────────────
    wall = time.time() - pipeline_start
    w = 60
    lines = [
        "", "═" * w,
        "  Pipeline complete",
        "═" * w,
        f"  Total processed : {progress['total_processed']:,}",
        f"  Total skipped   : {progress['total_skipped']:,}",
        f"  Total failed    : {progress['total_failed']:,}",
        f"  Wall time       : {_fmt(wall)}",
    ]
    if progress["total_failed"]:
        lines.append(f"  Failures logged : {FAILURES_FILE}")
    lines += ["═" * w, ""]
    for line in lines:
        _tprint(line)

    log.info(
        "Done. ok=%d skip=%d fail=%d elapsed=%s",
        progress["total_processed"],
        progress["total_skipped"],
        progress["total_failed"],
        _fmt(wall),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch-download GDC WSIs, extract patches, delete WSIs."
    )
    p.add_argument("--metadata",      required=True,
                   help="JSON metadata file (list of dicts with 'file_id').")
    p.add_argument("--wsi-dir",       default="./wsi_scratch",
                   help="Scratch directory for downloaded WSIs.")
    p.add_argument("--output-dir",    default="./output",
                   help="Directory for extracted patches.")
    p.add_argument("--batch-size",    type=int, default=500,
                   help="WSIs per batch (default: 500).")
    p.add_argument("--batch-pause-s", type=int, default=60,
                   help="Seconds to pause after each completed batch (default: 10).")
    p.add_argument("--gdc-token",     default=None,
                   help="GDC token file for controlled-access data (optional).")
    p.add_argument("--n-processes",   type=int, default=8,
                   help="gdc-client --n-processes (default: 8).")
    p.add_argument("--max-workers",   type=int, default=4,
                   help="Parallel patch-extraction threads per batch (default: 4).")
    p.add_argument("--dry-run",       action="store_true",
                   help="Show what would run without downloading anything.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        metadata_path=args.metadata,
        wsi_dir=args.wsi_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        batch_pause_s=args.batch_pause_s,
        gdc_token=args.gdc_token,
        n_processes=args.n_processes,
        max_workers=args.max_workers,
        dry_run=args.dry_run,
    )