# convert_molmo2_multiimageqa_to_jsonl.py

import argparse
import json
import logging
import mimetypes
import random
from pathlib import Path
from hashlib import sha256
from urllib.parse import urlparse

import requests
from datasets import load_dataset
from tqdm import tqdm

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

logger = logging.getLogger(__name__)


def guess_ext(url: str, content_type: str | None = None) -> str:
    """
    Guess image extension from URL or HTTP content-type.
    """
    if content_type:
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if ext:
            if ext == ".jpe":
                return ".jpg"
            return ext

    path = urlparse(url).path
    suffix = Path(path).suffix.lower()

    if suffix in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"]:
        return ".jpg" if suffix == ".jpeg" else suffix

    return ".jpg"


def download_image(
    url: str,
    out_dir: Path,
    expected_sha256: str | None = None,  # kept for backward compatibility; not verified
    timeout: int = 30,
) -> str:
    """
    Download one image and return its local path.

    This function tries exactly once. If the request fails, the exception is
    raised to the caller.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading image once: url=%s", url)

    resp = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()

    # Reject obvious non-image responses, such as HTML error pages.
    if content_type and not content_type.startswith("image/"):
        raise ValueError(
            f"URL did not return an image: url={url}, content_type={content_type}"
        )

    data = resp.content
    actual_sha = sha256(data).hexdigest()

    ext = guess_ext(url, resp.headers.get("Content-Type"))

    # Prevent saving .html, .txt, etc. as image files.
    if ext.lower() not in ALLOWED_IMAGE_EXTS:
        raise ValueError(
            f"Rejected non-image file extension: url={url}, ext={ext}, "
            f"content_type={resp.headers.get('Content-Type')}"
        )

    path = out_dir / f"{actual_sha}{ext}"

    if path.exists():
        logger.info("Image already exists: path=%s", path)
        return str(path)

    with open(path, "wb") as f:
        f.write(data)

    logger.info(
        "Saved image: path=%s url=%s bytes=%s sha256=%s content_type=%s",
        path,
        url,
        len(data),
        actual_sha,
        resp.headers.get("Content-Type"),
    )

    return str(path)


def normalize_qa_pairs(qa_pairs):
    """
    Convert dataset qa_pairs into a list of (question, answer).

    Expected common format:
        {
            "question": ["q1", "q2", ...],
            "answer": ["a1", "a2", ...]
        }
    """
    if isinstance(qa_pairs, dict):
        questions = qa_pairs.get("question", [])
        answers = qa_pairs.get("answer", [])

        if isinstance(questions, str):
            questions = [questions]
        if isinstance(answers, str):
            answers = [answers]

        return list(zip(questions, answers))

    if isinstance(qa_pairs, list):
        pairs = []

        for item in qa_pairs:
            if isinstance(item, dict):
                q = item.get("question")
                a = item.get("answer")

                if q is not None and a is not None:
                    pairs.append((q, a))

        return pairs

    raise ValueError(f"Unsupported qa_pairs format: {type(qa_pairs)}")


def build_conversation(question: str, answer: str):
    """
    Build a single-turn conversation for one QA pair.

    Multiple QA pairs from one source example are written as separate datapoints.
    """
    return [
        {
            "role": "user",
            "content": question,
            "img_loc": "before",
        },
        {
            "role": "assistant",
            "content": answer,
            "img_loc": None,
        },
    ]


def file_ends_with_newline(path: Path) -> bool:
    """
    Return True if a file is empty or ends with a newline.
    """
    if not path.exists() or path.stat().st_size == 0:
        return True

    with open(path, "rb") as f:
        f.seek(-1, 2)
        return f.read(1) == b"\n"


def load_existing_datapoint_ids_and_clean(output_jsonl: Path) -> set[str]:
    """
    Load existing datapoint IDs from a JSONL file.

    If the file has invalid or partial lines from an interrupted run, they are
    removed so appending can resume safely.
    """
    existing_ids: set[str] = set()

    if not output_jsonl.exists():
        return existing_ids

    valid_lines = []
    invalid_lines = 0

    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[RESUME] Dropping invalid JSONL line {line_num}")
                invalid_lines += 1
                continue

            item_id = obj.get("id")

            if item_id is None:
                print(f"[RESUME] Dropping JSONL line {line_num} with no id")
                invalid_lines += 1
                continue

            item_id = str(item_id)

            if item_id in existing_ids:
                print(f"[RESUME] Dropping duplicate datapoint id={item_id}")
                invalid_lines += 1
                continue

            existing_ids.add(item_id)
            valid_lines.append(json.dumps(obj, ensure_ascii=False))

    needs_rewrite = invalid_lines > 0 or not file_ends_with_newline(output_jsonl)

    if needs_rewrite:
        tmp_path = output_jsonl.with_suffix(output_jsonl.suffix + ".tmp")

        with open(tmp_path, "w", encoding="utf-8") as f:
            for line in valid_lines:
                f.write(line + "\n")

        tmp_path.replace(output_jsonl)

    return existing_ids


def load_download_manifest(manifest_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """
    Load persistent URL cache.

    Returns:
        downloaded_url_cache: url -> local image path
        failed_url_cache: url -> error message

    Supports both the current manifest format and the older simple
    url -> local_path format.
    """
    if not manifest_path.exists():
        return {}, {}

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[RESUME] Could not read manifest {manifest_path}: {e}")
        return {}, {}

    if not isinstance(data, dict):
        return {}, {}

    # Current format:
    # {
    #   "downloaded": {"url": "path"},
    #   "failed": {"url": "error"}
    # }
    if "downloaded" in data or "failed" in data:
        downloaded = data.get("downloaded", {})
        failed = data.get("failed", {})

        if not isinstance(downloaded, dict):
            downloaded = {}

        if isinstance(failed, list):
            failed = {str(url): "previously failed" for url in failed}
        elif not isinstance(failed, dict):
            failed = {}

        downloaded = {str(k): str(v) for k, v in downloaded.items()}
        failed = {str(k): str(v) for k, v in failed.items()}

        return downloaded, failed

    # Backward-compatible older format:
    # {
    #   "url": "path"
    # }
    downloaded = {str(k): str(v) for k, v in data.items()}
    return downloaded, {}


def save_download_manifest(
    manifest_path: Path,
    downloaded_url_cache: dict[str, str],
    failed_url_cache: dict[str, str],
) -> None:
    """
    Save persistent URL cache atomically.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "downloaded": downloaded_url_cache,
        "failed": failed_url_cache,
    }

    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    tmp_path.replace(manifest_path)


def get_cached_or_download_image(
    url: str,
    out_dir: Path,
    expected_sha256: str | None,
    downloaded_url_cache: dict[str, str],
    failed_url_cache: dict[str, str],
    manifest_path: Path,
) -> str:
    """
    Return local image path for a URL.

    Behavior:
    - If URL was already downloaded and local file exists, reuse it.
    - If URL previously failed, do not retry it.
    - Otherwise, try downloading exactly once.
    - On failure, remember that URL as failed.
    """
    if url in failed_url_cache:
        raise RuntimeError(
            f"URL previously failed; skipping without retry: {url}"
        )

    cached_path = downloaded_url_cache.get(url)

    if cached_path and Path(cached_path).exists():
        return cached_path

    if cached_path and not Path(cached_path).exists():
        logger.warning(
            "Cached image path no longer exists, removing cache entry: url=%s path=%s",
            url,
            cached_path,
        )
        downloaded_url_cache.pop(url, None)
        save_download_manifest(
            manifest_path,
            downloaded_url_cache,
            failed_url_cache,
        )

    try:
        local_path = download_image(
            url=url,
            out_dir=out_dir,
            expected_sha256=expected_sha256,
        )
    except Exception as e:
        failed_url_cache[url] = f"{type(e).__name__}: {e}"
        save_download_manifest(
            manifest_path,
            downloaded_url_cache,
            failed_url_cache,
        )
        raise

    downloaded_url_cache[url] = local_path
    failed_url_cache.pop(url, None)

    save_download_manifest(
        manifest_path,
        downloaded_url_cache,
        failed_url_cache,
    )

    return local_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", default="allenai/Molmo2-MultiImageQA")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--image_dir", required=True)

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help=(
            "Write this many successful datapoints. A datapoint is one QA pair, "
            "not one source example. If images fail, the script keeps trying "
            "later source examples until this many datapoints are written or "
            "the dataset is exhausted."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling.",
    )

    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle source examples before processing.",
    )

    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )

    parser.add_argument(
        "--on_image_error",
        choices=["skip", "raise"],
        default="skip",
        help=(
            "If an image fails to download, skip the whole source example or "
            "raise the error. Downloads are not retried."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from an existing JSONL file. Existing datapoint IDs are "
            "skipped, and the persistent download manifest is reused."
        ),
    )

    parser.add_argument(
        "--log_level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Python logging level.",
    )

    parser.add_argument(
        "--image_on_each_user_turn",
        action="store_true",
        help=(
            "Accepted for backward compatibility with older commands. Ignored "
            "because this script writes one QA pair per datapoint."
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    image_dir = Path(args.image_dir)
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    manifest_path = output_jsonl.with_suffix(
        output_jsonl.suffix + ".download_manifest.json"
    )

    if args.resume:
        existing_datapoint_ids = load_existing_datapoint_ids_and_clean(output_jsonl)
        downloaded_url_cache, failed_url_cache = load_download_manifest(manifest_path)

        print(f"[RESUME] Existing datapoints found: {len(existing_datapoint_ids)}")
        print(f"[RESUME] Cached downloaded URLs: {len(downloaded_url_cache)}")
        print(f"[RESUME] Cached failed URLs: {len(failed_url_cache)}")
    else:
        existing_datapoint_ids = set()
        downloaded_url_cache = {}
        failed_url_cache = {}

        if manifest_path.exists():
            manifest_path.unlink()

    print(f"Loading dataset: {args.dataset_name}, split={args.split}")
    print(f"Cache directory: {args.cache_dir}")

    ds = load_dataset(
        args.dataset_name,
        split=args.split,
        cache_dir=args.cache_dir,
    )

    total_source_examples = len(ds)
    indices = list(range(total_source_examples))

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(indices)

    target_samples = args.max_samples

    num_written_total = len(existing_datapoint_ids) if args.resume else 0
    num_written_new = 0

    num_processed_source_examples = 0
    num_skipped_source_examples = 0
    num_skipped_empty_qa_examples = 0
    num_skipped_image_failed_examples = 0
    num_skipped_already_done_datapoints = 0

    print(f"Total source examples in split: {total_source_examples}")
    print(f"Candidate source examples available: {len(indices)}")
    print(
        "Target successful datapoints: "
        f"{target_samples if target_samples is not None else 'all available'}"
    )

    if args.resume:
        print(
            "Current successful datapoints before this run: "
            f"{num_written_total}"
        )

    open_mode = "a" if args.resume else "w"

    with open(output_jsonl, open_mode, encoding="utf-8") as fout:
        for idx in tqdm(indices):
            if target_samples is not None and num_written_total >= target_samples:
                break

            num_processed_source_examples += 1

            ex = ds[idx]

            image_urls = list(ex["image_urls"])

            image_sha256s = ex.get("image_sha256s")
            if not image_sha256s:
                image_sha256s = [None] * len(image_urls)
            else:
                image_sha256s = list(image_sha256s)
                if len(image_sha256s) < len(image_urls):
                    image_sha256s.extend(
                        [None] * (len(image_urls) - len(image_sha256s))
                    )

            qa_pairs = normalize_qa_pairs(ex["qa_pairs"])

            if not qa_pairs:
                num_skipped_source_examples += 1
                num_skipped_empty_qa_examples += 1
                continue

            pending_qa_items = []

            for qa_idx, (question, answer) in enumerate(qa_pairs):
                item_id = f"{idx}_{qa_idx}"

                if item_id in existing_datapoint_ids:
                    num_skipped_already_done_datapoints += 1
                    continue

                pending_qa_items.append(
                    {
                        "item_id": item_id,
                        "qa_idx": qa_idx,
                        "question": question,
                        "answer": answer,
                    }
                )

            if not pending_qa_items:
                continue

            if target_samples is not None:
                remaining = target_samples - num_written_total

                if remaining <= 0:
                    break

                pending_qa_items = pending_qa_items[:remaining]

            local_image_paths = []

            try:
                for url, expected_hash in zip(image_urls, image_sha256s):
                    local_path = get_cached_or_download_image(
                        url=url,
                        out_dir=image_dir,
                        expected_sha256=expected_hash,
                        downloaded_url_cache=downloaded_url_cache,
                        failed_url_cache=failed_url_cache,
                        manifest_path=manifest_path,
                    )
                    local_image_paths.append(local_path)

            except Exception as e:
                if args.on_image_error == "raise":
                    raise

                print(
                    f"[SKIP] source example index={idx}, "
                    f"image download failed once: {e}"
                )
                num_skipped_source_examples += 1
                num_skipped_image_failed_examples += 1
                continue

            for qa_item in pending_qa_items:
                if target_samples is not None and num_written_total >= target_samples:
                    break

                item = {
                    "id": qa_item["item_id"],
                    "source_index": str(idx),
                    "qa_index": qa_item["qa_idx"],
                    "images": local_image_paths,
                    "conversation": build_conversation(
                        qa_item["question"],
                        qa_item["answer"],
                    ),
                }

                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                fout.flush()

                existing_datapoint_ids.add(qa_item["item_id"])
                num_written_total += 1
                num_written_new += 1

    print("Done.")
    print(f"New datapoints written this run:       {num_written_new}")
    print(f"Total datapoints in output now:        {num_written_total}")
    print(f"Processed source examples this run:    {num_processed_source_examples}")
    print(f"Skipped source examples:               {num_skipped_source_examples}")
    print(f"Skipped empty-QA examples:             {num_skipped_empty_qa_examples}")
    print(f"Skipped image-failed examples:         {num_skipped_image_failed_examples}")
    print(f"Skipped already-done datapoints:       {num_skipped_already_done_datapoints}")
    print(f"Cached downloaded URLs:                {len(downloaded_url_cache)}")
    print(f"Cached failed URLs:                    {len(failed_url_cache)}")

    if target_samples is not None and num_written_total < target_samples:
        print(
            f"Warning: requested {target_samples} datapoints, "
            f"but only found/wrote {num_written_total} successful datapoints."
        )

    print(f"Output:   {output_jsonl}")
    print(f"Images:   {image_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()