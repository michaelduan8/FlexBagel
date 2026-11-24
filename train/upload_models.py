#!/usr/bin/env python3
"""
upload_models_to_hf.py

Scan a root directory for directories whose relative path matches a given
glob-like pattern, then upload every leaf folder under each selected directory
as its own public model repo on the Hugging Face Hub and add every uploaded
repo to a single Hugging Face Collection.

Requires:
    pip install huggingface_hub

Auth:
    Either run `huggingface-cli login` beforehand, set the HF_TOKEN
    environment variable, or pass --token explicitly.

Example:
    python upload_models_to_hf.py \
        --root-dir /data/models \
        --pattern '*finetune*' \
        --namespace my-org \
        --collection-name "My Finetunes" \
        --collection-slug my-org/my-finetunes-64a1b2c3   # optional, see below

If you don't already have a collection, omit --collection-slug and the
script will create a new collection titled --collection-name under
--namespace, then print the resulting slug (save it for future runs so
everything lands in the same collection).

Dry run first to sanity check which folders would be uploaded:
    python upload_models_to_hf.py --root-dir /data/models --pattern '*finetune*' --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("upload_models_to_hf")


def find_selected_dirs(root_dir: Path, pattern: str) -> list[Path]:
    """Return directories under root_dir whose relative path matches pattern."""
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Root dir does not exist or is not a directory: {root_dir}")

    matches = sorted(
        p for p in root_dir.rglob("*")
        if p.is_dir() and p.relative_to(root_dir).match(pattern)
    )
    return matches


def find_leaf_dirs(selected_dir: Path) -> list[Path]:
    """Return every leaf directory under selected_dir, including itself if already a leaf."""
    leaf_dirs = []
    stack = [selected_dir]

    while stack:
        current_dir = stack.pop()
        child_dirs = sorted(p for p in current_dir.iterdir() if p.is_dir())
        if child_dirs:
            stack.extend(reversed(child_dirs))
            continue
        leaf_dirs.append(current_dir)

    return leaf_dirs


def build_repo_name(root_dir: Path, selected_dir: Path, leaf_dir: Path) -> str:
    """Build a stable repo name using the first directory under root_dir as the base."""
    root_relative_parts = leaf_dir.relative_to(root_dir).parts
    base_name = root_relative_parts[0]

    leaf_relative_parts = leaf_dir.relative_to(selected_dir).parts
    if not leaf_relative_parts:
        return base_name
    return "-".join((base_name, *leaf_relative_parts))


def get_or_create_collection(
    api: HfApi,
    collection_slug: str | None,
    collection_name: str | None,
    namespace: str | None,
    private: bool,
) -> str:
    """Return a collection slug, creating the collection if none was supplied."""
    if collection_slug:
        return collection_slug

    if not collection_name:
        raise ValueError("Must provide either --collection-slug or --collection-name.")

    log.info(f"No collection slug given, creating new collection '{collection_name}'...")
    collection = api.create_collection(
        title=collection_name,
        namespace=namespace,
        private=private,
        exists_ok=True,
    )
    log.info(f"Collection ready: slug={collection.slug}")
    return collection.slug


def upload_model_dir(
    api: HfApi,
    local_dir: Path,
    repo_id: str,
    private: bool,
    dry_run: bool,
) -> None:
    if dry_run:
        log.info(f"[dry-run] Would create repo '{repo_id}' and upload '{local_dir}'")
        return

    log.info(f"Creating (or reusing) repo: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    log.info(f"Uploading '{local_dir}' -> '{repo_id}' ...")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(local_dir),
        commit_message=f"Upload {local_dir.name}",
    )
    log.info(f"Done uploading {repo_id}")


def add_to_collection(api: HfApi, collection_slug: str, repo_id: str, dry_run: bool) -> None:
    if dry_run:
        log.info(f"[dry-run] Would add '{repo_id}' to collection '{collection_slug}'")
        return

    try:
        api.add_collection_item(
            collection_slug=collection_slug,
            item_id=repo_id,
            item_type="model",
            exists_ok=True,
        )
        log.info(f"Added {repo_id} to collection {collection_slug}")
    except HfHubHTTPError as e:
        log.warning(f"Could not add {repo_id} to collection {collection_slug}: {e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root-dir", required=True, type=Path, help="Root directory to recursively scan for model folder roots.")
    parser.add_argument("--pattern", required=True, help="Glob-like relative path pattern used to select directories under --root-dir, for example '*blahblah*/final'. All leaf folders beneath each match are uploaded.")
    parser.add_argument("--namespace", required=True, help="HF username or org to upload repos under (repo_id = namespace/folder_name).")
    parser.add_argument("--collection-slug", default=None, help="Existing collection slug, e.g. 'my-org/my-collection-64a1b2c3'. If omitted, a new collection is created.")
    parser.add_argument("--collection-name", default=None, help="Title for a new collection (used only if --collection-slug is not given).")
    parser.add_argument("--token", default=None, help="HF access token. Defaults to cached login / HF_TOKEN env var.")
    parser.add_argument("--private", action="store_true", help="Make uploaded repos AND the collection private (default: public).")
    parser.add_argument("--dry-run", action="store_true", help="List what would happen without uploading anything.")
    args = parser.parse_args()

    api = HfApi(token=args.token)

    try:
        selected_dirs = find_selected_dirs(args.root_dir, args.pattern)
    except FileNotFoundError as e:
        log.error(str(e))
        sys.exit(1)

    if not selected_dirs:
        log.warning(f"No directories under {args.root_dir} matched pattern '{args.pattern}'. Nothing to do.")
        sys.exit(0)

    log.info(f"Found {len(selected_dirs)} selected folder(s):")
    for d in selected_dirs:
        log.info(f"  - {d}")

    uploads: list[tuple[Path, str]] = []
    for selected_dir in selected_dirs:
        for leaf_dir in find_leaf_dirs(selected_dir):
            repo_name = build_repo_name(args.root_dir, selected_dir, leaf_dir)
            uploads.append((leaf_dir, f"{args.namespace}/{repo_name}"))

    if not uploads:
        log.warning("No leaf folders found under the selected folders. Nothing to do.")
        sys.exit(0)

    log.info(f"Resolved {len(uploads)} leaf folder(s) to upload:")
    for leaf_dir, repo_id in uploads:
        log.info(f"  - {leaf_dir} -> {repo_id}")

    collection_slug = get_or_create_collection(
        api=api,
        collection_slug=args.collection_slug,
        collection_name=args.collection_name,
        namespace=args.namespace,
        private=args.private,
    )

    failures = []
    for local_dir, repo_id in uploads:
        try:
            upload_model_dir(api, local_dir, repo_id, private=args.private, dry_run=args.dry_run)
            add_to_collection(api, collection_slug, repo_id, dry_run=args.dry_run)
        except Exception as e:
            log.error(f"Failed on {local_dir} ({repo_id}): {e}")
            failures.append(local_dir)

    log.info("=" * 60)
    log.info(f"Finished. {len(uploads) - len(failures)}/{len(uploads)} succeeded.")
    if failures:
        log.warning("Failed folders:")
        for f in failures:
            log.warning(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()