"""Unzip an archive that is already on a Modal volume, server-side (replaces doing it in a Modal notebook).

    UNZIP_VOLUME=surgsigma modal run scripts/modal_mm_tuning/unzip_volume.py --zip-path surgsigma_images.zip --dest surgsigma

Paths are relative to the volume root (--dest . unzips into the root). Existing files are overwritten; pass
--delete-zip to remove the archive afterwards. The volume name is read from UNZIP_VOLUME (default surgsigma)
and baked into the image env, since the container re-imports this file and must mount the same volume.
"""
import os
import subprocess

import modal

VOLUME = os.environ.get("UNZIP_VOLUME", "surgsigma")
MOUNT = f"/mnt/{VOLUME}"

app = modal.App("unzip-volume")
image = modal.Image.debian_slim().apt_install("unzip").env({"UNZIP_VOLUME": VOLUME})
vol = modal.Volume.from_name(VOLUME, create_if_missing=False)


@app.function(image=image, volumes={MOUNT: vol}, cpu=4, timeout=6 * 60 * 60)
def unzip(zip_path: str, dest: str, delete_zip: bool = False):
    archive = os.path.join(MOUNT, zip_path)
    target = os.path.normpath(os.path.join(MOUNT, dest))
    os.makedirs(target, exist_ok=True)
    subprocess.run(["unzip", "-q", "-o", archive, "-d", target], check=True)
    n = sum(len(files) for _, _, files in os.walk(target))
    print(f"unzipped {archive} -> {target} ({n} files under {dest})")
    if delete_zip:
        os.remove(archive)
        print(f"removed {archive}")
    vol.commit()


@app.local_entrypoint()
def main(zip_path: str, dest: str, delete_zip: bool = False):
    unzip.remote(zip_path, dest, delete_zip)
