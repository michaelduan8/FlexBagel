from huggingface_hub import HfApi
import zstandard as zstd
import json
import random
import requests
import os
import io
import gzip

def debug_zst(path, n_lines=3):
    print("path:", path)
    print("size bytes:", os.path.getsize(path))

    with open(path, "rb") as f:
        head = f.read(64)
    print("first 16 bytes:", head[:16])
    # Zstandard frames usually start with 0x28B52FFD (little-endian: b'(\xb5/\xfd')
    print("looks like zstd magic?:", head.startswith(b"\x28\xb5\x2f\xfd"))

    dctx = zstd.ZstdDecompressor()
    try:
        with open(path, "rb") as fh, dctx.stream_reader(fh) as zfh:
            text = io.TextIOWrapper(zfh, encoding="utf-8", errors="replace", newline="")
            for i, line in enumerate(text):
                print(f"line {i} len={len(line)} preview={line[:120]!r}")
                if i + 1 >= n_lines:
                    break
            text.close()
    except Exception as e:
        print("decompress/read error:", repr(e))

percentages = {
    "common_crawl": 0.7605,
    "olmocr": 0.1358,
    "stack_edu": 0.069, 
    "arxiv": 0.0255,
    "finemath": 0.0086,
    "wiki": 0.0004
}

def dataset_download_links(dataset_id: str, revision: str = "main"):
    """
    Returns direct download URLs for all files in a Hugging Face *dataset* repo.
    """
    api = HfApi()
    files = api.list_repo_files(
        repo_id=dataset_id,
        repo_type="dataset",
        revision=revision,
    )
    base = f"https://huggingface.co/datasets/{dataset_id}/resolve/{revision}/"
    return [base + f for f in files]


def download_file(url, output_path, hf_token=None):
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    r = requests.get(url, headers=headers)
    r.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(r.content)

    print(f"Downloaded to {output_path}")


def sample_jsonl_zst(local_path, n, seed=0):
    random.seed(seed)
    reservoir = []
    seen = 0

    dctx = zstd.ZstdDecompressor()
    with open(local_path, "rb") as fh, dctx.stream_reader(fh) as zfh:
        # Make it line-iterable
        text = io.TextIOWrapper(zfh, encoding="utf-8")
        for line in text:
            seen += 1
            if seen <= n:
                reservoir.append(line)
            else:
                j = random.randint(1, seen)
                if j <= n:
                    reservoir[j - 1] = line
        text.close()  # closes wrapper (zfh still closed by context manager)

    return [json.loads(x) for x in reservoir]

n = 1000
random.seed(2026)


def sample_jsonl_gz(local_path, n, seed=0):
    random.seed(seed)
    reservoir = []
    seen = 0

    with gzip.open(local_path, "rt", encoding="utf-8") as f:
        for line in f:
            seen += 1
            if seen <= n:
                reservoir.append(line)
            else:
                j = random.randint(1, seen)
                if j <= n:
                    reservoir[j - 1] = line

    return [json.loads(x) for x in reservoir]


# Example:
links = dataset_download_links("allenai/dolma3_mix-6T-1025-7B")  # replace with your dataset id
links = [l for l in links if l.startswith("https://huggingface.co/datasets/allenai/dolma3_mix-6T-1025-7B/resolve/main/data/")]

for l in links:
    assert any(key in l for key in percentages.keys()), f"Link '{l}' does not contain any of the expected keys."

cate_links = {key: [] for key in percentages.keys()}
for key in percentages.keys():
    cate_links[key] = [link for link in links if key in link]
    print(f"{key}: {len(cate_links[key])} files, {percentages[key]*100:.2f}% of total")

all_samples = []
for domain, links in cate_links.items():
    print(f"Downloading {len(links)} files for domain '{domain}'...")
    
    domain_size = percentages[domain] * n

    # Group links by directory
    dir_groups = {}
    for link in links:
        # Extract directory path from link
        dir_path = "/".join(link.split("/")[:-1])
        if dir_path not in dir_groups:
            dir_groups[dir_path] = []
        dir_groups[dir_path].append(link)
    
    dir_lengths = {dir_path: len(dir_links) for dir_path, dir_links in dir_groups.items()}
    dir_sizes = {dir_path: round(domain_size * (length / sum(dir_lengths.values()))) for dir_path, length in dir_lengths.items()}
    # Sample 1 file from each directory
    dir_samples = {dir_path: random.sample(dir_links, 1)[0] for dir_path, dir_links in dir_groups.items()}

    for dir_path, sample_link in dir_samples.items():
        save_path = os.path.join("/work/hdd/bfgx/seanlyu2/flexolmo", f"{domain}_{dir_path.split('/')[-1]}.jsonl.zst")
        assert sample_link.endswith(".jsonl.zst") or sample_link.endswith(".jsonl.gz"), f"Sample link '{sample_link}' does not end with '.jsonl.zst' or '.jsonl.gz'"
        if os.path.exists(save_path):
            print(f"File {save_path} already exists, skipping download.")
        else:
            download_file(sample_link, save_path)
        # debug_zst(save_path)
        cur_size = dir_sizes[dir_path]
        # assert False, cur_size
        if cur_size >= 0:
            if sample_link.endswith(".jsonl.zst"):
                samples = sample_jsonl_zst(save_path, dir_sizes[dir_path], seed=2026)
            elif sample_link.endswith(".jsonl.gz"):
                samples = sample_jsonl_gz(save_path, dir_sizes[dir_path], seed=2026)
            # print(f"Sampled {len(samples)} items from {save_path} for domain '{domain}'")
            all_samples.extend(samples)
    
print(f"Total samples collected: {len(all_samples)}")
random.shuffle(all_samples)
all_samples = all_samples[:n]

with open(os.path.join("/work/hdd/bfgx/seanlyu2/flexolmo", f"all_sampled_{n}.jsonl"), "w") as f:
    for sample in all_samples:
        f.write(json.dumps(sample) + "\n")