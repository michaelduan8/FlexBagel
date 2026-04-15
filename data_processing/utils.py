import csv
import json
import os
import pickle

from datasets import load_dataset
from tqdm import tqdm

def load_json(file_path):
    return json.load(smart_open.open(file_path, 'r'))

def load_jsonl(file_path):
    with smart_open.open(file_path, 'r') as f_in:
        return [json.loads(line) for line in tqdm(f_in)]

def write(file_name, contents, mode='jsonl', write_mode='w'):
    output_dir = os.path.dirname(file_name)
    os.makedirs(output_dir, exist_ok=True)

    if mode == 'jsonl':
        with open(file_name, write_mode) as f:
            for content in tqdm(contents):
                f.write(f"{json.dumps(content)}\n")
    elif mode == 'json':
        with open(file_name, write_mode) as f:
            json.dump(contents, f, indent=4)
    elif mode == 'csv':
        with open(file_name, write_mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=contents[0].keys())
            writer.writeheader()
            writer.writerows(contents)
    elif mode == 'pkl':
        with open(file_name, write_mode + 'b') as f:
            pickle.dump(contents, f, protocol=pickle.HIGHEST_PROTOCOL)
