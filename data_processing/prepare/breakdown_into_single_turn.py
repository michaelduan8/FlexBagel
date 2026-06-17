import json
import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Breakdown a conversation into single turns."
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def explode_conversation(row, i):
    conversation = row["conversation"]
    new_rows = []

    for idx, turn in enumerate(conversation):
        if turn["role"] != "assistant":
            continue

        # Skip malformed case where assistant is the first message.
        if idx == 0:
            continue

        new_row = dict(row)
        new_row["id"] = f"{row['id'] if 'id' in row else ''}_#{i}#assistant_{idx}"
        new_row["conversation"] = conversation[: idx + 1]

        new_rows.append(new_row)

    return new_rows


args = parse_args()
input_ = args.input
output_ = args.output if args.output is not None else input_.replace(".jsonl", "_flatten.jsonl")

n_in = 0
n_out = 0

with open(input_, "r", encoding="utf-8") as fin, \
     open(output_, "w", encoding="utf-8") as fout:

    for i, line in enumerate(fin):
        if not line.strip():
            continue

        row = json.loads(line)
        n_in += 1

        exploded_rows = explode_conversation(row, i)

        for new_row in exploded_rows:
            fout.write(json.dumps(new_row, ensure_ascii=False) + "\n")
            n_out += 1

print(f"Input rows:  {n_in}")
print(f"Output rows: {n_out}")