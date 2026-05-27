import json

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
        new_row["id"] = f"quilt_{i}#assistant_{idx}"
        new_row["conversation"] = conversation[: idx + 1]

        new_rows.append(new_row)

    return new_rows


input_ = "/mnt/quilt/quilt_instruct_w_length_w_path.jsonl"
output_ = "/mnt/quilt/quilt_instruct_w_length_w_path_flatten.jsonl"

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