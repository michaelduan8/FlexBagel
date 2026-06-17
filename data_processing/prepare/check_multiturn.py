import json

multiturn_count = []
with open("/flare/MatSciAI/xinxil/data/pathgen/train.jsonl", 'r') as f:
    for line in f:
        data = json.loads(line)
        multiturn_count.append(len(data["conversation"])>4)

print(sum(multiturn_count)/len(multiturn_count))