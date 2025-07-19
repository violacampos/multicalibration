import json

def read_jsonl(path):
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]

    return data


def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data