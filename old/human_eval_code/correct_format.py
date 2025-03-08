from human_eval.data import write_jsonl
import json

def read_jsonl(path):
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]

    return data

if __name__ == "__main__":
    data = read_jsonl('/data/stud/2025-MA-kuschnereit/generated_samples/human_eval/human-eval-samples-Qwen2.5-Coder-7B-Instruct-No-Param.jsonl')
    num_samples_per_task = 1
    samples = [
        dict(task_id=d["task_id"], completion=d["completion"]["text"])
        for d in data
        for _ in range(num_samples_per_task)
    ]
    write_jsonl("/data/stud/2025-MA-kuschnereit/generated_samples/human_eval/human-eval-samples-Qwen2.5-Coder-7B-Instruct-correct-No-Param.jsonl", samples)