import os
# Must be set before toch import
os.environ['HF_HOME'] = "/data/stud/2025-MA-kuschnereit/hf_models/"

from human_eval.data import write_jsonl, read_problems
from vllm import LLM, SamplingParams
import torch
from huggingface_hub import login

os.environ["CUDA_VISIBLE_DEVICES"]="7"
print('__CUDA Device:',torch.cuda.get_device_properties(0))

#MODEL_NAME = "/data/tyler/llms/llama3.1/huggingface/Meta-Llama-3.1-8B-Instruct"
HF_MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_NAME  = "Qwen2.5-Coder-7B-Instruct"

DATASET     = "human-eval"
BASE_DIR    = "/data/stud/2025-MA-kuschnereit/"
GENERATED_SAMPLES_DIR = BASE_DIR+"data/stud/2025-MA-kuschnereit/generated_samples/"

#sampling_params = SamplingParams(temperature=0.2, logprobs=0, max_tokens=1024, top_p=0.7)

# Params for QWEN

"""temperature=0.7, top_p=0.8, repetition_penalty=1.05,"""
"""
Params for Meta-Llama-3.1-8B-Instruct
sampling_params = llm.get_default_sampling_params()
sampling_params.temperature = 0.2
sampling_params.max_tokens = 1024
sampling_params.top_p = 0.7"""
# Needs to be always on 0 to get the chosen token logprob


def create_message(prompt):
    
    return ""


def generate_one_completion(task_id, prompt):
    RequestOutput = llm.generate(prompt, sampling_params)
    output = RequestOutput[0].outputs[0]
    return dict(task_id=task_id,
                token_ids=output.token_ids, 
                logprobs=[{key : [value.logprob, value.rank, value.decoded_token] for key, value in logprobs.items()} for logprobs in output.logprobs],
                cumulative_logprob=output.cumulative_logprob, 
                finish_reason=output.finish_reason), output.text

if __name__ == "__main__":

    llm = LLM(model=HF_MODEL_NAME)
    sampling_params = SamplingParams(max_tokens=512)
    sampling_params.logprobs = 0
    print(sampling_params)

    problems = read_problems()

    num_samples_per_task = 1
    samples = []
    details = []
    #samples = [
    #    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    #    for task_id in problems
    #    for _ in range(num_samples_per_task)
    #]

    for task_id in problems:
        info, completion = generate_one_completion(task_id, problems[task_id]["prompt"])   
        samples.append(dict(task_id=task_id, completion=completion))
        details.append(info)

    write_jsonl(GENERATED_SAMPLES_DIR+DATASET+"/samples-"+MODEL_NAME+".jsonl", samples)
    write_jsonl(GENERATED_SAMPLES_DIR+DATASET+"/details-"+MODEL_NAME+".jsonl", details)