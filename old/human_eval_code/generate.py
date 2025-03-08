import os
# Must be set before toch import
os.environ['HF_HOME'] = "/data/stud/2025-MA-kuschnereit/hf_models/"

from human_eval.data import write_jsonl, read_problems
from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams
import torch
from huggingface_hub import login

os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
print('__CUDA Device:',torch.cuda.get_device_properties(0))
print('__CUDA Device:',torch.cuda.get_device_properties(1))
print('__CUDA Device:',torch.cuda.get_device_properties(2))
print('__CUDA Device:',torch.cuda.get_device_properties(3))

HF_MODEL_NAME = "/data/tyler/llms/llama3.3/huggingface/Meta-Llama-3.3-70B-Instruct/"
#HF_MODEL_NAME   = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
MODEL_NAME      = "Meta-Llama-3.3-70B-Instruct"
PARAMS          = ""

DATASET     = "human-eval"
BASE_DIR    = "/data/stud/2025-MA-kuschnereit/masterarbeit/"
GENERATED_SAMPLES_DIR = BASE_DIR+"generated_samples/"


def generate_one_completion(task_id, prompt):
    RequestOutput = llm.generate(prompt, sampling_params)
    output = RequestOutput[0].outputs[0]
    return dict(task_id=task_id,
                token_ids=output.token_ids, 
                logprobs=[{key : [value.logprob, value.rank, value.decoded_token] for key, value in logprobs.items()} for logprobs in output.logprobs],
                cumulative_logprob=output.cumulative_logprob, 
                finish_reason=output.finish_reason), output.text

if __name__ == "__main__":
    llm = LLM(model=HF_MODEL_NAME, tensor_parallel_size=4, max_model_len=2048)

    sampling_params = SamplingParams(max_tokens=512)
    sampling_params.logprobs    = 0
    #sampling_params.temperature = 0
    #params = BeamSearchParams(beam_width=3, max_tokens=50)

    print(sampling_params)

    problems = read_problems()

    samples = []
    details = []

    for task_id in problems:
        info, completion = generate_one_completion(task_id, problems[task_id]["prompt"])   
        samples.append(dict(task_id=task_id, completion=completion))
        details.append(info)
        print(f"{len(samples)}/{len(problems)} Processed")

    write_jsonl(GENERATED_SAMPLES_DIR+DATASET+"/"+MODEL_NAME+"/samples-"+MODEL_NAME+PARAMS+".jsonl", samples)
    write_jsonl(GENERATED_SAMPLES_DIR+DATASET+"/"+MODEL_NAME+"/details-"+MODEL_NAME+PARAMS+".jsonl", details)