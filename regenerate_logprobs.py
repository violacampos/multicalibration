import os
# Must be set before toch import
os.environ['HF_HOME'] = "/data/stud/2025-MA-kuschnereit/hf_models/"

from tools.json_utils import read_json, read_jsonl
from human_eval.data import write_jsonl, read_problems
from vllm import LLM, SamplingParams
import torch
from huggingface_hub import login
from vllm.inputs import TokensPrompt

os.environ["CUDA_VISIBLE_DEVICES"]="7"
print('__CUDA Device:',torch.cuda.get_device_properties(0))

HF_MODEL_NAME   = "/data/tyler/llms/llama3.1/huggingface/Meta-Llama-3.1-8B-Instruct/"
MODEL_NAME      = "Llama-3.1-8B-Instruct"
PARAMS          = ""

DATASET     = "human-eval"
BASE_DIR    = "/data/stud/2025-MA-kuschnereit/masterarbeit/"
GENERATED_SAMPLES_DIR = BASE_DIR+"generated_samples/"

if __name__ == "__main__":

    llm = LLM(model=HF_MODEL_NAME)

    sp = SamplingParams()
    sp.max_tokens = 1
    sp.prompt_logprobs = 0

    details_file_path        = GENERATED_SAMPLES_DIR+DATASET+"/"+MODEL_NAME+"/details-"+MODEL_NAME+PARAMS+".jsonl"
    samples_file_path        = GENERATED_SAMPLES_DIR+DATASET+"/"+MODEL_NAME+"/samples-"+MODEL_NAME+PARAMS+".jsonl"
    details_data    = read_jsonl(details_file_path)
    sample_data     = read_jsonl(samples_file_path)

    print(f"Logit Data loaded from: {details_file_path}")

    for sample in sample_data:
        print(sample["task_id"])
        
        token_ids = next(filter(lambda a : a['task_id'] == sample["task_id"], details_data), None)['token_ids']
        tp = TokensPrompt(prompt_token_ids=token_ids)
        RequestOutput = llm.generate(tp, sp, use_tqdm=False)
        generated_logprobs = next(filter(lambda a : a['task_id'] == sample["task_id"], details_data), None)['logprobs']
        print(generated_logprobs[0])
        print(RequestOutput[0].prompt_logprobs[0])
        break
