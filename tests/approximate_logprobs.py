import os
# Must be set before toch import
os.environ['HF_HOME'] = "/data/stud/2025-MA-kuschnereit/hf_models/"
import numpy as np
from human_eval.data import write_jsonl, read_problems
from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams
import torch
from huggingface_hub import login


def prob_sample_to_recover(prob_sample, temp, top_p=1, hidden_vocab_num=None):
    prob_sample = prob_sample * top_p
    top_logprobs = len(prob_sample)

    if hidden_vocab_num is None:
        hidden_vocab_num = 1 if (1 - prob_sample.sum() > 0.001) else 0

    if hidden_vocab_num > 0:
        hidden_probs = [max(1 - prob_sample.sum(), 0) / hidden_vocab_num] * hidden_vocab_num
        prob_sample = np.concatenate([prob_sample, hidden_probs])

    prob_recover_unnorm = prob_sample ** temp
    prob_recover = prob_recover_unnorm / prob_recover_unnorm.sum()

    return prob_recover[:top_logprobs]

# Test case

"""# Sampling parameters
temp = 0.5
top_p = 0.9
top_logprobs = 3

# Original logits
logits = np.array([1, 0, -1, -2,])

# Compute prob_sample using softmax with temperature temp
logits_temp = logits / temp
exp_logits_temp = np.exp(logits_temp)
prob_sample = exp_logits_temp / exp_logits_temp.sum()



prob_sample /= top_p
prob_sample = prob_sample[:top_logprobs]  # Same to vllm's e**logprob
print(prob_sample)
prob_recover = prob_sample_to_recover(prob_sample, temp, top_p)

print(" prob_sample:", prob_sample)
print("prob_recover:", prob_recover)

# Ground truth probabilities
prob_gt = np.exp(logits) / np.exp(logits).sum()
print("     prob_gt:", prob_gt)
"""




os.environ["CUDA_VISIBLE_DEVICES"]="6"
os.environ["VLLM_USE_V1"]="1"
print('__CUDA Device:',torch.cuda.get_device_properties(0))

#HF_MODEL_NAME = "/data/tyler/llms/llama3.3/huggingface/Meta-Llama-3.3-70B-Instruct/"
HF_MODEL_NAME   = "Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_NAME      = "Qwen2.5-Coder-7B-Instruct"
PARAMS          = ""

DATASET     = "human-eval"
BASE_DIR    = "/data/stud/2025-MA-kuschnereit/masterarbeit/"
GENERATED_SAMPLES_DIR = BASE_DIR+"generated_samples/"


if __name__ == "__main__":
    llm = LLM(model=HF_MODEL_NAME, max_model_len=2048)

    sampling_params_temp_0 = SamplingParams(logprobs=1,
                                            max_tokens=512)
    sampling_params_temp_0.top_p = 0.8
    sampling_params_temp_0.temperature = 1

    print(sampling_params_temp_0)

    prompt = ("The Capital from Germany is ")

    RequestOutput_temp_0 = llm.generate(prompt, sampling_params_temp_0)
    output_temp_0 = RequestOutput_temp_0[0].outputs[0]

    print([{key : [value.logprob, value.rank, value.decoded_token] for key, value in logprobs.items()} for logprobs in output_temp_0.logprobs][:1])


    sampling_params_temp_1 = SamplingParams(logprobs=1, 
                                            max_tokens=512)

    #print(sampling_params_temp_1)

    RequestOutput = llm.generate(prompt, sampling_params_temp_1)
    output = RequestOutput[0].outputs[0]
    print([{key : [value.logprob, value.rank, value.decoded_token] for key, value in logprobs.items()} for logprobs in output.logprobs][:1])