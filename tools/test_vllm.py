import os
from vllm import LLM, SamplingParams
import torch

os.environ["CUDA_VISIBLE_DEVICES"]="7"

#print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
#print('__CUDA Device Name:',torch.cuda.get_device_name(0))
print('__CUDA Device:',torch.cuda.get_device_properties(0))
print('__CUDA Device:',torch.cuda.get_device_properties(1))

prompt = "The capital of France is"



llm = LLM(model="/data/tyler/llms/llama3.1/huggingface/Meta-Llama-3.1-70B-Instruct")
sampling_params = llm.get_default_sampling_params()
output = llm.generate(prompt, sampling_param)

print(f"Prompt: {prompt!r}, Generated text: {output!r}")
