import os
# Must be set before toch import
os.environ['HF_HOME'] = "/data/stud/2025-MA-kuschnereit/hf_models/"
os.environ["VLLM_USE_V1"]="1"
os.environ["VLLM_CACHE_ROOT"]="/data/stud/2025-MA-kuschnereit/.vllmcache/"

from vllm import LLM, SamplingParams
import torch
from tools import data
import argparse
import json

os.environ["CUDA_VISIBLE_DEVICES"]="7"

HF_MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

METHOD = "quantitative" # qualitativ

BASE_DIR    = "/data/stud/2025-MA-kuschnereit/masterarbeit/verbalized_data/"
GENERATED_DIR = BASE_DIR+METHOD+"/"


# FROM https://github.com/parameterlab/apricot/blob/main/src/prompts.py
# https://github.com/parameterlab/apricot/blob/main/src/constants.py

QUALITATIVE_SCALE = {
    "Very low": 0,
    "Low": 0.15,
    "Somewhat low": 0.3,
    "Medium": 0.45,
    "Somewhat high": 0.6,
    "High": 0.75,
    "Very high": 0.9,
}

TEMPLATE = "Answer as short as you can. \nHere is a snippet of generated code consisting of the prompt, generated code and test cases: \n {code}"

QUANT_VERBALIZED_CONFIDENCE_PROMPT = (
    "\nPlease provide your confidence in the correctness of the provided code only in percent (0-100 %): "
)
QUAL_VERBALIZED_CONFIDENCE_PROMPT = f"\nPlease provide your confidence in the correctness of the provided code only as one of {' / '.join([f'{exp}' for exp in QUALITATIVE_SCALE.keys()])}: "

all_lang = True

batch_size = 20

if __name__ == "__main__":
    dtype = "float16"
    if torch.cuda.is_bf16_supported():
        dtype = "bfloat16"

    llm = LLM(
            model=HF_MODEL_NAME,
            dtype=dtype,
            max_model_len=4096,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_num_batched_tokens=4096 ,
            enable_prefix_caching=True,
            enforce_eager=True
        )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dirs", type=str,  help="Directories with results. ", nargs="+")
    args = parser.parse_args()

    run_dirs = [x[0] for x in os.walk(args.dirs[0])]
    run_dirs.sort()

    if all_lang == True:
        run_dirs = [run_dirs[0]]

    for d in run_dirs:
        # if main dir is in list just continue
        if d == args.dirs[0] and ("humaneval" not in d and "mbpp" not in d):
            continue
        
        # Get run name
        run = d.split("/runs/", 1)[1]

        # load the data from the run directory
        run = run.replace('/', '')
        results, temperature, top_p, num_samples = data.load_multipl_e_run(d)

        save_dir = GENERATED_DIR+run+'/' 

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
               
        # probs -> confidence of the model
        # is_correct -> label 1: is correct, 0: is not correct
        _, _, programs, _, languages, names = data.load_samples(results)

        verbalized_data = {k: [] for k in languages}

        params = SamplingParams(temperature=temperature,
                                top_p=top_p, max_tokens=1024) 
        params.logprobs    = 0

        program_list = [programs[i:i+batch_size] for i in range(0, len(programs), batch_size)]
        languages_list = [languages[i:i+batch_size] for i in range(0, len(languages), batch_size)]
        names_list = [names[i:i+batch_size] for i in range(0, len(names), batch_size)]

        i = 0

        for programs, langs, names in zip(program_list, languages_list, names_list):
            prompts = []

            for p in programs:
                prompt = TEMPLATE.format(code=p)

                if METHOD == 'quantitative':
                    prompt += QUANT_VERBALIZED_CONFIDENCE_PROMPT
                if METHOD == 'qualitativ':
                    prompt += QUAL_VERBALIZED_CONFIDENCE_PROMPT
                prompts.append(prompt)


            outputs = llm.generate(prompts, params)
            outputs = [o.outputs[0] for o in outputs]

            for out, name, lang in zip(outputs, names, langs):
                d = {}
                verbalized_data[lang][name]= dict(  text=out.text,
                                                    token_ids=out.token_ids, 
                                                    logprobs=[{key : [value.logprob, value.rank, value.decoded_token] for key, value in logprobs.items()} for logprobs in out.logprobs],
                                                    cumulative_logprob=out.cumulative_logprob, 
                                                    finish_reason=out.finish_reason)

                
            print(f"Batch number {i}")
            i=i+1

        with open(save_dir+'verbalized_data_'+METHOD+'.json', 'w') as fp:
            json.dump(verbalized_data, fp)
