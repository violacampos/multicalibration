import os
# Must be set before toch import
os.environ['HF_HOME'] = "/data/stud/2025-MA-kuschnereit/hf_models/"
os.environ["VLLM_USE_V1"]="1"
os.environ["VLLM_CACHE_ROOT"]="/data/stud/2025-MA-kuschnereit/.vllmcache/"


from vllm import LLM, SamplingParams
import torch
from pathlib import Path 
import json

os.environ["CUDA_VISIBLE_DEVICES"]="4,5"
print('__CUDA Device:',torch.cuda.get_device_properties(0))

HF_MODEL_NAME   = "Qwen/Qwen2.5-Coder-32B-Instruct"

REPAIR_SET = "defects4j"#"bugsphp"
MODEL = "Qwen-Qwen2.5-Coder-32B-Instruct"

VARIANT = "test"
BASE_DIR    = "/data/stud/2025-MA-kuschnereit/masterarbeit/"

complete_prompt_dir = BASE_DIR+'/program_repair/complete/'+REPAIR_SET+'/'+VARIANT+'/'+MODEL+'/'
only_prompt_dir = BASE_DIR+'/program_repair/prompts/'+REPAIR_SET+'/repair/'+REPAIR_SET+'/'

eval_dir = '/data/tyler/dok/viola/exchange/robin/evaluation/'+REPAIR_SET+'/'

logprob_data_dir = BASE_DIR+'/program_repair/runs/'+REPAIR_SET+'/'+VARIANT+'/'+MODEL+'/'
if not os.path.isdir(logprob_data_dir):
    os.makedirs(logprob_data_dir)

skipped = 0   
if __name__ == "__main__":

    llm = LLM(  model=HF_MODEL_NAME,
                tensor_parallel_size=2,
                gpu_memory_utilization=0.70,
                max_model_len=4096,
                max_num_batched_tokens=4096,
                enable_prefix_caching=False,
                enforce_eager=True)
    
    params = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=1) 
    params.prompt_logprobs = 0
    #params.logprobs = 0

    prompt_files = [f for f  in Path(complete_prompt_dir).glob("*_"+VARIANT+"_*.prompt")]

    data = {}

    for prompt_file in prompt_files:

        with open(prompt_file, 'r') as f:
            prompt_content = f.read()
        
        prompt_file_name = prompt_file.name.split('#')
        print(prompt_file_name)
        details = prompt_file_name[1].split('_')
        category = details[0].rsplit('-', 1)[0]
        name = details[0]

        eval_file_dir = eval_dir+category+'/'+name+'/all_validation/valkyrie/output/'
        eval_file_name = MODEL+'-'+prompt_file_name[0].rsplit('_', 1)[0]
     
        is_correct = 42
        for result in ['incorrect', 'invalid', 'plausible']:
            eval_files = [f for f  in Path(eval_file_dir+result+'/').glob(eval_file_name+'*.diff')]

            if len(eval_files) == 0:
                continue
            else:
                if result in ['plausible']:
                    is_correct = 1
                    print("r")
                elif result in ['incorrect', 'invalid']:
                    is_correct = 0
                    print("f")

        only_prompt_files = [f for f  in Path(only_prompt_dir).glob('*'+prompt_file_name[1])]
        
        if is_correct == 42:
            skipped += 1
            continue
        if len(only_prompt_files) != 1:
            skipped += 1
            continue  

        with open(only_prompt_files[0], 'r') as f:
            only_prompt = f.read() 
            only_prompt = only_prompt[51:len(only_prompt)]

        prompt_token_ids = llm.get_tokenizer().encode(only_prompt)

        tokens_content_len = len(llm.get_tokenizer().encode(prompt_content))

        if tokens_content_len >= 4096:
            skipped += 1
            continue
        generation = llm.generate(prompt_content, params)

        complete_token_ids = [o.prompt_token_ids for o in generation][0]
        complete_logprobs = [o.prompt_logprobs for o in generation][0]

        # remove prompt from generation
        complete_token_ids = complete_token_ids[len(prompt_token_ids):]
        complete_logprobs = complete_logprobs[len(prompt_token_ids):]

        name = prompt_file.name.replace('.prompt', '')

        cumulative_logprob = 0
        for logprob in complete_logprobs:
            cumulative_logprob += list(logprob.values())[0].logprob

        data[name] = {
            "prompt": only_prompt,
            "program": prompt_content,
            "token_logprobs": [{key : [value.logprob, value.rank, value.decoded_token] for key, value in logprobs.items()} for logprobs in complete_logprobs],
            "token_ids": complete_token_ids,
            "is_correct": is_correct,
            "cumulative_logprob": cumulative_logprob,
            "token_count": len(complete_token_ids)
        }

    with open(logprob_data_dir+'data.json', 'w') as fp:
        json.dump(data, fp)
    print(f"Skipped {skipped} prompts")
