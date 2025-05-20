import os
# Must be set before toch import
os.environ['HF_HOME'] = "/data/stud/2025-MA-kuschnereit/hf_models/"
os.environ["VLLM_USE_V1"]="1"
os.environ["VLLM_CACHE_ROOT"]="/data/stud/2025-MA-kuschnereit/.vllmcache/"

from pathlib import Path
import re

os.environ["CUDA_VISIBLE_DEVICES"]="5"

REPAIR_SET = "defects4j"
MODEL = "Qwen-Qwen2.5-Coder-32B-Instruct"

VARIANT = "test"

TIMESTAMP = "12_05_12"

if __name__ == "__main__":
    # /data/tyler/dok/viola/exchange/robin/braco_generations/raw_patches/bugsphp/
    # change for defects4j because its in the masterarbeit folder
    if REPAIR_SET == 'bugsphp':
        output_dir = '/data/tyler/dok/viola/exchange/robin/braco_generations/raw_patches/'
    elif REPAIR_SET == 'defects4j':
        output_dir = '/data/stud/2025-MA-kuschnereit/masterarbeit/program_repair/patches/'

    prompt_dir  = '/data/stud/2025-MA-kuschnereit/masterarbeit/program_repair/prompts/'
    save_dir    = '/data/stud/2025-MA-kuschnereit/masterarbeit/program_repair/complete/'+REPAIR_SET+'/'+VARIANT+'/'+MODEL+'/'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    prompt_dir = Path(prompt_dir+REPAIR_SET+'/repair/'+REPAIR_SET+'/')

    for prompt_file in prompt_dir.iterdir():

        details = prompt_file.name.split('_')

        category = details[2].rsplit('-', 1)[0]
        name = details[2]
        hint = details[3]
        variant = details[-1].split('.')[0]

        if variant != VARIANT:
            continue

        file_output_dir = output_dir+REPAIR_SET+'/'+category+'/'+name+'/'+MODEL+'/'

        output_files = [f for f  in Path(file_output_dir).glob("*_"+variant+"_"+TIMESTAMP+"_*.output")]       

        if len(output_files) != 1:
            #exit("Found (too many/no) output files!")  
            continue     

        print(prompt_file.name)

        with open(prompt_file, 'r') as f:
            prompt_content = f.read()
        prompt_content = prompt_content[51:len(prompt_content)]

        with open(output_files[0], 'r') as f:
            output_content = f.read()

        print(output_files[0].name)
        
        generations_index = [m.start() for m in re.finditer('~~~~~~~~~~~~~~~~~', output_content)]
        generations_index = [52] + generations_index

        for n, m, num in zip(generations_index[0::1], generations_index[1::1], range(0, len(generations_index)-1)):
            ll_content = prompt_content + output_content[n:m]
            if num != 0:
                all_content = prompt_content + output_content[n+18:m]
            else:
                all_content = prompt_content + output_content[n:m]
            fn = str(num)+'_'+output_files[0].name.replace('.output', '#')+details[2]+'_'+details[3]+'_'+details[4]+'_'+variant+'.prompt'
            with open(save_dir+fn, 'w') as f_save:
                f_save.write(all_content)
        print()