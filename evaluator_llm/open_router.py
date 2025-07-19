from openai import OpenAI
import os
import argparse
import json
from tools import data
import requests
import json

model = "gpt_4o_mini"

METHOD = "qualitativ" #  quantitative

BASE_DIR    = "/data/stud/2025-MA-kuschnereit/masterarbeit/verbalized_data/"
GENERATED_DIR = BASE_DIR+METHOD+"/"

API_KEY = ""

# FROM https://github.com/parameterlab/apricot/blob/main/src/prompts.py
# https://github.com/parameterlab/apricot/blob/main/src/constants.py

QUALITATIVE_SCALE = {
    "Very low": 0,
    "Low": 0.3,
    "Somewhat low": 0.45,
    "Medium": 0.5,
    "Somewhat high": 0.65,
    "High": 0.7,
    "Very high": 1,
}

TEMPLATE = "Answer as short as you can. \nHere is a snippet of generated {language} code consisting of the prompt, generated code and test cases: \n {code}"

QUANT_VERBALIZED_CONFIDENCE_PROMPT = (
    "\nPlease provide your confidence in the correctness of the provided code only in percent (0-100 %): "
)
QUAL_VERBALIZED_CONFIDENCE_PROMPT = f"\nPlease provide your confidence in the correctness of the provided code only as one of {' / '.join([f'{exp}' for exp in QUALITATIVE_SCALE.keys()])}: "

all_lang = True

if __name__ == "__main__":

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

        save_dir = GENERATED_DIR+run+'/'+model+'/' 

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
               
        # probs -> confidence of the model
        # is_correct -> label 1: is correct, 0: is not correct
        _, _, programs, _, languages, names, _ = data.load_samples(results, None)

        file_name = save_dir+'data.json'
        if os.path.isfile(file_name):
            with open(file_name, 'r') as fp:
                file_contents = json.load(fp)
        else: 
            file_contents = None  

        if file_contents is None:
            verbalized_data = {k: {} for k in languages}
        else:
            verbalized_data = file_contents      

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=API_KEY,
        )

        i = 0
        for program, lang, name in zip(programs, languages, names):
            if name in verbalized_data[lang]:
                continue

            prompt = TEMPLATE.format(code=program, language=lang)

            if METHOD == 'quantitative':
                prompt += QUANT_VERBALIZED_CONFIDENCE_PROMPT
            if METHOD == 'qualitativ':
                prompt += QUAL_VERBALIZED_CONFIDENCE_PROMPT

            completion = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                max_completion_tokens=100,
                store=False,
                messages=[
                {
                    "role": "user",
                    "content": prompt
                }
                ]
            )
            
            if hasattr(completion, 'error'):
                with open(file_name, 'w') as fp:
                    json.dump(verbalized_data, fp)
                print(f"Error code: {completion.error['code']}")
                print(f"Message: {completion.error['message']}")
                print(f"Metadata: {completion.error['metadata']}")
                exit()

            verbalized_data[lang][name] = dict( prompt=prompt,
                                                completion=completion.choices[0].message.content,
                                                finish_reason=completion.choices[0].finish_reason,
                                                prompt_tokens=completion.usage.prompt_tokens,
                                                completion_tokens=completion.usage.completion_tokens,
                                                total_tokens=completion.usage.total_tokens)

            
            if i % 100 == 0:
                with open(file_name, 'w') as fp:
                    json.dump(verbalized_data, fp)
                response = requests.get(
                    url="https://openrouter.ai/api/v1/auth/key",
                    headers={
                        "Authorization": f"Bearer {API_KEY}"
                    }
                )

                print(f'Used Credits: {json.dumps(response.json()["data"]["usage"])}')
                print(f'Remaining: {json.dumps(response.json()["data"]["limit_remaining"])}')
            i = i + 1
            print(f"Processed: {i}")
            if i == 1001:
                break
        
        with open(file_name, 'w') as fp:
                    json.dump(verbalized_data, fp)
