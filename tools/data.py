import json
from pathlib import Path
import gzip
from typing import Optional
import itertools
import numpy as np
import os
import datetime
import re
import configparser

def gunzip_json(path: Path) -> Optional[dict]:
    """
    Reads a .json.gz file, but produces None if any error occurs.
    """
    try:
        with gzip.open(path, "rt") as f:
            return json.load(f)
    except Exception as e:
        return None

def gzip_json(path: Path, data: dict) -> None:
    with gzip.open(path, "wt") as f:
        json.dump(data, f)


def for_file(path: Path):
    if path.suffix == ".gz":
        data = gunzip_json(path)
    else:
        with open(path, 'r') as f:
            data = json.load(f)

    if data is None:
        return None
    
    return_values = []
    n = len(data["results"])
    for d, res in zip(data["tokens_info"], data["results"]):
        token_infos         = d
        cumulative_logprob  = token_infos["cumulative_logprob"]
        token_logprobs      = token_infos["token_logprobs"]
        token_ids           = token_infos["len"]
        c = 1 if res["status"] == "OK" and res["exit_code"] == 0 else 0
        
        res_dict = {
            "name": data["name"], 
            "language": data["language"],
            "prompt": data["prompt"],
            "program": res["program"],
            "n": n,
            "c": c,
            "temperature": data["temperature"] if "temperature" in data else 0.2,
            "top_p": data["top_p"],
            "cumulative_logprob": cumulative_logprob,
            "token_ids": token_ids,
            "token_logprobs": token_logprobs
        }

        return_values.append(res_dict)

    return return_values

# StackOverflow https://stackoverflow.com/questions/27789665/check-if-a-given-directory-contains-any-directory-in-python
def folders_in(path_to_parent):
    for fname in os.listdir(path_to_parent):
        if os.path.isdir(os.path.join(path_to_parent,fname)):
            yield os.path.join(path_to_parent,fname)


def load_multipl_e_run(path):
    subfolders = list(folders_in(path))

    if not subfolders:
        # load the data from the run directory
        results = [for_file(p) for p in itertools.chain(
                    Path(path).glob("*.results.json"), Path(path).glob("*.results.json.gz"))]
        results = [r for r in results if r is not None]

        temperature = list(set(r[0]["temperature"] for r in results))[0]
        top_p = list(set(r[0]["top_p"] for r in results))[0]
        n = list(set(r[0]["n"] for r in results))[0]

        num_samples = len(results) * n
    else:
        results = []
        for folder in subfolders:
            results_folder = [for_file(p) for p in itertools.chain(
                                Path(folder).glob("*.results.json"), Path(folder).glob("*.results.json.gz"))]
            results.extend([r for r in results_folder if r is not None])

        temperature = list(set(r[0]["temperature"] for r in results))[0]
        top_p = list(set(r[0]["top_p"] for r in results))[0]
        n = list(set(r[0]["n"] for r in results))[0]

        num_samples = len(results) * n

    return results, temperature, top_p, num_samples

def load_json_data(path):
    with open(path, 'r') as file:
        json_data = json.load(file)
    return json_data

#  Teile aus https://github.com/parameterlab/apricot/blob/main/src/eval.py
def proability_and_correctness_for_samples(results, verb_data, type="avg_logprob"):
    prob_value_list = []
    is_correct = []
    prompts = []
    programms = []
    languages = []
    names = []
    successful = []
    token_logprobs = []

    QUALITATIVE_SCALE = {
        "Very low": 0,
        "Low": 0.3,
        "Somewhat low": 0.45,
        "Medium": 0.5,
        "Somewhat high": 0.65,
        "High": 0.7,
        "Very high": 1,
    }

    # Get the token probailities from the samples and create arrays
    for r in results:
        for sample in r:
            token_count = len(sample["token_ids"])
            cumulative_logprob = sample["cumulative_logprob"]
            
            if sample["language"] == 'elixir':
                lang = "ex"
            elif sample["language"] == 'go_test.go':
                lang = "go"
            else:
                lang = sample["language"]

            if type == "avg_logprob":
                prob = np.round(np.exp(cumulative_logprob / token_count), 2)
            elif type == "quantitativ":
                try:
                    template = r"\d{1,3}(?:\.\d+)?\s?\%?"
                    d = verb_data[lang][sample["name"]]
                    #d = next((item for item in verb_data[lang] if item["task_id"] == sample["name"]), None)
                    res = re.search(template, d["completion"]).group(0)
                    prob = float(res.replace("%", "")) / 100
                    if not (0 <= prob <= 1):
                        successful.append(False)
                        continue
                except AttributeError:
                    successful.append(False)
                successful.append(True)
            elif type == "qualitativ":
                try:
                    template = rf"({'|'.join(QUALITATIVE_SCALE.keys())})"
                    d = verb_data[lang][sample["name"]]
                    #d = next((item for item in verb_data[lang] if item["task_id"] == sample["name"]), None)
                    res = re.search(template, d["completion"]).group(0)
                    prob = QUALITATIVE_SCALE[res]

                except AttributeError:
                    successful.append(False)
                successful.append(True)
                                     
            # collect average token probabilty and correctnes value
            programms.append(sample["program"])
            languages.append(lang)

            names.append(sample["name"])
            prompts.append(sample["prompt"])
            prob_value_list.append(prob)
            is_correct.append(1) if sample["c"] == 1 else is_correct.append(0)
            token_logprobs.append(sample["token_logprobs"])
    if type in ["qualitativ", "quantitativ"]:
        print(f"Succesful extracted: {len(successful)}")
    prob_value_list     = np.array(prob_value_list)
    is_correct          = np.array(is_correct)
    languages           = np.array(languages)

    return prob_value_list, is_correct, programms, prompts, languages, names, token_logprobs

def avg_token_probability(cumulative_logprob, token_count):
    return np.round(np.exp(cumulative_logprob / token_count), 2) 


def load_scc_data(run, languages, names):
    scc_infos = []
    with open('./scc/'+run+'.json') as scc:
        scc_data = json.load(scc)
        file_list = []
        comp = []
        for i in range(len(scc_data)):
            file_list.extend(scc_data[i]["Files"])

        for lang, name in zip(languages, names):
            temp = {}
            t = list(filter(lambda sample: (sample['Filename'].replace('.'+sample['Extension'], '') == name) and (sample['Extension'] == lang), file_list))
            temp["Bytes"] = t[0]["Bytes"]
            temp["Lines"] = t[0]["Lines"]
            temp["Code"] = t[0]["Code"]
            temp["Comment"] = t[0]["Comment"]
            temp["Blank"] = t[0]["Blank"]
            temp["Complexity"] = t[0]["Complexity"]
            comp.append(t[0]["Complexity"])
            scc_infos.append(temp)   
        """print(np.histogram(np.array([d["Blank"] for d in scc_infos]), bins=[0,5,10,20,30,40,100,500]))
        print(np.histogram(np.array([d["Comment"] for d in scc_infos]), bins=[0,5,10,20,30,40,100,500]))
        print(np.histogram(np.array([d["Code"] for d in scc_infos]), bins=[0,5,10,20,30,40,100,500]))
        print(np.mean(np.array([d["Code"] for d in scc_infos])))
        print(np.median(np.array([d["Code"] for d in scc_infos])))
        print(np.histogram(np.array([d["Lines"] for d in scc_infos]), bins=[0,5,10,20,30,40,100,500]))"""
    return scc_infos 

def generate_save_dir(run, method, binning, prob_generation, split, model, calibration_data=False, history_data=False):
    """
    Generate a directory with the following structure for different runs and methods
    (dir) runs
        (dir) *run_name*
            (dir) *calibration_method*
                (dir) *binning_method*
                    (dir) *prob_generation_method*
                        (file) timestamp file for traceability
                        (file) comparison_bar_chart
                        (file) group_calibration
                        (file) scores (txt/json)
                        (dir) *model* (only for qualitativ/qunatitativ prob_generation)
                            (file) timestamp file for traceability
                            (file) comparison_bar_chart
                            (file) group_calibration
                            (file) scores (txt/json)

    """

    config = load_config()

    dir = config["Paths"]["base_dir"]+"runs/"+run+"/"+method+"/"+binning+"/"+prob_generation+"/"

    if split:
        dir += "split/"
    else:
        dir += "all/"

    if model is not None:
        dir += model+"/"

    if not os.path.isdir(dir):
        os.makedirs(dir)
    
    if calibration_data:
        if not os.path.isdir(dir+'calibration_data/'):
            os.makedirs(dir+'calibration_data/')
    
    if history_data:
        if not os.path.isdir(dir+'history_data/'):
            os.makedirs(dir+'history_data/')

    content = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    with open(f"{dir}_created_at.txt", "w") as f:
        f.write(content)
    
    return dir

def load_config():
    config = configparser.ConfigParser()
    try:
        file = open("config.ini", "r")
    except FileNotFoundError:
        print("Can't find config.ini!")
        exit()

    config.read_file(file)

    if not config.has_option("Paths", "base_dir"):
        print("Can't find base_dir in the Paths section of config.ini!")
        exit()

    return config


