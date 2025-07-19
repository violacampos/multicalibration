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
from tools.groups import groups
import pandas as pd

class data_loader:

    def __init__(self, args, run_dir, extern, method):
        self.run = None
        self.save_dir = None
        self.num_samples = None
        self.data = None

        self.setup_data(args, run_dir, extern, method)
        self.check_data_loaded()


    def gunzip_json(self, path: Path) -> Optional[dict]:
        """
        Reads a .json.gz file, but produces None if any error occurs.
        """
        try:
            with gzip.open(path, "rt") as f:
                return json.load(f)
        except Exception as e:
            return None

    def gzip_json(self, path: Path, data: dict) -> None:
        with gzip.open(path, "wt") as f:
            json.dump(data, f)


    def for_file(self, path: Path):
        if path.suffix == ".gz":
            data = self.gunzip_json(path)
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
    def folders_in(self, path_to_parent):
        for fname in os.listdir(path_to_parent):
            if os.path.isdir(os.path.join(path_to_parent,fname)):
                yield os.path.join(path_to_parent,fname)


    def load_multipl_e_run(self, path):
        subfolders = list(self.folders_in(path))

        if not subfolders:
            # load the data from the run directory
            results = [self.for_file(p) for p in itertools.chain(
                        Path(path).glob("*.results.json"), Path(path).glob("*.results.json.gz"))]
            results = [r for r in results if r is not None]

            temperature = list(set(r[0]["temperature"] for r in results))[0]
            top_p = list(set(r[0]["top_p"] for r in results))[0]
            n = list(set(r[0]["n"] for r in results))[0]

            num_samples = len(results) * n
        else:
            results = []
            for folder in subfolders:
                results_folder = [self.for_file(p) for p in itertools.chain(
                                    Path(folder).glob("*.results.json"), Path(folder).glob("*.results.json.gz"))]
                results.extend([r for r in results_folder if r is not None])

            temperature = list(set(r[0]["temperature"] for r in results))[0]
            top_p = list(set(r[0]["top_p"] for r in results))[0]
            n = list(set(r[0]["n"] for r in results))[0]

            num_samples = len(results) * n

        return results, temperature, top_p, num_samples

    def load_json_data(self, path):
        with open(path, 'r') as file:
            json_data = json.load(file)
        return json_data

    #  Teile aus https://github.com/parameterlab/apricot/blob/main/src/eval.py
    def proability_and_correctness_for_samples(self, results, verb_data, type="avg_logprob"):
        prob_value_list = []
        is_correct = []
        prompts = []
        programms = []
        languages = []
        names = []
        successful = []
        token_logprobs = []

        """QUALITATIVE_SCALE = {
            "Very low": 0,
            "Low": 0.3,
            "Somewhat low": 0.45,
            "Medium": 0.5,
            "Somewhat high": 0.65,
            "High": 0.7,
            "Very high": 1,
        }"""
        QUALITATIVE_SCALE = {
            "Very low": 0,
            "Low": 0.15,
            "Somewhat low": 0.3,
            "Medium": 0.45,
            "Somewhat high": 0.60,
            "High": 0.75,
            "Very high": 0.9,
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

    def avg_token_probability(self, cumulative_logprob, token_count):
        return np.round(np.exp(cumulative_logprob / token_count), 2) 


    def load_scc_data(self, run, languages, names):
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
        return scc_infos 

    def generate_save_dir(self, run, method, binning, prob_generation, grouping_style, split, model, calibration_data=False, history_data=False):
        """
        Generate a directory with the following structure for different runs and methods
        (dir) runs
            (dir) *run_name*
                (dir) *calibration_method*
                    (dir) *binning_method*
                        (dir) *prob_generation_method*
                            (dir) *grouping_style*
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

        config = self.load_config()

        dir = config["Paths"]["base_dir"]+"runs/"+run+"/"+method+"/"+binning+"/"+prob_generation+"/"+grouping_style+"/"

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

    def load_config(self):
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


    def load_program_repair_data(self, path):
        with open(path+'data.json', 'r') as f:
            data = json.load(f)

        names = list(data.keys())  
        probs = []
        is_correct = []
        programs = []
        prompts = []
        languages = []
        token_logprobs = []


        for d in data.values():
            prob = np.round(np.exp(d["cumulative_logprob"] / d["token_count"]), 2)
            probs.append(prob)
            is_correct.append(d["is_correct"])
            programs.append(d["program"])
            prompts.append(d["prompt"])
            if "bugsphp" in path:
                languages.append("php")
            elif "defects4j" in path:
                languages.append("java")
            token_logprobs.append(d["token_logprobs"])
        
        probs     = np.array(probs)
        is_correct          = np.array(is_correct)
        languages           = np.array(languages)

        return probs, is_correct, programs, prompts, languages, names, token_logprobs

    def setup_data(self, args, run_dir, extern, method):
        base_dir = self.load_config()["Paths"]["base_dir"]

        if args.problem == "code-gen":    
            # load the data from the run directory
            results, temperature, top_p, num_samples = self.load_multipl_e_run(run_dir)
            run = run_dir.split("/runs/", 1)[1]
            run = run.replace('/', '')
        elif args.problem == "program-repair":
            run = run_dir.split("/runs/", 1)[1]  
            temperature = 1.0
            top_p = 0.95

        # create directory for chart generation
        if not extern:
            save_dir = self.generate_save_dir(run, method, args.binning_type, args.prob_method, args.grouping_style ,args.split, args.model, calibration_data=args.save_data , history_data=args.save_history)
        else:
            save_dir = base_dir

        verb_data = None
        if args.prob_method in ["quantitativ", "qualitativ"]:          
            verb_data_path = 'verbalized_data/'+args.prob_method+'/'+run+'/'+args.model+'/data.json'
            verb_data = self.load_json_data(verb_data_path)
        
        if args.problem == "code-gen":  
            # probs -> confidence of the model
            # is_correct -> label 1: is correct, 0: is not correct
            probs, is_correct, programs, prompts, languages, names, token_logprobs = self.proability_and_correctness_for_samples(results, verb_data, type=args.prob_method)
        elif args.problem == "program-repair":
            probs, is_correct, programs, prompts, languages, names, token_logprobs = self.load_program_repair_data(run_dir)
            num_samples = len(probs)
        else:
            exit("Couldn't find data for problem.")

        group_obj = groups(programs, prompts, languages, names, base_dir)

        if args.grouping_style == 'scc' or args.grouping_style == 'all':
            scc_infos = self.load_scc_data(run, languages, names)
            groups_w = group_obj.create_groups(args.problem, run, group_style=args.grouping_style, scc_infos=scc_infos)
        else:
            # Define group matrix
            groups_w = group_obj.create_groups(args.problem, run, group_style=args.grouping_style)
        
        self.run = run
        self.save_dir = save_dir
        self.num_samples = num_samples

        print(f"Loaded {num_samples} samples")

        self.data = pd.DataFrame(
            {
                "probs": probs,
                "is_correct": is_correct,
                "programs": programs,
                "prompts": prompts,
                "languages": languages,
                "names": names,
                "token_logprobs": token_logprobs,
                "groups": groups_w.tolist()
            }
        )

    def check_data_loaded(self):
        check = [self.__dict__.values()]
        if any(x is None for x in check):
            exit("Failed to load all neded data!")

