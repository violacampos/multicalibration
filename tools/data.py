import json
from pathlib import Path
import gzip
from typing import Optional
import itertools
import numpy as np
import os
import datetime
import re
from tools.groups import Groups
import pandas as pd

class data_loader:

    def __init__(self, args, run_dir, extern, method):
        """
            Initilaizes a data loader object

            :param args: passed arguments from command line
            :param run_dir: directory of the samples to load
            :param extern: is the method called from an external method (comparison)
            :param method: Specifies which methods usese the data loader

            :return: grid, chartmaker
        """
        self.run = None
        self.save_dir = None
        self.num_samples = None
        self.data = None

        self.setup_data(args, run_dir, extern, method)
        self.check_data_loaded()

    
    def gunzip_json(self, path: Path) -> Optional[dict]:
        """
            Reads a .json.gz file, but produces None if any error occurs.
            Used from https://github.com/nuprl/MultiPL-E/blob/main/multipl_e/util.py
        """
        try:
            with gzip.open(path, "rt") as f:
                return json.load(f)
        except Exception as e:
            return None

    """def gzip_json(self, path: Path, data: dict) -> None:
        with gzip.open(path, "wt") as f:
            json.dump(data, f)"""

    def for_file(self, path: Path):
        """
            Loads the sample data from one file.
            Parts used from https://github.com/nuprl/MultiPL-E/blob/main/multipl_e/completions.py

            :param path: Path to load data from

            :return: dict with sample data
        """
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

    def folders_in(self, path_to_parent):
        """
            Checks if samples lie in the subfolder of the passed dir.
            Used from StackOverflow https://stackoverflow.com/questions/27789665/check-if-a-given-directory-contains-any-directory-in-python

            :param path_to_parent: Partend folder path

            :return: subfolder
        """
        for fname in os.listdir(path_to_parent):
            if os.path.isdir(os.path.join(path_to_parent,fname)):
                yield os.path.join(path_to_parent,fname)


    def load_livecodebench_data(self, path):
        results = []
        files = list(Path(path).glob("*.jsonl"))
        assert len(files) > 0, "No .jsonl files found in the specified path."
        for file in files:
            with open(file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    results.append(data)
        assert len(results) > 0, "No data found in the specified file."
        num_samples = len(results) * len((results[0]["program"]))
        return [results], num_samples

    def load_multipl_e_run(self, path):
        """
            Loads the results for every sample in the run path.
            Parts used from https://github.com/nuprl/MultiPL-E/blob/main/multipl_e/completions.py

            :param path: Path of the multipl_e run

            :return: results, temeprature, top_p, number of samples
        """
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
        """
            Loads stored json data from a path.

            :param path: Path of the json file

            :return: json data
        """
        with open(path, 'r') as file:
            json_data = json.load(file)
        return json_data

    def load_samples(self, results, verb_data, type="avg_logprob"):
        """
            Loads needed data for all samples.
            Parts used from https://github.com/parameterlab/apricot/blob/main/src/eval.py. (Quantative and qualitative data)

            :param results: Loaded Multipl_E results
            :param verb_data: Loaded verbalized data
            :param type: Type of the probability that should be loaded (avg_logprob, quantitativ, qualitativ)

            :return: probabilities, labels, programms, prompts, languages, names, token probabilities
        """
        prob_value_list = []
        is_correct = []
        prompts = []
        programms = []
        languages = []
        names = []
        successful = []
        token_logprobs = []
        difficulty = []


        QUALITATIVE_SCALE = {
            "Very low": 0,
            "Low": 0.15,
            "Somewhat low": 0.3,
            "Medium": 0.45,
            "Somewhat high": 0.60,
            "High": 0.75,
            "Very high": 0.9,
        }

        # Get the token probabilities from the samples and create arrays
        for r in results:
            for sample in r:
                token_count = len(sample["token_ids"]) if 'token_ids' in sample else sample["token_count"]
                cumulative_logprob = sample["cumulative_logprob"]
                
                # Some name changes are needed for mapping.
                if sample["language"] == 'elixir':
                    lang = "ex"
                elif sample["language"] == 'go_test.go':
                    lang = "go"
                else:
                    lang = sample["language"]

                # Differentiate in different probability types
                if type == "avg_logprob":
                    #prob = np.round(np.exp(cumulative_logprob / token_count), 2)
                    prob = [
                        np.exp(cum_log / count) for cum_log, count in zip(cumulative_logprob, token_count)
                            ] if isinstance(token_count, list) else np.exp(cumulative_logprob / token_count)
                elif type == "quantitativ":
                    try:
                        template = r"\d{1,3}(?:\.\d+)?\s?\%?"
                        d = verb_data[lang][sample["name"]]
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
                        res = re.search(template, d["completion"]).group(0)
                        prob = QUALITATIVE_SCALE[res]

                    except AttributeError:
                        successful.append(False)
                    successful.append(True)
                                        
                # collect values and add to return list
                programms.append(sample["program"])
                languages.append(lang)
                names.append(sample["name"])
                prompts.append(sample["prompt"])
                prob_value_list.append(prob)
                if 'is_correct' in sample:
                    is_correct.append([1 if x == True else 0 for x in sample["is_correct"]])
                else:
                    is_correct.append(1) if sample["c"] == 1 else is_correct.append(0)
                token_logprobs.append(sample["token_logprobs"])
                if 'difficulty' in sample:
                    difficulty.append(sample["difficulty"])

        # Check if all samples got a probability
        if type in ["qualitativ", "quantitativ"]:
            print(f"Succesful extracted: {len(successful)}")

        prob_value_list     = np.array(prob_value_list)
        is_correct          = np.array(is_correct)
        languages           = np.array(languages)

        return prob_value_list, is_correct, programms, prompts, languages, names, token_logprobs, difficulty

    def avg_token_probability(self, cumulative_logprob, token_count):
        """
            Calculates the average token probability.

            :param cumulative_logprob: Sum of all token probabailities
            :param token_count: Total of tokens in the sample

            :return: average token probability
        """
        return np.exp(cumulative_logprob / token_count)


    def load_scc_data(self, run, languages, names):
        """
            Loads the scc data to later create groups with the complexity scores. To load a samples scc data the name and the languages needs to be known.

            :param run: Name of the run
            :param languages: programming languages of the sample
            :param names: Name of each sample.
            
            :return: average token probability
        """
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

            :param run: Name of the run
            :param method: Name of the used calibration method
            :param binning: Binning type
            :param prob_generation: How the probabilities were generated. (avg_logprob, quantitativ, qualitativ)
            :param grouping_style: How the groups were generated.
            :param split: If the data is splitted
            :param model: Name of the used model for generation
            :param calibration_data: Flag to save calibrated data
            :param history_data: Flag to save the history of certain methods
            
            :return: directory string         

        """
        dir = "runs/"+run+"/"+method+"/"+binning+"/"+prob_generation+"/"+grouping_style+"/"

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

    def load_program_repair_data(self, path):
        """
            Loads the data for the code repair problem

            :param path: Path of the sample
            
            :return: probabilities, labels, programs, prompts, languages, names, token probabailities
        """
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
            prob = np.exp(d["cumulative_logprob"] / d["token_count"])
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
        """
            Is called on intialization. Loads data for given arguments and creates directories.

            :param args: Passed commandline arguments
            :param run_dir: Path of the run directory
            :param extern: Flag specifies if its called from another script
            :param method: Name of the method that calls the data loader
            
        """
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
            save_dir = ''

        verb_data = None
        if args.prob_method in ["quantitativ", "qualitativ"]:          
            verb_data_path = 'verbalized_data/'+args.prob_method+'/'+run+'/'+args.model+'/data.json'
            verb_data = self.load_json_data(verb_data_path)
        
        if args.problem == "code-gen":  
            # probs -> confidence of the model
            # is_correct -> label 1: is correct, 0: is not correct
            probs, is_correct, programs, prompts, languages, names, token_logprobs, difficulty = self.load_samples(results, verb_data, type=args.prob_method)
        elif args.problem == "program-repair":
            probs, is_correct, programs, prompts, languages, names, token_logprobs, difficulty = self.load_program_repair_data(run_dir)
            num_samples = len(probs)
        else:
            exit("Couldn't find data for problem.")

        # Creates group obj and group matrix
        group_obj = Groups(programs, prompts, languages, names, include_counter=args.counter_groups)

        if args.grouping_style == 'scc' or args.grouping_style == 'all':
            scc_infos = self.load_scc_data(run, languages, names)
            groups_w = group_obj.create_groups(args.problem, 
                                               run, 
                                               group_style=args.grouping_style, 
                                               scc_infos=scc_infos, 
                                               difficulty=difficulty)
        else:
            # Define group matrix
            groups_w = group_obj.create_groups(args.problem, 
                                               run, 
                                               group_style=args.grouping_style, 
                                               difficulty=difficulty)

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
        """
            Checks if the values for every element in the data dict are set.    
        """
        check = [self.__dict__.values()]
        if any(x is None for x in check):
            exit("Failed to load all neded data!")

