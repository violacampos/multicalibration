import gzip
import itertools
import json
import os
from pathlib import Path
import random
from typing import List, Optional
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class GroupConfig:
    def __init__(
        self,
        add_counter=False,
        language=True,
        larger_than_median_loc=True,
        larger_than_median_prompt=True,
        larger_than_median_output=True,
        difficulty_easy=True,
        difficulty_medium=True,
        difficulty_hard=True,
    ):
        self.add_counter = add_counter
        self.language = language
        self.larger_than_median_loc = larger_than_median_loc
        self.larger_than_median_prompt = larger_than_median_prompt
        self.larger_than_median_output = larger_than_median_output
        self.difficulty_easy = difficulty_easy
        self.difficulty_medium = difficulty_medium
        self.difficulty_hard = difficulty_hard





class LiveCodeBenchDataset(Dataset):

    def __init__(
        self,
        jsonl_path: str,
        split: str = "train",
        group_config: GroupConfig = None,
        benchmark="livecodebench",
        n: int = 10,
        val_ratio: float = 0.25,
        seed: int = 42,
        args = None
    ):
        self.data = {"train": [], "val": [], "test": []}
        self.data_path = jsonl_path
        self.split = split
        self.n = n  # Number of generations per sample
        self.run, self.save_dir = self.get_run_and_outdir_from_path(
            jsonl_path, benchmark
        )

        self.group_config = group_config
        self.group_names = []
        self.is_names_set = False
        self.median_prompt = None
        self.median_loc = None
        self.median_output = None
        self.languages = []
        entries = []
        with open(jsonl_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                
                entries.append(entry)

        # Shuffle and split
        random.seed(seed)
        random.shuffle(entries)
        split_idx1 = int(len(entries) * (1 - val_ratio * 2))
        split_idx2 = int(len(entries) * (1 - val_ratio))
        split_entries = {
            "train": entries[:split_idx1],
            "val": entries[split_idx1:split_idx2],
            "test": entries[split_idx2:],
        }

        # Broadcast after split
        for split, split_entries in split_entries.items():
            data = {key: [] for key in split_entries[0].keys()}

            for entry in split_entries:
                for key, value in entry.items():
                    if isinstance(value, list) and len(value) == self.n:
                        data[key].extend(value)
                    else:
                        data[key].extend([value] * self.n)
            self.data[split] = pd.DataFrame(data)
            if args and args.prob_method=='code_prob':
                self.data[split].dropna(subset=['code_logprob'], inplace=True)
            # some postprocessing:
            self.data[split]["is_correct"] = self.data[split]["is_correct"].astype(int)
            self.data[split]["avg_prob"] = np.exp(
                self.data[split]["cumulative_logprob"] / self.data[split]["token_count"]
            )
            self.add_group_info(split=split)

            #self.data[split]["code_prob"] = np.exp(self.data[split]['code_logprob'])
            #self.data[split]["tail_prob"] = np.exp(self.data[split]['tail_logprob'])
            #self.data[split]["code_top20_prob"] = np.exp(self.data[split]['avg_top20_code_probs'])
            #self.data[split]["tail_top20_prob"] = np.exp(self.data[split]['avg_top20_tail'])

    @staticmethod
    def get_run_and_outdir_from_path(path: str, benchmark_name: str):
        model = path.split("/")[-3]
        filename = path.split("/")[-1].rstrip(".jsonl")
        run = benchmark_name + "_" + model + "_" + filename
        dir = os.path.join("results", run, "output")
        os.makedirs(dir, exist_ok=True)
        return run, dir
    
    def get_model(self)->str:
        return self.data_path.split("/")[-3]

    def add_group_info(self, split=None):
        """
        Adds group information to the dataset based on the provided configuration.
        """
        split = self.split if split is None else split
        self.median_prompt = (
            self.data[split]["prompt"].str.len().median()
            if self.median_prompt == None
            else self.median_prompt
        )
        self.median_loc = (
            self.data[split]["program"].str.count("\n").add(1).median()
            if "program" in self.data[split]
            else None
        )
        self.median_output = (
            self.data[split]["output_size"].median()
            if self.median_output == None
            else self.median_output
        )

        groups = []
        for idx, row in self.data[split].iterrows():
            check = []
            
            if self.group_config.difficulty_easy:
                if not self.is_names_set:
                        self.group_names.append('comp_easy')
                check.append(1) if row["difficulty"] == "easy" else check.append(0)
            if self.group_config.difficulty_medium:
                if not self.is_names_set:
                        self.group_names.append('comp_medium')
                check.append(1) if row["difficulty"] in ["medium", "middle"] else check.append(0)
            if self.group_config.difficulty_hard:
                if not self.is_names_set:
                        self.group_names.append('comp_hard')
                check.append(1) if row["difficulty"] == "hard" else check.append(0)

            if self.group_config.larger_than_median_prompt:
                if not self.is_names_set:
                        self.group_names.append('prompt_len_high')
                check.append(1 if len(row["prompt"]) > self.median_prompt else 0)
                if self.group_config.add_counter:
                    if not self.is_names_set:
                        self.group_names.append('prompt_len_low')
                    check.append(0 if len(row["prompt"]) > self.median_prompt else 1)
            if self.group_config.larger_than_median_loc:
                if not self.is_names_set:
                        self.group_names.append('loc_high')
                check.append(
                    1 if row["program"] != None and row["program"].count("\n") + 1 > self.median_loc else 0
                )
                if self.group_config.add_counter:
                    if not self.is_names_set:
                        self.group_names.append('loc_low')
                    
                    check.append(
                        1 if row['program'] == None  or row["program"].count("\n") + 1 < self.median_loc else 0
                    )
            if self.group_config.larger_than_median_output:
                if not self.is_names_set:
                    self.group_names.append('len_high')
                check.append(1 if row["output_size"] > self.median_output else 0)
                if self.group_config.add_counter:
                    if not self.is_names_set:
                        self.group_names.append('len_low')
                    check.append(0 if row["output_size"] > self.median_output else 1)
            if self.group_config.language:
                if len(self.languages) == 0:
                    self.languages = sorted(set(self.data[split]["language"]))
                for language in self.languages:
                    if not self.is_names_set:
                        self.group_names.append('lang_' + language)
                    check.append(1) if row["language"] == language else check.append(0)
            groups.append(check)
            if not self.is_names_set:
                self.is_names_set = True
        
        self.data[split]["groups"] = groups

    # def collect_languages(self):
    #     languages = set()
    #     for split_df in self.data.values():
    #         languages |= set(split_df['language'])
    #     self.languages = sorted(languages)

    def __len__(self):
        return len(self.data[self.split])

    def __getitem__(self, idx):
        # Each item is a list of 10 values
        return self.data[self.split].iloc[idx]

    ########################################################

    def get_train_probs(self, prob_type:str):
        return self.data["train"][prob_type]

    def get_train_is_correct(self):
        return self.data["train"]["is_correct"]

    def get_train_groups(self):
        return np.array(self.data["train"]["groups"].to_list())

    ########################################################

    def get_test_probs(self, prob_type:str):
        return self.data["test"][prob_type]

    def set_test_probs(self, new_probs):
        self.data["test"]["probs"] = new_probs

    def get_test_is_correct(self):
        return self.data["test"]["is_correct"]

    def get_test_groups(self):
        return np.array(self.data["test"]["groups"].to_list())

    def get_test_languages(self):
        return self.data["test"]["language"]

    def get_test_names(self):
        return self.data["test"]["name"]

    def get_test_programs(self):
        return self.data["test"]["program"] if "program" in self.data["test"] else None

    def get_test_prompts(self):
        return self.data["test"]["prompt"]

    def get_test_token_logprobs(self):
        return self.data["test"]["token_logprobs"] if "token_logprobs" in self.data["test"] else None

    #########################################################

    def get_val_probs(self, prob_type:str):
        return self.data["val"][prob_type]

    def get_val_is_correct(self):
        return self.data["val"]["is_correct"]

    def get_val_groups(self):
        return np.array(self.data["val"]["groups"].to_list())



class HumanEvalDataset(LiveCodeBenchDataset):
    def __init__(
        self,
        jsonl_path: str,
        run_dir:str,
        split: str = "train",
        group_config: GroupConfig = None,
        benchmark="humaneval",
        n: int = 10,
    ):
        self.data = {"train": [], "val": [], "test": []}
        self.data_path = jsonl_path
        self.split = split
        self.n = n  # Number of generations per sample
        self.run, self.save_dir = self.get_run_and_outdir_from_path(
            jsonl_path, benchmark
        )

        self.group_config = group_config
        self.group_names = []
        self.is_names_set = False
        self.median_prompt = None
        self.median_loc = None
        self.median_output = None
        self.languages = []
        
        results, num_samples = self.load_multipl_e_run(run_dir)
        
        data = self.load_samples(results)

        self.split_in_train_test(data)
        for split in self.data:
            self.add_group_info(split=split)
            
            
    def get_test_probs(self, prob_type:str):
        return self.data["test"]['probs'] # ignore type for humanEval
    
    def get_train_probs(self, prob_type:str):
        return self.data["train"]['probs'] # ignore type for humanEval
    
    def get_val_probs(self, prob_type:str):
        return self.data["val"]['probs'] # ignore type for humanEval

    def for_file(self, path: Path):
        """
        Loads the sample data from one file.
        Parts used from https://github.com/nuprl/MultiPL-E/blob/main/multipl_e/completions.py

        :param path: Path to load data from

        :return: dict with sample data
        """
        if path.suffix == ".gz":
            try:
                with gzip.open(path, "rt") as f:
                    data = json.load(f)
            except Exception as e:
                data = None
            
        else:
            with open(path, "r") as f:
                data = json.load(f)

        if data is None:
            return None

        return_values = []
        n = len(data["results"])
        for d, res in zip(data["tokens_info"], data["results"]):
            token_infos = d
            cumulative_logprob = token_infos["cumulative_logprob"]
            token_logprobs = token_infos["token_logprobs"]
            token_ids = token_infos["len"]
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
                "token_logprobs": token_logprobs,
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
            if os.path.isdir(os.path.join(path_to_parent, fname)):
                yield os.path.join(path_to_parent, fname)


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
            results = [
                self.for_file(p)
                for p in itertools.chain(
                    Path(path).glob("*.results.json"),
                    Path(path).glob("*.results.json.gz"),
                )
            ]
            results = [r for r in results if r is not None]

            
            n = list(set(r[0]["n"] for r in results))[0]

            num_samples = len(results) * n
        else:
            results = []
            for folder in subfolders:
                results_folder = [
                    self.for_file(p)
                    for p in itertools.chain(
                        Path(folder).glob("*.results.json"),
                        Path(folder).glob("*.results.json.gz"),
                    )
                ]
                results.extend([r for r in results_folder if r is not None])

            
            n = list(set(r[0]["n"] for r in results))[0]

            num_samples = len(results) * n

        return results, num_samples
        
    def split_in_train_test(self, df:pd.DataFrame, fractions:Optional[List[float]] = [0.5, 0.25, 0.25]):
        """
            Split the data into train, validation and test. 

            :param df: Dataframe containing the data for the run
        """  

        # clean split along problems
        group_names = ["train", "val", "test"]
        rng = np.random.default_rng(42)
        unique_names = df['name'].unique()
        rng.shuffle(unique_names)
        
        n = len(unique_names)
        sizes = (np.array(fractions) * n).astype(int)
        sizes[-1] = n - sizes[:-1].sum()  # fix rounding
        splits = np.split(unique_names, np.cumsum(sizes)[:-1])


        name_to_split = {name: group for group, names in zip(group_names, splits) for name in names}
        df["split"] = df["name"].map(name_to_split)

        # --- 4. get split dataframes ---
        self.data['train'] = df[df["split"] == "train"].copy().drop(['split'], axis=1)
        self.data['val'] = df[df["split"] == "val"].copy().drop(['split'], axis=1)
        self.data['test'] = df[df["split"] == "test"].copy().drop(['split'], axis=1)

    def load_samples(self, results:List[List[dict]], type="avg_logprob"):
        """
        Loads needed data for all samples.
        Parts used from https://github.com/parameterlab/apricot/blob/main/src/eval.py. (Quantative and qualitative data)

        :param results: Loaded Multipl_E results
        :param verb_data: Loaded verbalized data
        :param type: Type of the probability that should be loaded (avg_logprob, quantitativ, qualitativ)

        :return: probabilities, labels, programs, prompts, languages, names, token probabilities
        """
        prob_value_list = []
        is_correct = []
        prompts = []
        programs = []
        languages = []
        names = []
        output_sizes = []
  
        token_logprobs = []
        


        # Get the token probabilities from the samples and create arrays
        for r in results:
            for sample in r:
                token_count = (
                    len(sample["token_ids"])
                    if "token_ids" in sample
                    else sample["token_count"]
                )
                cumulative_logprob = sample["cumulative_logprob"]

                # Some name changes are needed for mapping.
                if sample["language"] == "elixir":
                    lang = "ex"
                elif sample["language"] == "go_test.go":
                    lang = "go"
                else:
                    lang = sample["language"]

                # Differentiate in different probability types
                if type == "avg_logprob":
                    # prob = np.round(np.exp(cumulative_logprob / token_count), 2)
                    prob = (
                        [
                            np.exp(cum_log / count)
                            for cum_log, count in zip(cumulative_logprob, token_count)
                        ]
                        if isinstance(token_count, list)
                        else np.exp(cumulative_logprob / token_count)
                    )

                # collect values and add to return list
                programs.append(sample["program"])
                languages.append(lang)
                names.append(sample["name"])
                prompts.append(sample["prompt"])
                prob_value_list.append(prob)
                if "is_correct" in sample:
                    is_correct.append(
                        [1 if x == True else 0 for x in sample["is_correct"]]
                    )
                else:
                    is_correct.append(1) if sample["c"] == 1 else is_correct.append(0)
                token_logprobs.append(sample["token_logprobs"])
                output_sizes.append(len(sample["program"]))
               

        prob_value_list = np.array(prob_value_list)
        is_correct = np.array(is_correct)
        languages = np.array(languages)


        
        return pd.DataFrame(
            {
                "probs": prob_value_list,
                "is_correct": is_correct,
                "program": programs,
                "prompt": prompts,
                "language": languages,
                "name": names,
                "token_logprobs": token_logprobs,
                "output_size": output_sizes
               
            }
        )



if __name__ == "__main__":
    # DEBUG
    path = "../LiveCodeBench/output/Qwen3-Coder-30B-A3B/preprocessed/codegeneration_10_0.2.jsonl"
    
    config = GroupConfig(
        add_counter=False,
        larger_than_median_loc=True,
        larger_than_median_prompt=True,
        difficulty_easy=False,
        difficulty_medium=False,
        difficulty_hard=False,
        language=True
    )


    dataset = LiveCodeBenchDataset(path, split="train", group_config=config)
    print(f"Dataset size: {len(dataset)}")
    print(f"First item: {dataset[0]}")
