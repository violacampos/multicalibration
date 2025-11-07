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
from datasets import Dataset as HF_Dataset, load_dataset


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


#### static helper functions

def add_features(batch):
    result = {}
    if "output" in batch:
        outputs = batch["output"]
        result["output_size"] = [len(output) for output in outputs]
    logprobs = batch["token_logprobs"]
    cumprobs = [sum([float(prob) for (prob, _) in logprob]) for logprob in logprobs]
    result["cumulative_logprob"] = cumprobs
    result["avg_prob"] = [
            np.exp(cumprob / len(logprobs))
            for cumprob, logprobs in zip(cumprobs, logprobs)
        ]
    result["is_correct"] = [int(flag) for flag in batch["is_correct"]]
    return result




class CalibrationDataset:
    """
    Dataset wrapper for CALIBRI datasets.

    """

    def __init__(
        self,
        model: str = "qwen3",
        benchmark: str = "livecodebench",
        group_config: GroupConfig = None,
        split: Optional[str] = None,
        cache_dir: Optional[str] = None,
        args=None,
    ):
        config_name = f"{benchmark}_{model}"
        self.config_name = config_name
        self.dataset = load_dataset(
            "violasara/CALIBRI", config_name, split=split, cache_dir=cache_dir
        )
        
        # always save dataset in dict form for easier handling of splits
        if isinstance(self.dataset, HF_Dataset):
            self.dataset = {split: self.dataset}

        for split in self.dataset:
            self.dataset[split] = self.unfold(self.dataset[split])
            self.dataset[split] = self.dataset[split].map(
                add_features, batched=True
            )

        self.run, self.save_dir = self.init_run_and_outputdir(model, benchmark)

        self.group_config = group_config

        self.median_loc = self.get_median_loc()
        self.median_output = self.get_median_length("output")
        self.median_prompt = self.get_median_length("prompt")    
        self.languages = sorted(set(self.dataset[split]["language"])) # use only last split, should be ok
        self.group_names = self.get_group_names()
        
        for split in self.dataset:
            self.dataset[split] = self.dataset[split].map(
                lambda x: self.add_group_info(x), batched=True
            )


    def get_median_length(self, feature) -> float:
        lengths = [
                    pd.array(self.dataset[split][feature]).map(len)
                    for split in self.dataset
                    if feature in self.dataset[split].features
                ]
        if len(lengths) == 0:
            return 0.0
        else:
            return np.median(  np.concatenate(lengths))
        
        
    def get_median_loc(self) -> float:
        locs = [
                    pd.array(self.dataset[split]["program"])
                    .dropna()
                    .map(lambda x: x.count('\n'))
                    for split in self.dataset
                    if 'program' in self.dataset[split].features
                ]
        if len(locs) == 0:
            return 0.0
        else:
            return np.median( np.concatenate( locs ))
                
        
    def get_group_names(self) -> List[str]:
        
        feature_names = ['language', 
                         'larger_than_median_loc', 
                         'larger_than_median_prompt',
                         'larger_than_median_output',
                         'difficulty_easy', 
                         'difficulty_medium',
                         'difficulty_hard',
                         ]
        group_names = {
                         'larger_than_median_loc': 'loc_high', 
                         'larger_than_median_prompt': 'prompt_len_high',
                         'larger_than_median_output': 'len_high',
                         'difficulty_easy': 'comp_easy', 
                         'difficulty_medium': 'comp_medium',
                         'difficulty_hard': 'comp_hard',}
        enabled_features = [name for name in feature_names if getattr(self.group_config, name, False)]
        result = []
        for feature in enabled_features:
            if feature == 'language':
                for language in self.languages:
                    result.append("lang_" + language)
            else:
                result.append(group_names[feature])
                if self.group_config.add_counter:
                    if feature.startswith('larger'):
                        result.append(group_names[feature].replace('_high', '_low'))
        return result
        
    def add_group_info(self, batch):
        config = self.group_config
        # optional values
        if config.difficulty_easy or config.difficulty_medium or config.difficulty_hard:
            if 'difficulty' not in batch:
                raise ValueError("MultiPL-E does not contain 'difficulty' feature, cannot create difficulty groups.")
            else:
                difficulty = batch["difficulty"]
        if config.larger_than_median_output:
            if 'output' not in batch:
                raise ValueError("MultiPL-E does not contain 'output' feature, cannot create output based groups.")
            else:
                outputs = batch["output"]
        groups = [[] for _ in range(len(batch['id']))]
        
        languages = batch["language"]
        
        if config.language:
            for lang in self.languages:
                for i, l in enumerate(languages):
                    groups[i].append(1 if l == lang else 0)
                    
        if config.larger_than_median_loc:
            for i, program in enumerate(batch["program"]):
                loc_count = program.count("\n") if program is not None else 0
                groups[i].append(1 if loc_count > self.median_loc else 0)
                if config.add_counter:
                    groups[i].append(0 if loc_count > self.median_loc else 1)
        if config.larger_than_median_prompt:
            for i, prompt in enumerate(batch["prompt"]):
                groups[i].append(1 if len(prompt) > self.median_prompt else 0)
                if config.add_counter:
                    groups[i].append(0 if len(prompt) > self.median_prompt else 1)
        if config.larger_than_median_output:
            for i, output in enumerate(outputs):
                groups[i].append(1 if len(output) > self.median_output else 0)
                if config.add_counter:
                    groups[i].append(0 if len(output) > self.median_output else 1)
                
        
        if config.difficulty_easy:
            for i, d in enumerate(difficulty):
                groups[i].append(1 if d == "easy" else 0)
                if config.add_counter:
                    groups[i].append(0 if d == "easy" else 1)
        if config.difficulty_medium:
            for i, d in enumerate(difficulty):
                groups[i].append(1 if d in ["medium", "middle"] else 0)
                if config.add_counter:
                    groups[i].append(0 if d in ["medium", "middle"] else 1)
        if config.difficulty_hard:
            for i, d in enumerate(difficulty):
                groups[i].append(1 if d == "hard" else 0)
                if config.add_counter:
                    groups[i].append(0 if d == "hard" else 1)
            
        return {'groups': groups}

    @staticmethod
    def unfold(dataset: HF_Dataset) -> HF_Dataset:
        unfolded_data = []
        for example in dataset:
            for i in range(10):
                unfolded_example = {
                    # Scalar fields (keep as-is)
                    "id": example["id"],
                    "prompt": example["prompt"],
                    "language": example["language"],
                    # Sequence fields (extract i-th element)
                    "program": example["program"][i],
                    "is_correct": example["is_correct"][i],
                    "token_logprobs": example["token_logprobs"][i],
                    # Add sample index
                    "sample_idx": i,
                }

                # Handle optional fields
                if "output" in example:
                    unfolded_example["output"] = example["output"][i]
                if "difficulty" in example:
                    unfolded_example["difficulty"] = example["difficulty"]
                if "name" in example:
                    unfolded_example["name"] = example["name"]
                if "code_token_idx" in example:
                    unfolded_example["code_token_idx"] = example["code_token_idx"][i]

                unfolded_data.append(unfolded_example)

        return HF_Dataset.from_list(unfolded_data)

    @staticmethod
    def init_run_and_outputdir(model: str, benchmark: str):
        run = benchmark + "_" + model
        dir = os.path.join("results", run, "output")
        os.makedirs(dir, exist_ok=True)
        return run, dir
    
    def get_train_probs(self, prob_type: str):
        return np.array(self.dataset["train"][prob_type])

    def get_train_is_correct(self):
        return np.array(self.dataset["train"]["is_correct"])

    def get_train_groups(self):
        return np.array(self.dataset["train"]["groups"])

    ########################################################

    def get_test_probs(self, prob_type: str):
        return np.array(self.dataset["test"][prob_type])

    def set_test_probs(self, new_probs):
        self.dataset["test"]["probs"] = new_probs

    def get_test_is_correct(self):
        return np.array(self.dataset["test"]["is_correct"])

    def get_test_groups(self):
        return np.array(self.dataset["test"]["groups"])

    def get_test_languages(self):
        return self.dataset["test"]["language"]

    def get_test_names(self):
        return self.dataset["test"]["name"]

    def get_test_programs(self):
        return self.dataset["test"]["program"] if "program" in self.dataset["test"] else None

    def get_test_prompts(self):
        return self.dataset["test"]["prompt"]

    def get_test_token_logprobs(self):
        return (
            self.dataset["test"]["token_logprobs"]
            if "token_logprobs" in self.dataset["test"]
            else None
        )

    #########################################################

    def get_val_probs(self, prob_type: str):
        return np.array(self.dataset["validation"][prob_type])

    def get_val_is_correct(self):
        return np.array(self.dataset["validation"]["is_correct"])

    def get_val_groups(self):
        return np.array(self.dataset["validation"]["groups"])


