import os

from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from datasets import Dataset, IterableDataset, load_dataset


class GroupConfig:
    def __init__(
        self,
        add_counter=False,
        language=False,
        larger_than_median_loc=False,
        larger_than_median_prompt=False,
        larger_than_median_output=False,
        difficulty_easy=False,
        difficulty_medium=False,
        difficulty_hard=False,
    ):
        self.add_counter = add_counter
        self.language = language
        self.larger_than_median_loc = larger_than_median_loc
        self.larger_than_median_prompt = larger_than_median_prompt
        self.larger_than_median_output = larger_than_median_output
        self.difficulty_easy = difficulty_easy
        self.difficulty_medium = difficulty_medium
        self.difficulty_hard = difficulty_hard


#### static helper functions ######################

def add_features(batch):
    result = {}
    if "output" in batch:
        outputs = batch["output"]
        result["output_size"] = [len(output) for output in outputs]
    logprobs = batch["token_logprobs"]
    cumprobs = [sum([float(prob) for (prob, _) in logprob]) for logprob in logprobs]
    result["cumulative_logprob"] = cumprobs
    result["avg_prob"] = [
        np.exp(cumprob / len(logprobs)) for cumprob, logprobs in zip(cumprobs, logprobs)
    ]
    
    tail_probs = [sum([float(prob) for (prob, _) in logprob[-40:]]) for logprob in logprobs]
    result["tail_prob"] = [np.exp(tail_prob / 40) for tail_prob in tail_probs]
    if "code_token_idx" in batch:
        codeprobs = [probs[start:end] for probs, (start, end) in zip(logprobs, batch["code_token_idx"])]
        result['code_prob'] = [np.exp(sum([float(prob) for (prob, _) in codeprob]) / len(codeprob)) if len(codeprob) > 0 else 0.0 for codeprob in codeprobs]
    result["is_correct"] = [int(flag) for flag in batch["is_correct"]]
    return result


def explode(batch):
    result = {}
    list_features = ["program", "output", "is_correct", "token_logprobs", "code_token_idx"]
    single_val_features = ["id", "name", "prompt", "language", "difficulty", "model"]
    features = batch.data.keys()
    for feature in features:
        if feature in list_features:
            result[feature] = [x for sublist in batch[feature] for x in sublist]
        elif feature in single_val_features:
            result[feature] = [x for x in batch[feature] for _ in range(10)]
    return result


### Dataset class ###################################

class CalibrationDataset:
    """
    Dataset wrapper for CALIBRI datasets.

    """

    def __init__(
        self,
        group_config: GroupConfig,
        model: str = "qwen3",
        benchmark: str = "livecodebench",
        split: Optional[str] = None,
        cache_dir: Optional[str] = None,
        args=None,
    ):
        config_name = f"{benchmark}_{model}"
        self.config_name = config_name
        dataset = load_dataset(
            "violasara/CALIBRI", config_name, split=split, cache_dir=cache_dir
        )

        # always save dataset in dict[str, HF_Dataset] form for easier handling of splits
        if isinstance(dataset, Dataset) or isinstance(dataset, IterableDataset):
            # prefer .name for NamedSplit-like objects, fall back to str(split)
            key = getattr(split, "name", None) if split is not None else None
            key = str(key) if key is not None else ("default" if split is None else str(split))
            self.dataset: Dict[str, Dataset | IterableDataset] = {key: dataset}
        else:
            # self.dataset is mapping-like: coerce keys to plain strings
            self.dataset = {str(k): v for k, v in dataset.items()}

        for split_name in self.dataset:
            self.dataset[split_name] = (
                self.dataset[split_name]
                .map(
                    explode,
                    batched=True,
                    remove_columns=self.dataset[split_name].column_names,
                )
                .flatten_indices()
            )

            self.dataset[split_name] = self.dataset[split_name].map(add_features, batched=True)

        self.run, self.save_dir = self.init_run_and_outputdir(model, benchmark)

        self.group_config = group_config

        self.median_loc = self.get_median_loc()
        self.median_output = self.get_median_length("output")
        self.median_prompt = self.get_median_length("prompt")
        self.languages = sorted(
            set(self.dataset[split_name]["language"])
        )  # use only last split here, should be ok
        self.group_names = self.get_group_names()

        for split_name in self.dataset:
            self.dataset[split_name] = self.dataset[split_name].map(
                lambda x: self.add_group_info(x), batched=True
            )

    def get_median_length(self, feature) -> float:
        lengths = [
            pd.Series(self.dataset[split_name][feature]).map(len)
            for split_name in self.dataset
            if feature in self.dataset[split_name].features
        ]
        if len(lengths) == 0:
            return 0.0
        else:
            return float(np.median(np.concatenate(lengths)))

    def get_median_loc(self) -> float:
        locs = [
            pd.array(self.dataset[split_name]["program"])
            .dropna()
            .map(lambda x: x.count("\n"))
            for split_name in self.dataset
            if "program" in self.dataset[split_name].features
        ]
        if len(locs) == 0:
            return 0.0
        else:
            return float(np.median(np.concatenate(locs)))

    def get_group_names(self) -> List[str]:

        feature_names = [
            "language",
            "larger_than_median_loc",
            "larger_than_median_prompt",
            "larger_than_median_output",
            "difficulty_easy",
            "difficulty_medium",
            "difficulty_hard",
        ]
        group_names = {
            "larger_than_median_loc": "loc_high",
            "larger_than_median_prompt": "prompt_len_high",
            "larger_than_median_output": "len_high",
            "difficulty_easy": "comp_easy",
            "difficulty_medium": "comp_medium",
            "difficulty_hard": "comp_hard",
        }
        enabled_features = [
            name for name in feature_names if getattr(self.group_config, name, False)
        ]
        result = []
        for feature in enabled_features:
            if feature == "language":
                for language in self.languages:
                    result.append("lang_" + language)
            else:
                result.append(group_names[feature])
                if self.group_config.add_counter:
                    if feature.startswith("larger"):
                        result.append(group_names[feature].replace("_high", "_low"))
        return result

    def add_group_info(self, batch):
        config = self.group_config
        # optional values
        if config.difficulty_easy or config.difficulty_medium or config.difficulty_hard:
            if "difficulty" not in batch:
                raise ValueError(
                    "MultiPL-E does not contain 'difficulty' feature, cannot create difficulty groups."
                )
            else:
                difficulty = batch["difficulty"]
        if config.larger_than_median_output:
            if "output" not in batch:
                raise ValueError(
                    "MultiPL-E does not contain 'output' feature, cannot create output based groups."
                )
            else:
                outputs = batch["output"]
        groups = [[] for _ in range(len(batch["id"]))]

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

        return {"groups": groups}


    @staticmethod
    def init_run_and_outputdir(model: str, benchmark: str):
        run = benchmark + "_" + model
        dir = os.path.join("results", run, "output")
        os.makedirs(dir, exist_ok=True)
        return run, dir
    
    ########################################################

    def get_train_probs(self, prob_type: str):
        if prob_type not in self.dataset["train"].features:
            raise ValueError(f"The {self.config_name.split('_')[0]} dataset does not support '{prob_type}'. Please try another benchmark or probablity method.")
        return np.array(self.dataset["train"][prob_type])

    def get_train_is_correct(self):
        return np.array(self.dataset["train"]["is_correct"])

    def get_train_groups(self):
        return np.array(self.dataset["train"]["groups"])

    ########################################################

    def get_test_probs(self, prob_type: str):
        if prob_type not in self.dataset["test"].features:
            raise ValueError(f"The {self.config_name.split('_')[0]} dataset does not support '{prob_type}'. Please try another benchmark or probablity method.")
        return np.array(self.dataset["test"][prob_type])

    def get_test_is_correct(self):
        return np.array(self.dataset["test"]["is_correct"])

    def get_test_groups(self):
        return np.array(self.dataset["test"]["groups"])

    def get_test_languages(self):
        return self.dataset["test"]["language"]

    def get_test_names(self):
        return self.dataset["test"]["name"]

    def get_test_programs(self):
        return (
            self.dataset["test"]["program"]
            if "program" in self.dataset["test"]
            else None
        )

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
        if prob_type not in self.dataset["validation"].features:
            raise ValueError(f"The {self.config_name.split('_')[0]} dataset does not support '{prob_type}'. Please try another benchmark or probablity method.")
        return np.array(self.dataset["validation"][prob_type])

    def get_val_is_correct(self):
        return np.array(self.dataset["validation"]["is_correct"])

    def get_val_groups(self):
        return np.array(self.dataset["validation"]["groups"])
