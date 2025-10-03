import json
import os
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class GroupConfig:
    def __init__(self, 
                 add_counter=False, 
                 larger_than_median_loc=True, 
                 larger_than_median_prompt=True,
                 larger_than_median_output=True,
                 difficulty_easy=True,
                 difficulty_medium=True,
                 difficulty_hard=True):
        self.add_counter = add_counter
        self.larger_than_median_loc = larger_than_median_loc
        self.larger_than_median_prompt = larger_than_median_prompt
        self.larger_than_median_output = larger_than_median_output
        self.difficulty_easy = difficulty_easy  
        self.difficulty_medium = difficulty_medium
        self.difficulty_hard = difficulty_hard



class LiveCodeBenchDataset(Dataset):
    
    def __init__(self, 
                 jsonl_path:str, 
                 split:str='train', 
                 group_config:GroupConfig=None,
                 n:int=10, 
                 val_ratio:float=0.25, 
                 seed:int=42):
        self.data = {'train':[], 'val':[], 'test':[]}
        self.data_path = jsonl_path
        self.split = split
        self.n = n  # Number of generations per sample
        self.run, self.save_dir = self.get_run_and_outdir_from_path(jsonl_path)

        self.group_config = group_config
        self.median_prompt = None
        self.median_loc = None
        self.median_output = None
        entries = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                entries.append(entry)
                
        # Shuffle and split
        random.seed(seed)
        random.shuffle(entries)
        split_idx1 = int(len(entries) * (1 - val_ratio * 2))
        split_idx2 = int(len(entries) * (1 - val_ratio))
        split_entries = {
            'train': entries[:split_idx1],
            'val': entries[split_idx1:split_idx2],
            'test': entries[split_idx2:]
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
            # some postprocessing:
            self.data[split]['is_correct'] = self.data[split]['is_correct'].astype(int)
            self.data[split]['probs'] = np.exp(self.data[split]['cumulative_logprob'] / self.data[split]['token_count'])
            self.add_group_info(split=split)

    @staticmethod
    def get_run_and_outdir_from_path(path:str):
        model = path.split("/")[-3]
        filename = path.split("/")[-1].rstrip(".jsonl")
        run = 'LiveCodeBench_'+ model + '_' + filename
        dir = "runs/"+run+"/output/"
        os.makedirs(dir, exist_ok=True)
        return run, dir

    def add_group_info(self, split=None):
        """
        Adds group information to the dataset based on the provided configuration.
        """
        split = self.split if split is None else split
        self.median_prompt = self.data[split]['prompt'].str.len().median()
        self.median_loc = self.data[split]['program'].str.count('\n').add(1).median()
        self.median_output = self.data[split]['output_size'].median()
        groups = []
        for idx, row in self.data[split].iterrows():
            check = []
            if self.group_config.difficulty_easy:
                check.append(1) if row['difficulty'] == 'easy' else check.append(0) 
            if self.group_config.difficulty_medium:
                check.append(1) if row['difficulty'] == 'medium' else check.append(0)
            if self.group_config.difficulty_hard:
                check.append(1) if row['difficulty'] == 'hard' else check.append(0)
            
            if self.group_config.larger_than_median_prompt:
                check.append(1 if len(row['prompt']) > self.median_prompt else 0)
                if self.group_config.add_counter:
                    check.append(0 if len(row['prompt']) > self.median_prompt else 1)
            if self.group_config.larger_than_median_loc:
                check.append(1 if row['program'].count('\n') + 1 > self.median_loc else 0)
                if self.group_config.add_counter:
                    check.append(0 if row['program'].count('\n') + 1 > self.median_loc else 1)
            if self.group_config.larger_than_median_output:
                check.append(1 if row['output_size'] > self.median_output else 0)
                if self.group_config.add_counter:
                    check.append(0 if row['output_size'] > self.median_output else 1)
            groups.append(check)
        self.data[split]['groups'] = groups



    def __len__(self):
        return len(self.data[self.split])

    def __getitem__(self, idx):
        # Each item is a list of 10 values
        return self.data[self.split].iloc[idx]
    
    ########################################################
    
    def get_train_probs(self):
        return self.data['train']['probs']
    
    def get_train_is_correct(self):
        return self.data['train']['is_correct']
    
    def get_train_groups(self):
        return np.array(self.data['train']['groups'].to_list())

    ########################################################

    def get_test_probs(self):
        return self.data['test']['probs']
    
    def set_test_probs(self, new_probs):
        self.data['test']['probs'] = new_probs

    def get_test_is_correct(self):
        return self.data['test']['is_correct']

    def get_test_groups(self):
        return np.array(self.data['test']['groups'].to_list())

    def get_test_languages(self):
        return self.data['test']['language']

    def get_test_names(self):
        return self.data['test']['name']

    def get_test_programs(self):
        return self.data['test']['program']

    def get_test_prompts(self):
        return self.data['test']['prompt']

    def get_test_token_logprobs(self):
        return self.data['test']['token_logprobs']
    
    #########################################################
    
    def get_val_probs(self):
        return self.data['val']['probs']
    
    def get_val_is_correct(self):
        return self.data['val']['is_correct']

    def get_val_groups(self):
        return np.array(self.data['val']['groups'].to_list())
    
        
if __name__ == "__main__":
    # DEBUG
    path = "../LiveCodeBench/output/Qwen3-Coder-30B-A3B/preprocessed/codegeneration_10_0.2.jsonl"
    config = GroupConfig(add_counter=False, 
                 larger_than_median_loc=True, 
                 larger_than_median_prompt=True,
                 difficulty_easy=True,
                 difficulty_medium=True,
                 difficulty_hard=True)

    dataset = LiveCodeBenchDataset(path, split='train', group_config=config)
    print(f"Dataset size: {len(dataset)}")
    print(f"First item: {dataset[0]}")