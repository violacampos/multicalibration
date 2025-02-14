import json
from pathlib import Path
import gzip
from typing import Optional
import itertools
import numpy as np


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
    n = len(data["results"])
    token_infos         = data["tokens_info"][0]
    cumulative_logprob  = token_infos["cumulative_logprob"]
    token_logprobs      = token_infos["token_logprobs"]
    token_ids           = token_infos["len"]
    c = len([True for r in data["results"] if r["status"]
            == "OK" and r["exit_code"] == 0])
    return {
        "name": data["name"], 
        "n": n,
        "c": c,
        "temperature": data["temperature"] if "temperature" in data else 0.2,
        "top_p": data["top_p"],
        "cumulative_logprob": cumulative_logprob,
        "token_ids": token_ids,
        "token_logprobs": token_logprobs
    }

def load_multipl_e_run(path):
    # load the data from the run directory
    results = [for_file(p) for p in itertools.chain(
                Path(path).glob("*.results.json"), Path(path).glob("*.results.json.gz"))]
    results = [r for r in results if r is not None]
    temperature = list(set(r["temperature"] for r in results))[0]
    top_p = list(set(r["top_p"] for r in results))[0]

    num_samples = len(results)

    return results, temperature, top_p, num_samples

def proability_and_correctness_for_samples(results, type="avg_logprob"):
    prob_value_list = []
    is_correct = []

    # Get the token probailities from the samples and create arrays
    for r in results:
        token_count = len(r["token_ids"])
        cumulative_logprob = r["cumulative_logprob"]
        # for later use
        logprobs = [list(logprob.values())[0][0] for logprob in r["token_logprobs"]]

        if type == "avg_logprob":
            prob = np.round(np.exp(cumulative_logprob / token_count), 2) 
            
        # collect average token probabilty and correctnes value
        prob_value_list.append(prob)
        is_correct.append(1) if r["c"] == 1 else is_correct.append(0)

    prob_value_list     = np.array(prob_value_list)
    is_correct          = np.array(is_correct)

    return prob_value_list, is_correct

def avg_token_probability(cumulative_logprob, token_count):
    return np.round(np.exp(cumulative_logprob / token_count), 2) 


