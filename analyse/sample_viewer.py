import streamlit as st
import pickle
import pandas as pd
import numpy as np
from colour import Color
import json
from pathlib import Path
from typing import Optional
import gzip

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
            "token_logprobs": token_logprobs,
            "status": res["status"]
        }

        return_values.append(res_dict)

    return return_values

#st.set_page_config(layout="wide")
st.html("""
    <style>
        .stMainBlockContainer {
            max-width:60rem;
        }
    </style>
    """
)
run = 'humaneval-all-keep-Qwen2.5_Coder_7B-Instruct-1.0-comp-1'
method = 'comparison'
binning_type = 'linear'
prob_method = 'avg_logprob'
grouping_style = 'simple'
split = 'split'
model = 'gpt_4o_mini'

with open('./runs/'+run+'/'+method+'/'+binning_type+'/'+prob_method+'/'+grouping_style+'/'+split+'/calibration_data/calibration.pkl', 'rb') as f:
    data = pickle.load(f)

with open('./runs/'+run+'/'+method+'/'+binning_type+'/qualitativ/'+grouping_style+'/'+split+'/'+model+'/calibration_data/calibration.pkl', 'rb') as f:
    data_quant = pickle.load(f)

with open('./runs/'+run+'/'+method+'/'+binning_type+'/quantitativ/'+grouping_style+'/'+split+'/'+model+'/calibration_data/calibration.pkl', 'rb') as f:
    data_qual = pickle.load(f)

st.title("Sample viewer")

correctness = st.toggle("Correct (ON -> all correct, OFF -> all incorrect)")

df = pd.DataFrame({'names': data["names"], 
                   'language': data["language"], 
                   'is_correct': data["is_correct"],
                   'uncalib_avg_token_probs': data["uncalibrated_probs"],
                   'uncalib_qual_probs': data_qual["uncalibrated_probs"],
                   'uncalib_quant_probs': data_quant["uncalibrated_probs"],
                   'hb_probs': data["calibrated_probs_hb"],
                   'lr_probs': data["calibrated_probs_lr"],
                   'ighb_probs': data["calibrated_probs_ighb"],
                   'iglb_probs': data["calibrated_probs_iglb"],
                   'groups': data["groups"].tolist(),
                   'prompts': data["prompts"],
                   'token_logprobs': data["token_logprobs"],
                   'programs': data["programs"]})

if correctness:
    df = df[df["is_correct"] == 1]
else:
    df = df[df["is_correct"] == 0]   

st.write(f"Total: {len(df)}")

selection = [s1 + '_' + s2 + '_' + str(s3) for s1, s2, s3 in zip(data["language"], data["names"], data["is_correct"])]

sorted_lang = list(dict.fromkeys(df["language"]))
sorted_lang.sort()

lang = st.selectbox(
                        "**Language:**",
                        sorted_lang,
                    )

df = df[df["language"] == lang]

name = st.selectbox(
                        "**Name:**",
                        df["names"],
                    )

df = df[df["names"] == name]
p = Path('../MultiPL-E/runs/'+run+'/'+lang+'/'+name+'.results.json.gz')

multiple_res = for_file(p)


if df["is_correct"].values[0] == 1:
    st.markdown("**Evaluation:** :green-badge[:material/check: Correct]")
else:
    st.markdown("**Evaluation:** :red-badge[:material/close: Inorrect]")

st.write(f'**Evaluation result:** {multiple_res[0]["status"]}')

st.write("**Probailities:**")
prob_df = pd.DataFrame(
    {
        "Uncalib (avg)": [df["uncalib_avg_token_probs"].values[0]],
        "Uncalib (qual)": [df["uncalib_qual_probs"].values[0]],
        "Uncalib (quant)": [df["uncalib_quant_probs"].values[0]],
        "HB": [np.round(df["hb_probs"].values[0], 2)],
        "LR": [np.round(df["lr_probs"].values[0], 2)],
        "IGHB": [np.round(df["ighb_probs"].values[0], 2)],
        "IGLB": [np.round(df["iglb_probs"].values[0], 2)],
    }
)
st.dataframe(prob_df.style.format("{:.0%}"), hide_index=True)

st.write("**Groups:**")
groups = pd.DataFrame([df["groups"].values[0]])
#groups.columns = ["simple complexity", "more complex", "complex", "untestable", "prompt >= 500", "not >= 500", "has_examples", "not example", "Longer then median loc", "not loc"]
st.dataframe(groups, hide_index=True)

red = Color("red")
colors = list(red.range_to(Color("green"),100))
with st.expander("See prompt"):
    st.write("**Prompt:**")
    st.code(df["prompts"].values[0])

output = ''
for token in df["token_logprobs"].values[0]:
    tv = list(token.values())[0]
    prob = np.round(np.exp(tv[0]),2)
    output += '<span style="color:'+str(colors[int(prob*100)-1])+'">'+tv[2].replace("Ġ", " ").replace("Ċ", "\n")+'</span>'

with st.expander("See higlighted code"):
    st.write("**Generated code:**")
    st.html('<pre style="background-color: #EBECE4">'+output+'</pre>')

st.write("**Evaluated program:**")
st.code(df["programs"].values[0])


