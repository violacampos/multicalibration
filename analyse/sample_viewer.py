import streamlit as st
import pickle
import pandas as pd
import numpy as np
from colour import Color

st.set_page_config(layout="wide")
run = 'humaneval-all-keep-Qwen2.5_Coder_14B-Instruct-1.0-comp-1'
binning_type = 'linear'

with open('./calibration_data/'+run+'/'+binning_type+'/calibration.pkl', 'rb') as f:
    data = pickle.load(f)


st.title("Sample viewer")

selection = [s1 + '_' + s2 + '_' + str(s3) for s1, s2, s3 in zip(data["language"], data["names"], data["is_correct"])]

option = st.selectbox(
                        "**Name:**",
                        selection,
                    )

item_id = selection.index(option)

st.write("**Name:**", data["names"][item_id])
st.write("**Language:** ", data["language"][item_id])


if data["is_correct"][item_id] == 1:
    st.markdown("**Evaluation:** :green-badge[:material/check: Correct]")
else:
    st.markdown("**Evaluation:** :red-badge[:material/close: Inorrect]")

st.write("**Probailities**:")
df = pd.DataFrame(
    {
        "Uncalib": [data["uncalibrated_probs"][item_id]],
        "HB": [np.round(data["calibrated_probs_hb"][item_id], 2)],
        "LR": [np.round(data["calibrated_probs_lr"][item_id], 2)],
        "IGHB": [np.round(data["calibrated_probs_ighb"][item_id], 2)],
        "IGLB": [np.round(data["calibrated_probs_iglb"][item_id], 2)],
    }
)
st.dataframe(df.style.format("{:.0%}"), hide_index=True)

st.write("**Groups**:")
groups = pd.DataFrame([data["groups"][item_id]])
groups.columns = ["simple complexity", "more complex", "complex", "untestable", "prompt >= 500", "not >= 500", "has_examples", "not example", "Longer then median loc", "not loc"]
st.dataframe(groups, hide_index=True)

red = Color("red")
colors = list(red.range_to(Color("green"),100))
with st.expander("See prompt"):
    st.write("**Prompt**:")
    st.code(data["prompts"][item_id])

output = ''
for token in data["token_logprobs"][item_id]:
    tv = list(token.values())[0]
    prob = np.round(np.exp(tv[0]),2)
    output += '<span style="color:'+str(colors[int(prob*100)-1])+'">'+tv[2].replace("Ġ", " ").replace("Ċ", "\n")+'</span>'

with st.expander("See higlighted code"):
    st.write("**Generated code**:")
    st.html('<pre style="background-color: #EBECE4">'+output+'</pre>')

st.write("**Evaluated program**:")
st.code(data["programs"][item_id])


