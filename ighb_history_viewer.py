import streamlit as st
import pickle
import pandas as pd
import numpy as np

with open('ighb_history.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

iteration = st.slider("Choose iteration", 1, len(loaded_dict), 1)
st.header("Before", divider=True)
st.bar_chart(loaded_dict[iteration][0], x_label="Confidence", y_label="Correctness")

st.markdown(f"**Bin**: {loaded_dict[iteration][2][0]}")
st.markdown(f"**Group**: {loaded_dict[iteration][2][1]}")
st.markdown(f"**Delta**: {np.round(loaded_dict[iteration][2][2], 2)}")
st.markdown(f"**Num element changed**: {loaded_dict[iteration][2][3]}")
st.header("After", divider=True)
st.bar_chart(loaded_dict[iteration][1], x_label="Confidence", y_label="Correctness")
st.header("Total per group", divider=True)
st.bar_chart(loaded_dict[iteration][3], x_label="Confidence", y_label="Total per group")

t = np.array(loaded_dict[iteration][2][4][1])

data = {'Confidence': loaded_dict[iteration][2][4][0]}

# Create DataFrame
df = pd.DataFrame(data)

df[["prompt >= 500", "not >= 500", "has_examples", "not example", "Longer then median loc", "not loc"]] = t
st.header("Element details", divider=True)
st.table(df)