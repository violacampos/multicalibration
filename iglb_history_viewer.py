import streamlit as st
import pickle
import pandas as pd
import numpy as np
import altair as alt

#st.set_page_config(layout="wide")

groups = st.toggle("Enable Groups")

grid = np.round(np.arange(0.0, 1+(1/10), 1/10),2)

with open('iglb_history.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)


iteration = st.slider("Choose iteration", 1, len(loaded_dict)-1, 1)

if groups:
    group = st.slider("Choose Group", 0, 9, 1)
    before = np.array(loaded_dict[iteration][0])[:, group]
    after = np.array(loaded_dict[iteration][1])[:, group]
else:
    before = loaded_dict[iteration][4]
    after = loaded_dict[iteration][5]

st.header("Before", divider=True)
st.bar_chart(before, x_label="Confidence", y_label="Correctness")

st.markdown(f"**Bin**: {loaded_dict[iteration][2][1]}; **Group**: {loaded_dict[iteration][2][2]}; **Tau**: {'>=' if 1 == loaded_dict[iteration][2][0] else '<='}")

st.markdown(f"**Num element changed**: {loaded_dict[iteration][2][4]}")
st.header("After", divider=True)
st.bar_chart(after, x_label="Confidence", y_label="Correctness")
st.header("Total per group", divider=True)
st.bar_chart(loaded_dict[iteration][3], x_label="Confidence", y_label="Total per group")

t = np.array(loaded_dict[iteration][2][5][1])
c = np.array(loaded_dict[iteration][2][5][2]).reshape(-1, 1)

data = {'Confidence': loaded_dict[iteration][2][5][0]}

# Create DataFrame
df = pd.DataFrame(data)
df[["simple complexity", "more complex", "complex", "untestable", "prompt >= 500", "not >= 500", "has_examples", "not example", "Longer then median loc", "not loc"]] = t
df[["is_correct"]] = c

st.header("Element details", divider=True)
st.table(df)

st.header("Scores", divider=True)

gasce = []
for idx, i in enumerate(loaded_dict["score"]):
    gasce.append(i[-1])
    loaded_dict["score"][idx] = i[:-1]

df = pd.DataFrame(loaded_dict["score"])
df.columns = ['Run', 
            'Type',
            'ECE', 
            'ASCE', 
            'MSE',
            'brier_ref', 
            'skill_score']
st.table(df)

print(np.array(gasce))
df = pd.DataFrame()
df[["simple complexity", "more complex", "complex", "untestable", "prompt >= 500", "not >= 500", "has_examples", "not example", "Longer then median loc", "not loc"]] = gasce
st.table(df)

st.header("Groups", divider=True)
st.markdown("0. simple complexity (< 11)")
st.markdown("1. more complex (11 - 20)")
st.markdown("2. complex  (21 - 50)")
st.markdown("3. untestable (> 50)")
st.markdown("4. prompt >= 500")
st.markdown("5. not 4")
st.markdown("6. examples")
st.markdown("7. not 6")
st.markdown("8. longer then median LoC")
st.markdown("9. not 8")