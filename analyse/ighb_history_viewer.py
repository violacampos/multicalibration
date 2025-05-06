import streamlit as st
import pickle
import pandas as pd
import numpy as np
import altair as alt

#st.set_page_config(layout="wide")
fold = st.toggle("Enable 5-Fold")
split = st.toggle("Enable Full Dataset", disabled=fold)
groups = st.toggle("Enable Groups")
each_iteration = st.toggle("Enable Iteration")

grid = np.round(np.arange(0.0, 1+(1/10), 1/10),2)

if split:
    with open('./history_data/ighb_history_no_split.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
elif fold:
    with open('./history_data/ighb_history_kfold.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    fold = st.slider("Choose Fold", 1, 5, 1)
    all_fold = loaded_dict
    loaded_dict = loaded_dict[fold-1]
else:
    with open('./history_data/ighb_history.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)

if each_iteration:
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

    st.markdown(f"**Bin**: {loaded_dict[iteration][2][0]}; **Group**: {loaded_dict[iteration][2][1]}; **Delta**: {np.round(loaded_dict[iteration][2][2], 2)}")

    st.markdown(f"**Num element changed**: {loaded_dict[iteration][2][3]}")
    st.header("After", divider=True)
    st.bar_chart(after, x_label="Confidence", y_label="Correctness")
    st.header("Total per group", divider=True)
    st.bar_chart(loaded_dict[iteration][3], x_label="Confidence", y_label="Total per group")

    t = np.array(loaded_dict[iteration][2][4][1])
    c = np.array(loaded_dict[iteration][2][4][2]).reshape(-1, 1)

    data = {'Confidence': loaded_dict[iteration][2][4][0]}

    # Create DataFrame
    df = pd.DataFrame(data)
    df[["simple complexity", "more complex", "complex", "untestable", "prompt >= 500", "not >= 500", "has_examples", "not example", "Longer then median loc", "not loc"]] = t
    df[["is_correct"]] = c

    st.header("Element details", divider=True)
    st.table(df)

else:
    
    fold_mean = st.toggle("Mean over folds")

    if fold_mean:
        print(all_fold)
        exit()
    else:
        if groups:
            group = st.slider("Choose Group", 0, 9, 1)

            before =  pd.DataFrame({'Confidence': np.array(loaded_dict[1][0])[:, group], 'bin': grid})
            before_total = pd.DataFrame({group: loaded_dict[1][3][:, group] ,'bin': grid})

            after =  pd.DataFrame({'Confidence': np.array(loaded_dict[len(loaded_dict)-1][1])[:, group], 'bin': grid})
            after_total = pd.DataFrame({group: loaded_dict[len(loaded_dict)-1][3][:, group] ,'bin': grid})
        else:       
            before =  pd.DataFrame({'Confidence': loaded_dict[1][4], 'bin': grid})
            before_total = {'bin': grid}
            for idx, g in enumerate(np.array(loaded_dict[1][3]).T):
                before_total[idx] = loaded_dict[1][3].T[idx]
            before_total = pd.DataFrame(before_total)

            after =  pd.DataFrame({'Confidence': loaded_dict[len(loaded_dict)-1][5], 'bin': grid})
            after_total = {'bin': grid}
            for idx, g in enumerate(np.array(loaded_dict[len(loaded_dict)-1][3]).T):
                after_total[idx] = loaded_dict[len(loaded_dict)-1][3].T[idx]
            after_total = pd.DataFrame(after_total)

        # Create DataFrame
        st.header("Before", divider=True)
        st.bar_chart(data=before, x="bin", y="Confidence", x_label="Confidence", y_label="Correctness")
        st.header("After", divider=True)
        st.bar_chart(data=after, x="bin", y="Confidence", x_label="Confidence", y_label="Correctness")
        st.header("Before Total per group", divider=True)
        st.bar_chart(data=before_total, x="bin", x_label="Confidence", y_label="Total per group")
        st.header("After Total per group", divider=True)
        st.bar_chart(data=after_total, x="bin", x_label="Confidence", y_label="Total per group")

st.header("Scores", divider=True)
#print(loaded_dict["score"])
gasce = []
for idx, i in enumerate(loaded_dict["score"]):
    gasce.append(i[-1])
    loaded_dict["score"][idx] = i[:-1]
#scores = {x: loaded_dict["score"][x] for x in loaded_dict["score"] if x not in ["GASCE"]}
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