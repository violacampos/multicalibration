import os

problem = "code-gen"

# code-gen
run = "../MultiPL-E/runs/humaneval-all-keep-Qwen2.5_Coder_7B-Instruct-1.0-comp-1"

if problem == "code-gen":
    grouping_styles = ["simple", "scc"]
else:
    grouping_styles = ["simple"]

if problem == "code-gen":
    prob_methods = ["avg_logprob", "qualitativ", "quantitativ"]
else:
    prob_methods = ["avg_logprob"]

split = ["", "--split"]

for prob_method in prob_methods:
    if prob_method in ["qualitativ", "quantitativ"]:
        model = "--model gpt_4o_mini"
    else:
        model = ""
    for grouping_style in grouping_styles:
        for s in split:
            os.system(f"python compare_methods.py --dir {run} --prob-method {prob_method} --problem {problem} --grouping-style {grouping_style} --save-charts --save-table {s} {model}")