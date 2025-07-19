import os
from tools.data import data_loader

# load results from the generation process
results, _, _, _ =  data_loader.load_multipl_e_run('./MultiPL-E/runs/humaneval-all-keep-Qwen2.5_Coder_7B-Instruct-1.0-comp-1_results/')

_, _, programs, _, languages, names = data_loader.load_samples(results)

# create a file for every sample with the programming lanugage extension
for program, lang, name in zip(programs, languages, names):
    if lang == 'elixir':
        lang = 'ex'
    fn = "programs/"+lang+"/"+name+"."+lang
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    f = open(fn, "w", encoding="utf-8")
    f.write(program)
    f.close()

# Evaluate the created programs with the scc command line tool
os.system("scc -f json -o humaneval-all-keep-Qwen2.5_Coder_7B-Instruct-1.0-comp-1_results.json --by-file programs")