import os
from tools import data

results, _, _, _ =  data.load_multipl_e_run('./MultiPL-E/runs/humaneval-all-keep-Qwen2.5_Coder_7B-Instruct-1.0-comp-1_results/')

_, _, programs, _, languages, names = data.proability_and_correctness_for_samples(results)

for program, lang, name in zip(programs, languages, names):
    if lang == 'elixir':
        lang = 'ex'
    fn = "programs/"+lang+"/"+name+"."+lang
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    f = open(fn, "w", encoding="utf-8")
    f.write(program)
    f.close()


"""os.system("scc -f json -o results.json --by-file programs")"""