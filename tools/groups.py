import re
import numpy as np
import json

class groups:

    def __init__(self, programs, prompts, languages, names, base_dir):
        self.prompts = prompts
        self.programs = programs
        self.languages = languages
        self.names = names
        self.groups_w = []
        
        self.median_loc = 0
        self.run = None
        self.base_dir = base_dir
    
    def create_groups(self, problem, run, group_style=None, scc_infos=None):
        self.run = run

        if problem == 'code-gen':
            if group_style == 'scc':
                self.median_loc = np.mean(np.array([d["Code"] for d in scc_infos]))
                for program, prompt, infos in zip(self.programs, self.prompts, scc_infos):
                    self.groups_w.append(self.check_groups_code_gen(prompt, program, group_style=group_style, infos=infos))
            elif group_style == 'simple' or group_style == 'categories':
                self.set_median_loc()
                for program, prompt in zip(self.programs, self.prompts):
                    self.groups_w.append(self.check_groups_code_gen(prompt, program, group_style=group_style))                

        elif problem == 'program-repair':
            if group_style == 'simple':
                repair_set = self.run.split("/")[0]
                group_file = self.base_dir+"program_repair/groups/"+repair_set+"/groups.json"
                with open(group_file, 'r') as f:
                    group_content = json.load(f)   

                for program, prompt, name in zip(self.programs, self.prompts, self.names):
                    self.groups_w.append(self.check_groups_program_repair(prompt, program, name, group_content, group_style=group_style))

        return np.array(self.groups_w)

    def set_median_loc(self):
        loc = np.array([program.count("\n")+1 for program in self.programs])
        self.median_loc = np.median(loc)

    def check_groups_code_gen(self, prompt, program, group_style=None, include_counter=True, infos=None):
        check = []
        if group_style == 'simple':
            #prompt_longer_then_500 = 1 if len(prompt) >= 500 else 0
            #counter_prompt_longer_then_500 = 0 if len(prompt) >= 500 else 1
            check.append(1 if len(prompt) >= 500 else 0)
            if include_counter: check.append(0 if len(prompt) >= 500 else 1)

            #has_examples = 1 if "example" in prompt else 0
            #counter_has_examples = 0 if "example" in prompt else 1
            check.append(1 if "example" in prompt else 0)
            if include_counter: check.append(0 if "example" in prompt else 1)

            #longer_then_mean_loc = 1 if program.count("\n")+1 > self.median_loc else 0
            #counter_longer_then_mean_loc = 0 if program.count("\n")+1 > self.median_loc else 1
            check.append(1 if program.count("\n")+1 > self.median_loc else 0)
            if include_counter: check.append(0 if program.count("\n")+1 > self.median_loc else 1)

        elif group_style == 'scc':
            # https://en.wikipedia.org/wiki/Cyclomatic_complexity
            # does not need a counter group because all samples are mapped in these 4 categories
            #simple = 1 if infos["Complexity"] < 11 else 0
            check.append(1 if infos["Complexity"] < 11 else 0)

            #more_complex = 1 if infos["Complexity"] >= 11 and infos["Complexity"] < 21 else 0
            check.append(1 if infos["Complexity"] >= 11 and infos["Complexity"] < 21 else 0)

            #complex = 1 if infos["Complexity"] >= 21 and infos["Complexity"] <= 50 else 0
            check.append(1 if infos["Complexity"] >= 21 and infos["Complexity"] <= 50 else 0)

            #untestable = 1 if infos["Complexity"] > 50 else 0
            check.append(1 if infos["Complexity"] > 50 else 0)

            #has_examples = 1 if "example" in prompt else 0
            #counter_has_examples = 0 if "example" in prompt else 1
            check.append(1 if "example" in prompt else 0)
            if include_counter: check.append(0 if "example" in prompt else 1)

            #prompt_longer_then_500 = 1 if len(prompt) >= 500 else 0
            #counter_prompt_longer_then_500 = 0 if len(prompt) >= 500 else 1
            check.append(1 if len(prompt) >= 500 else 0)
            if include_counter: check.append(0 if len(prompt) >= 500 else 1)

            #longer_then_mean_loc = 1 if infos["Code"] > self.median_loc else 0
            #counter_longer_then_mean_loc = 0 if infos["Code"] > self.median_loc else 1
            check.append(1 if infos["Code"] > self.median_loc else 0)
            if include_counter: check.append(0 if infos["Code"] > self.median_loc else 1)

        elif group_style == 'categories':
            check.append(1 if "word" in prompt or "string" in prompt or "char" in prompt else 0)
            if include_counter: check.append(0 if "word" in prompt or "string" in prompt or "char" in prompt else 1)

            check.append(1 if "math" in prompt or "integer" in prompt or "float" in prompt or ("number" in prompt and "calculate" in prompt) else 0)
            if include_counter: check.append(0 if "math" in prompt or "integer" in prompt or "float" in prompt or ("number" in prompt and "calculate" in prompt) else 1)

            check.append(1 if "list" in prompt or "array" in prompt or "dict" in prompt or "tree" in prompt else 0)
            if include_counter: check.append(0 if "list" in prompt or "array" in prompt or "dict" in prompt or "tree" in prompt else 1)
        else:
            exit("Grouping sytle not found!")

        return check
    def check_groups_program_repair(self, prompt, program, name, groups_file, group_style=None, include_counter=True):   
        check = []
        
        if group_style == 'simple':
            problem_name = name.split("#")[1].split("_")[0]
            
            check.append(1 if problem_name in groups_file["single_line"]  else 0)
            check.append(1 if problem_name in groups_file["single_hunk"]  else 0)
            check.append(1 if problem_name in groups_file["multi_hunk"]  else 0)
        else:
            exit("Grouping sytle not found!")

        return check
    """def check_groups_scc_code_gen(self, prompt, program, infos, include_counter=True):
        check = []

        # https://en.wikipedia.org/wiki/Cyclomatic_complexity
        # does not need a counter group because all samples are mapped in these 4 categories
        simple = 1 if infos["Complexity"] < 11 else 0
        check.append(simple)

        more_complex = 1 if infos["Complexity"] >= 11 and infos["Complexity"] < 21 else 0
        check.append(more_complex)

        complex = 1 if infos["Complexity"] >= 21 and infos["Complexity"] <= 50 else 0
        check.append(complex)

        untestable = 1 if infos["Complexity"] > 50 else 0
        check.append(untestable)

        has_examples = 1 if "example" in prompt else 0
        counter_has_examples = 0 if "example" in prompt else 1
        check.append(has_examples)
        if include_counter: check.append(counter_has_examples)

        prompt_longer_then_500 = 1 if len(prompt) >= 500 else 0
        counter_prompt_longer_then_500 = 0 if len(prompt) >= 500 else 1
        check.append(prompt_longer_then_500)
        if include_counter: check.append(counter_prompt_longer_then_500)

        longer_then_mean_loc = 1 if infos["Code"] > self.median_loc else 0
        counter_longer_then_mean_loc = 0 if infos["Code"] > self.median_loc else 1
        check.append(longer_then_mean_loc)
        if include_counter: check.append(counter_longer_then_mean_loc)


        return check"""