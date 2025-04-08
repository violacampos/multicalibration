import re
import numpy as np


class groups:

    def __init__(self, programs, prompts):
        self.prompts = prompts
        self.programs = programs
        self.groups_w = []
        
        self.median_loc = 0
    
    def create_groups(self, scc=None):
        if scc is None:
            self.set_median_loc()

            for program, prompt in zip(self.programs, self.prompts):
                self.groups_w.append(self.check_groups(prompt, program))
        else:
            self.median_loc = np.mean(np.array([d["Code"] for d in scc]))
            for program, prompt, infos in zip(self.programs, self.prompts, scc):
                self.groups_w.append(self.check_groups_scc(prompt, program, infos))

        return np.array(self.groups_w)

    def set_median_loc(self):
        loc = np.array([program.count("\n")+1 for program in self.programs])
        self.median_loc = np.median(loc)

    def check_groups(self, prompt, program, include_counter=True):
        check = []

        prompt_longer_then_500 = 1 if len(prompt) >= 500 else 0
        counter_prompt_longer_then_500 = 0 if len(prompt) >= 500 else 1
        check.append(prompt_longer_then_500)
        if include_counter: check.append(counter_prompt_longer_then_500)

        has_examples = 1 if "example" in prompt else 0
        counter_has_examples = 0 if "example" in prompt else 1
        check.append(has_examples)
        if include_counter: check.append(counter_has_examples)

        longer_then_mean_loc = 1 if program.count("\n")+1 > self.median_loc else 0
        counter_longer_then_mean_loc = 0 if program.count("\n")+1 > self.median_loc else 1
        check.append(longer_then_mean_loc)
        if include_counter: check.append(counter_longer_then_mean_loc)

        return check
    
    def check_groups_scc(self, prompt, program, infos, include_counter=True):
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


        return check