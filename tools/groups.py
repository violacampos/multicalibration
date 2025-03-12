import re
import numpy as np


class groups:

    def __init__(self, programs, prompts):
        self.prompts = prompts
        self.programs = programs
        self.groups_w = []
        
        self.median_loc = 0
    
    def create_groups(self):
        self.set_median_loc()

        for program, prompt in zip(self.programs, self.prompts):
            self.groups_w.append(self.check_groups(prompt, program))

        return self.groups_w

    def set_median_loc(self):
        loc = np.array([program.count("\n")+1 for program in self.programs])
        self.median_loc = np.median(loc)

    def check_groups(self, prompt, program):
        check = []
        
        """contains_import = 1 if "import" in program else 0
        check.append(contains_import)"""

        prompt_greater_then_500 = 1 if len(prompt) >= 500 else 0
        check.append(prompt_greater_then_500)

        has_examples = 1 if "example" in prompt else 0
        check.append(has_examples)

        input_vars = re.search("\((.*?)\)", program)
        input_var_count = len(input_vars[0].split(","))
        res = 1 if input_var_count > 2 else 0
        check.append(res)

        longer_then_mean_loc = 1 if program.count("\n")+1 > self.median_loc else 0
        check.append(longer_then_mean_loc)

        """contains_return = 1 if "return" in program else 0
        check.append(contains_return)"""

        return check