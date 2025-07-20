import numpy as np
import json

class groups:

    def __init__(self, programs, prompts, languages, names, include_counter=False):
        """
            Initilaizes a group object.

            :param programs: List of programs
            :param prompts: List of prompts
            :param languages: List of languages
            :param names: List of task names
        """
        self.prompts = prompts
        self.programs = programs
        self.languages = languages
        self.names = names
        self.groups_w = []
        
        self.median_loc = 0
        self.median_prompt = 0
        self.run = None
        self.include_counter = include_counter
    
    def create_groups(self, problem, run, group_style=None, scc_infos=None):
        """
            Creates the group matrix

            :param problem: Check if the groups are created for code generation or program repair.
            :param run: Name of the run
            :param group_style: Flag which grouping style to use
            :param scc_infos: List of scc infos (for complexity grouping)

            :return: group matrix
        """
        self.run = run

        if problem == 'code-gen': 
            self.set_median_loc()
            self.set_median_prompt()
            if scc_infos is None:
                for program, prompt  in zip(self.programs, self.prompts):
                    self.groups_w.append(self.check_groups_code_gen(prompt, program, group_style=group_style))   
            else:
                for program, prompt, infos  in zip(self.programs, self.prompts, scc_infos):
                    self.groups_w.append(self.check_groups_code_gen(prompt, program, group_style=group_style, infos=infos))   

        elif problem == 'program-repair':
            if group_style == 'simple':
                repair_set = self.run.split("/")[0]
                group_file = "program_repair/groups/"+repair_set+"/groups.json"
                with open(group_file, 'r') as f:
                    group_content = json.load(f)   

                for program, prompt, name in zip(self.programs, self.prompts, self.names):
                    self.groups_w.append(self.check_groups_program_repair(prompt, program, name, group_content, group_style=group_style))

        return np.array(self.groups_w)

    def set_median_loc(self):
        """
            Sets the median line of code for group creation purposes.
        """
        loc = np.array([program.count("\n")+1 for program in self.programs])
        self.median_loc = np.median(loc)

    def set_median_prompt(self):
        """
            Sets the median length of the prompt for group creation purposes.
        """
        length = np.array([len(prompts) for prompts in self.prompts])
        self.median_prompt = np.median(length)

    def check_groups_code_gen(self, prompt, program, group_style=None, infos=None):
        """
            Checks the if the sample is in the defined groups with the corresponding grouping style.

            :param prompt: Check if the groups are created for code generation or program repair.
            :param program: Name of the run
            :param group_style: Flag which grouping style to use
            :param infos: List of scc infos (for complexity grouping)

            :return: List of assigned groups (zero or one for group assignment)
        """
        check = []
        if group_style == 'simple':
            # Median prompt length
            check.append(1 if len(prompt) >= self.median_prompt else 0)
            if self.include_counter: check.append(0 if len(prompt) >= self.median_prompt else 1)
            
            # example
            check.append(1 if "example" in prompt else 0)
            if self.include_counter: check.append(0 if "example" in prompt else 1)
            
            # Median line of code
            check.append(1 if program.count("\n")+1 > self.median_loc else 0)
            if self.include_counter: check.append(0 if program.count("\n")+1 > self.median_loc else 1)

        elif group_style == 'scc':
            # https://en.wikipedia.org/wiki/Cyclomatic_complexity
            # does not need a counter group because all samples are mapped in these 4 categories
            check.append(1 if infos["Complexity"] < 11 else 0)

            check.append(1 if infos["Complexity"] >= 11 and infos["Complexity"] < 21 else 0)

            check.append(1 if infos["Complexity"] >= 21 and infos["Complexity"] <= 50 else 0)

            check.append(1 if infos["Complexity"] > 50 else 0)

        elif group_style == 'categories':
            # String operations
            check.append(1 if "word" in prompt or "string" in prompt or "char" in prompt else 0)
            if self.include_counter: check.append(0 if "word" in prompt or "string" in prompt or "char" in prompt else 1)

            # Mathematical
            check.append(1 if "math" in prompt or "integer" in prompt or "float" in prompt or ("number" in prompt and "calculate" in prompt) else 0)
            if self.include_counter: check.append(0 if "math" in prompt or "integer" in prompt or "float" in prompt or ("number" in prompt and "calculate" in prompt) else 1)

            # Data objects
            check.append(1 if "list" in prompt or "array" in prompt or "dict" in prompt or "tree" in prompt else 0)
            if self.include_counter: check.append(0 if "list" in prompt or "array" in prompt or "dict" in prompt or "tree" in prompt else 1)
        
        elif group_style == 'all':
            # simple
            # Median prompt length
            check.append(1 if len(prompt) >= self.median_prompt else 0)
            if self.include_counter: check.append(0 if len(prompt) >= self.median_prompt else 1)
            
            # example
            check.append(1 if "example" in prompt else 0)
            if self.include_counter: check.append(0 if "example" in prompt else 1)
            
            # Median line of code
            check.append(1 if program.count("\n")+1 > self.median_loc else 0)
            if self.include_counter: check.append(0 if program.count("\n")+1 > self.median_loc else 1)

            # complexity
            check.append(1 if infos["Complexity"] < 11 else 0)
            check.append(1 if infos["Complexity"] >= 11 and infos["Complexity"] < 21 else 0)
            check.append(1 if infos["Complexity"] >= 21 and infos["Complexity"] <= 50 else 0)
            check.append(1 if infos["Complexity"] > 50 else 0)

            # categories
            # String operations
            check.append(1 if "word" in prompt or "string" in prompt or "char" in prompt else 0)
            if self.include_counter: check.append(0 if "word" in prompt or "string" in prompt or "char" in prompt else 1)
            # Mathematical
            check.append(1 if "math" in prompt or "integer" in prompt or "float" in prompt or ("number" in prompt and "calculate" in prompt) else 0)
            if self.include_counter: check.append(0 if "math" in prompt or "integer" in prompt or "float" in prompt or ("number" in prompt and "calculate" in prompt) else 1)
            # Data objects
            check.append(1 if "list" in prompt or "array" in prompt or "dict" in prompt or "tree" in prompt else 0)
            if self.include_counter: check.append(0 if "list" in prompt or "array" in prompt or "dict" in prompt or "tree" in prompt else 1)
        else:
            exit("Grouping sytle not found!")

        return check
    
    def check_groups_program_repair(self, prompt, program, name, groups_file, group_style=None):  
        """
            Checks the if the sample is in the defined groups with the corresponding grouping style for program repair.

            :param prompt: Check if the groups are created for code generation or program repair.
            :param program: Name of the run
            :param name: Name of the task
            :param groups_file: File where the groups assignement is stored
            :param group_style: Flag which grouping style to use

            :return: List of assigned groups (zero or one for group assignment)
        """ 
        check = []
        
        if group_style == 'simple':
            problem_name = name.split("#")[1].split("_")[0]
            
            check.append(1 if problem_name in groups_file["single_line"]  else 0)
            check.append(1 if problem_name in groups_file["single_hunk"]  else 0)
            check.append(1 if problem_name in groups_file["multi_hunk"]  else 0)
        else:
            exit("Grouping sytle not found!")

        return check