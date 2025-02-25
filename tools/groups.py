import re

def check_groups(prompt, program):
    check = []
    
    contains_import = 1 if "import" in program else 0
    check.append(contains_import)

    prompt_greater_then_500 = 1 if len(prompt) >= 500 else 0
    check.append(prompt_greater_then_500)

    has_examples = 1 if "example" in prompt else 0
    check.append(has_examples)

    input_vars = re.search("\((.*?)\)", program)
    input_var_count = len(input_vars[0].split(","))
    res = 1 if input_var_count > 2 else 0
    check.append(res)

    contains_return = 1 if "return" in program else 0
    check.append(contains_return)

    return check