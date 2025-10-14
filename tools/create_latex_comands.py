import re
from typing import Dict, List

import numpy as np


name_to_latex = {'Qwen3-Coder-30B-A3B': 'QwenIII',
                 'livecodebench': 'LCB',
                 'humaneval': 'HE'}




def define_latex_cmd(name, value):
    name = re.sub(r"[^a-zA-Z]", "", name)  # sanity check for valid latex commands
    print("\\newcommand{\\%s }{ %s }" % (name, f"{value:.3f}"))


def print_commands_for(
    calibration_result: Dict, scoring: str, calibration: str, model: str, benchmark: str
):

    scores = (
        calibration_result["scores_calibrated"]["Calib"]
        if "scores_calibrated" in calibration_result
        else calibration_result["scores_uncalibrated"]["Uncalib"]
    )
    for score, val in scores.items():
        if not isinstance(val, np.ndarray):
            name = f"{benchmark}{model}{scoring}{calibration}{score}"
            define_latex_cmd(name, val)


def print_latex_commands(
    model: str,
    benchmark: str,
    initial_scoring: str,
    results: Dict[str, Dict[str, dict]],
) -> str:
    for method, result_dict in results.items():

        print_commands_for(
            result_dict,
            calibration=method,
            scoring=initial_scoring,
            model=name_to_latex[model],
            benchmark=name_to_latex[benchmark],
        )
