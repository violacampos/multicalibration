import numpy as np
from tools import calibration_scores, binning
from termcolor import colored

class hb_calibration:
    
    def __init__(self, grid, outputs, debug):
        self.grid = grid
        self.debug = debug
        self.outputs = outputs
        self.delta_p_f_ = []
        self.f_dach = []
        self.fit_correct_per_bin = []


    def calculate_scores(self, is_fit, probs, label, correct_per_bin, correct_count, num_samples, total_per_bin, average_bin_confidence):
        if is_fit:
            prefix = 'Fit'
            color = 'cyan'
        else:
            prefix = 'Predict'
            color = 'red'
        
        # Calculate scores for Data
        ece = calibration_scores.ece(correct_per_bin, average_bin_confidence, total_per_bin, num_samples)
        if self.outputs: print(f"{colored(prefix, color)} ECE: {ece}")

        brier_actual = calibration_scores.brier_actual(probs, label, num_samples)
        if self.outputs: print(f"{colored(prefix, color)} Brier actual: {brier_actual}")

        p_r, brier_ref = calibration_scores.brier_ref(correct_count, num_samples)
        if self.outputs: print(f"{colored(prefix, color)} Baseline: {p_r}")
        if self.outputs: print(f"{colored(prefix, color)} Brier ref: {brier_ref}")

        skill_score = calibration_scores.skill_score(brier_ref, brier_actual)
        if self.outputs: print(f"{colored(prefix, color)} Skill Score: {skill_score}\n") 

        results =   {
                        prefix: {
                            "ECE": ece,
                            "Brier actual": brier_actual,
                            "Brier ref": brier_ref,
                            "Skill Score": skill_score
                        } 
                    }  

        return results

 
    def fit(self, X, y):
        num_samples = len(X)
        correct_count = np.count_nonzero(y == 1)
        train_bin_assignments = binning.round_model_to_grid(X, self.grid)

        # Calculated the mean correcteness of the assigned bins
        if self.debug: print(f"TRAIN Assigned Bins: {train_bin_assignments}")
        total_per_bin, fit_correct_per_bin, average_bin_confidence = binning.bin_round_probabilities(train_bin_assignments, X, y, self.grid)

        if self.debug: print(f"Uniform Grid: {self.grid}")
        if self.debug: print(f"TRAIN Correct per bin: {fit_correct_per_bin}") 

        scores = self.calculate_scores(True, X, y, fit_correct_per_bin, correct_count, num_samples, total_per_bin, average_bin_confidence)

        # Calculate correcteness bias in the given bin
        self.delta_p_f_ =  self.grid - fit_correct_per_bin

        if self.outputs: print(f"{colored('Deltas', 'green')}: {self.delta_p_f_}\n")

        return fit_correct_per_bin, scores

    def predict(self, X, y):
        num_samples = len(X)
        correct_count = np.count_nonzero(y == 1)

        if self.debug: print(f"TEST Preditions: {X}")    

        bin_assignment = binning.round_model_to_grid(X, self.grid)
        if self.debug: print(f"TEST Assigned Bins: {bin_assignment}")

        total_per_bin, correct_per_bin, average_bin_confidence = binning.bin_round_probabilities(bin_assignment, X, y, self.grid)

        if self.debug: print(f"TEST Correct per bin: {correct_per_bin}")

        scores = self.calculate_scores(False, X, y, correct_per_bin, correct_count, num_samples, total_per_bin, average_bin_confidence)

        f_dach = np.clip(correct_per_bin + self.delta_p_f_, 0, 1)
        if self.debug: print(f"TEST Corrected Values: {f_dach}")

        return f_dach, scores