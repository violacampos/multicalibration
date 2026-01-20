from argparse import Namespace
import os
import numpy as np
from typing import Any
from numpy.typing import NDArray
from termcolor import colored
from tabulate import tabulate
from tools import binning


class Score:
    """
    Class for calculating calibration scores and metrics.
    
    Supports various calibration metrics including ECE, ASCE, MSE, 
    Brier score, skill score, and group-aware metrics.
    """
    
    def __init__(self, bins: binning.Binning, args: Namespace):
        """
        Initialize the Score Class

        Args:
            bins: Binning object with grid for calculating scores
            args: Configuration arguments containing debug and print_info flags

        """
        self.debug = args.debug
        self.outputs = args.print_info
        self.bins = bins
        self.p_r = 0
        self.brier_ref_score = 0

        self.score_table = []
        self.printable_table = ""

    def ece(self, correctness_per_bin, confidence_per_bin, total_per_bin, num_samples):
        """
        Calculate Expected Calibration Error (ECE).
        
        Weighted average of the absolute deviation between the fraction of 
        predictions that are correct and the average estimated probability.
        
        Args:
            correctness_per_bin: Probability of correctness per bin
            confidence_per_bin: Average confidence of the model per bin
            total_per_bin: Count of samples per bin
            num_samples: Total count of samples in the dataset
        
        Returns:
            ECE score
        """
        ece = 0
        for corr, conf, count in zip(correctness_per_bin, confidence_per_bin, total_per_bin):
            ece += (abs(count) / abs(num_samples)) * abs(corr - conf)
        return ece
        


    def ece_not_rounded(self, labels: NDArray[Any], confidences: NDArray[Any]) -> float:

        """
        Calculate ECE without rounding probabilities to grid.
        
        Args:
            labels: Ground truth labels (0 or 1)
            confidences: Predicted probabilities
        
        Returns:
            ECE score
        """
        n_bins = len(self.bins.grid) - 1
        n = len(labels)
        bin_indices = np.digitize(confidences, self.bins.grid) - 1
        
        cumulative_error = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if np.any(mask):
                bin_count = np.sum(mask)
                prob_avg = np.mean(confidences[mask])
                true_avg = np.mean(labels[mask])
                cumulative_error += (bin_count / n) * abs(prob_avg - true_avg)
        
        return cumulative_error
    


    def asce(
        self, correctness_per_bin, confidence_per_bin, total_per_bin, num_samples
    ):
        """
        Calculate Average Squared Calibration Error (ASCE).
        
        Weighted average of the squared deviation between the fraction of 
        predictions that are correct and the average estimated probability.
        
        Args:
            correctness_per_bin: Probability of correctness per bin
            confidence_per_bin: Average confidence of the model per bin
            total_per_bin: Count of samples per bin
            num_samples: Total count of samples in the dataset
        
        Returns:
            ASCE score
        """
        asce = 0
        for corr, conf, count in zip(correctness_per_bin, confidence_per_bin, total_per_bin):
            asce += (count / num_samples) * (corr - conf) ** 2
        return asce


    def asce_not_rounded(self, labels: NDArray[Any], confidences: NDArray[Any]) -> float:
        """
        Calculate ASCE without rounding probabilities to grid.
        
        Args:
            labels: Ground truth labels (0 or 1)
            confidences: Predicted probabilities
        
        Returns:
            ASCE score
        """
        n_bins = len(self.bins.grid) - 1
        n = len(labels)
        bin_indices = np.digitize(confidences, self.bins.grid) - 1
        
        asce = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if np.any(mask):
                bin_count = np.sum(mask)
                prob_avg = np.mean(confidences[mask])
                true_avg = np.mean(labels[mask])
                asce += (bin_count / n) * (prob_avg - true_avg) ** 2
        
        return asce
        


    def brier_ref(self, correct_sample_count, num_samples):
        """
        Calculate the baseline Brier score of the naive estimator.
        
        The naive estimator puts every prediction into one bin, where p_r 
        is the average correctness.
        
        Args:
            correct_sample_count: Number of correct samples in the dataset
            num_samples: Total count of samples in the dataset
        
        Returns:
            tuple: (p_r, brier_ref) - base rate and reference Brier score
        """
        p_r = correct_sample_count / num_samples
        return p_r, p_r * (1 - p_r)
        

    def mse(self, confidences, labels, num_problems):
        """
        Calculate the Mean Squared Error (Brier score).
        
        Args:
            confidences: List of prediction probabilities
            labels: List of labels indicating if the sample is correct
            num_samples: Total number of samples
        
        Returns:
            MSE score
        """
        brier_score = np.mean((labels - confidences) ** 2)
        return brier_score

    @staticmethod
    def bss(confidences:np.ndarray, labels:np.ndarray) -> float:
        """
        Calculate the Brier Skill Score (BSS).

        Baseline score is 0.0. Negative scores show deterioration, 
        positive scores show improvement over the baseline score,  
        BSS=1.0 is a perfect prediction.
        
        Args:
            confidences: List of prediction probabilities
            labels: List of labels indicating correctness
        
        Returns:
            BSS score
        """
        brier_score = np.mean((labels - confidences) ** 2)
        base_rate = np.mean(labels)
        brier_ref = base_rate * (1 - base_rate)
        if brier_ref == 0:
            return 1.0 if brier_score == 0 else -np.inf
        return float((brier_ref - brier_score) / brier_ref)
        


    def skill_score(self, brier_ref, brier_actual):
        """
        Calculate the skill score.

        Baseline score is 0.0. Negative scores indicate deterioration, 
        positive scores indicate improvement.
        
        Args:
            brier_ref: Brier baseline score
            brier_actual: Actual Brier score of the dataset
        
        Returns:
            Skill score
        """
        return (brier_ref - brier_actual) / brier_ref


    

    def gasce(self, assigned_bins, labels, groups, grid=None):
        """
        Calculate Group Average Squared Calibration Error (GASCE).
        
        Args:
            assigned_bins: Assigned bins for each probability
            labels: List of labels
            groups: 2D array of group assignments
            grid: List of grid points (uses self.bins.grid if None)
        
        Returns:
            Array of GASCE values per group
        """
        if grid is None:
            grid = self.bins.grid
            
        n_per_group = np.sum(groups, axis=0)
        
        # Calculate correctness per bin and group
        correct_per_bin_group = np.array([
            [
                self._safe_divide(
                    len(assigned_bins[(np.round(assigned_bins, 3) == np.round(i, 3)) & 
                                     (labels == 1) & (g == 1)]),
                    len(assigned_bins[(np.round(assigned_bins, 3) == np.round(i, 3)) & (g == 1)])
                )
                for g in groups.T
            ]
            for i in grid
        ])
        correct_per_bin_group = np.nan_to_num(correct_per_bin_group)
        
        # Calculate total per bin and group
        total_per_bin_group = np.array([
            [len(assigned_bins[(np.round(assigned_bins, 3) == np.round(i, 3)) & (g == 1)]) 
             for g in groups.T]
            for i in grid
        ])
        
        # Calculate bin sums per group
        bin_sums_group = np.array([
            [assigned_bins[(np.round(assigned_bins, 3) == np.round(i, 3)) & (g == 1)].sum() 
             for g in groups.T]
            for i in grid
        ])
        
        # Calculate average confidence per bin and group
        average_bin_group_confidence = np.divide(
            bin_sums_group,
            total_per_bin_group,
            where=total_per_bin_group != 0,
            out=np.zeros_like(bin_sums_group, dtype=float)
        )
        
        # Calculate GASCE
        gasce = np.sum(
            (total_per_bin_group / n_per_group) * 
            (correct_per_bin_group - average_bin_group_confidence) ** 2,
            axis=0
        )
        
        return gasce


    def gasce_not_rounded(
        self, confidences: NDArray[Any], labels: NDArray[Any], groups: NDArray[Any]
    ) -> NDArray[Any]:
        """
        Calculate GASCE without rounding probabilities to grid.
        
        Args:
            confidences: Predicted probabilities
            labels: Ground truth labels
            groups: 2D array of group assignments
        
        Returns:
            Array of GASCE values per group
        """
        n_bins = len(self.bins.grid) - 1
        n_per_group = np.sum(groups, axis=0)
        bin_indices = np.digitize(confidences, self.bins.grid) - 1
        n_groups = groups.shape[1]
        
        gasce = np.zeros(n_groups)
        
        for g in range(n_groups):
            g_mask = groups[:, g] == 1
            for i in range(n_bins):
                bin_mask = bin_indices == i
                mask = g_mask & bin_mask
                if np.any(mask):
                    bin_count = np.sum(mask)
                    prob_avg = np.mean(confidences[mask])
                    true_avg = np.mean(labels[mask])
                    gasce[g] += (bin_count / n_per_group[g]) * (prob_avg - true_avg) ** 2
        
        return gasce


    def calc_all(self, confidences, labels, groups=None, set_brier_ref=False):
        """
        Calculate all calibration scores for the given data.
        
        Args:
            confidences: List of probabilities
            labels: List of labels
            groups: 2D array of group assignments (optional)
            set_brier_ref: Flag to set the Brier reference score initially
        
        Returns:
            Dictionary containing all calculated scores
        """
        prefix = "Uncalib" if set_brier_ref else "Calib"
        color = "red" if set_brier_ref else "green"
        
        # Discretize probabilities to bins
        assigned_bins = self.bins.round_probabilities_to_grid(confidences)
        
        # Calculate bin statistics
        total_per_bin, correctness_per_bin, confidence_per_bin = (
            self._compute_bin_statistics(assigned_bins, labels, self.bins.grid)
        )
        
        num_samples = len(assigned_bins)
        num_correct = sum(labels)
        
        # Calculate metrics
        ece = self.ece(correctness_per_bin, confidence_per_bin, total_per_bin, num_samples)
        ece_not_rounded = self.ece_not_rounded(labels, confidences)
        mse = self.mse(confidences, labels, num_samples)
        asce = self.asce(correctness_per_bin, confidence_per_bin, total_per_bin, num_samples)
        asce_not_rounded = self.asce_not_rounded(labels, confidences)
        
        # Calculate accuracy
        y_pred = (confidences > 0.5).astype(int)
        acc = np.mean(y_pred == labels)
        
        # Print results if enabled
        if self.outputs:
            self._print_metrics(prefix, color, ece, ece_not_rounded, mse, 
                              asce, asce_not_rounded, acc)
        
        # Set reference score if needed
        if set_brier_ref:
            self.p_r, self.brier_ref_score = self.brier_ref(num_correct, num_samples)
            if self.outputs:
                print(f"{colored(prefix, color)} Base rate (p(correct)): {self.p_r}")
                print(f"{colored(prefix, color)} Brier ref: {self.brier_ref_score}")
        
        # Calculate skill score
        skill_score = self.skill_score(self.brier_ref_score, mse)
        bss = self.bss(confidences, labels)
        if self.outputs:
            print(f"{colored(prefix, color)} Skill Score: {skill_score}")
            print(f"{colored(prefix, color)} BSS: {bss}")
        
        # Build results dictionary
        results = {
            prefix: {
                "ECE": ece,
                "ASCE": asce,
                "MSE": mse,
                "Brier ref": self.brier_ref_score,
                "Skill Score": skill_score,
                "ACC": acc,
            }
        }
        
        # Calculate and add GASCE if groups are provided
        if groups is not None:
            gasce = self.gasce(assigned_bins, labels, groups)
            gasce_not_rounded = self.gasce_not_rounded(confidences, labels, groups)
            if self.outputs:
                print(f"{colored(prefix, color)} GASCE: {np.round(gasce, 3)}")
                print(f"{colored(prefix, color)} GASCE (not rounded): {np.round(gasce_not_rounded, 3)}")
            results[prefix]["GASCE"] = np.round(gasce, 3)
        
        return results


    def _compute_bin_statistics(self, assigned_bins, is_correct, grid):
        """
        Calculate bin probabilities with discretized values.
        
        Args:
            assigned_bins: List of discretized probabilities
            is_correct: List of labels
            grid: List of grid points
        
        Returns:
            tuple: (total_per_bin, correctness_per_bin, average_confidence_per_bin)
        """
        # Calculate correctness per bin
        correct_per_bin = np.array([
            self._safe_divide(
                len(assigned_bins[(np.round(assigned_bins, 2) == np.round(i, 2)) & 
                                 (is_correct == 1)]),
                len(assigned_bins[(np.round(assigned_bins, 2) == np.round(i, 2))])
            )
            for i in grid
        ])
        correct_per_bin[np.isnan(correct_per_bin)] = 0
        
        # Calculate total per bin
        total_per_bin = np.array([
            len(assigned_bins[(np.round(assigned_bins, 2) == np.round(i, 2))])
            for i in grid
        ])
        
        # Calculate bin sums
        bin_sums = np.array([
            assigned_bins[np.round(assigned_bins, 2) == np.round(i, 2)].sum()
            for i in grid
        ])
        
        # Calculate average confidence per bin
        average_bin_confidence = np.divide(
            bin_sums, 
            total_per_bin, 
            where=total_per_bin != 0,
            out=np.zeros_like(bin_sums, dtype=float)
        )
        
        return total_per_bin, correct_per_bin, average_bin_confidence

    

    def get_total_and_correctness(self, confidences, labels, groups):
        """
        Calculate total and correctness values per bin and per group.
        
        Args:
            confidences: List of probabilities
            labels: List of labels
            groups: 2D array of group assignments
        
        Returns:
            tuple: (total_per_group, correctness_per_group, 
                   total_per_bin, correctness_per_bin)
        """
        assigned_bins = self.bins.round_probabilities_to_grid(confidences)
        
        # Correctness per bin and group
        correctness_group = np.array([
            [
                self._safe_divide(
                    len(labels[(np.round(assigned_bins, 3) == np.round(i, 3)) & 
                              (labels == 1) & (g == 1)]),
                    len(labels[(np.round(assigned_bins, 3) == np.round(i, 3)) & (g == 1)])
                )
                for g in groups.T
            ]
            for i in self.bins.grid
        ])
        correctness_group[np.isnan(correctness_group)] = 0
        
        # Total per bin and group
        total_group = np.array([
            [len(assigned_bins[(np.round(assigned_bins, 3) == np.round(i, 3)) & (g == 1)]) 
             for g in groups.T]
            for i in self.bins.grid
        ])
        
        # Correctness per bin (overall)
        correctness_bin = np.array([
            self._safe_divide(
                len(labels[(np.round(assigned_bins, 3) == np.round(i, 3)) & (labels == 1)]),
                len(labels[(np.round(assigned_bins, 3) == np.round(i, 3))])
            )
            for i in self.bins.grid
        ])
        correctness_bin[np.isnan(correctness_bin)] = 0
        
        # Total per bin (overall)
        total_bin = np.array([
            len(assigned_bins[(np.round(assigned_bins, 3) == np.round(i, 3))]) 
            for i in self.bins.grid
        ])
        
        return total_group, correctness_group, total_bin, correctness_bin
        

    def get_correctness_per_group(self, confidences, labels, groups):
        """
        Calculate total, correctness and average confidence per group.
        
        Args:
            confidences: List of probabilities
            labels: List of labels
            groups: 2D array of group assignments
        
        Returns:
            tuple: (total_per_group, correctness_per_group, average_confidence_per_group)
        """
        # Correctness per group
        correctness_group = np.array([
            self._safe_divide(
                len(labels[(labels == 1) & (g == 1)]), 
                len(labels[(g == 1)])
            )
            for g in groups.T
        ])
        correctness_group[np.isnan(correctness_group)] = 0
        
        # Total per group
        total_group = np.array([len(confidences[(g == 1)]) for g in groups.T])
        
        # Average confidence per group
        bin_sums_group = np.array([confidences[(g == 1)].sum() for g in groups.T])
        average_group_confidence = np.divide(
            bin_sums_group, 
            total_group, 
            where=total_group != 0,
            out=np.zeros_like(bin_sums_group, dtype=float)
        )
        
        return total_group, correctness_group, average_group_confidence
        

    


    def _format_table(self):
        self.printable_table = tabulate(
            self.score_table,
            headers=["Run", "Type", "ECE", "ASCE", "MSE", "brier_ref", 
                    "skill_score", "ACC", "GASCE"],
            tablefmt="orgtbl",
        )
    

    @staticmethod
    def _safe_divide(numerator, denominator):
        """Safely divide, returning 0 if denominator is 0."""
        return numerator / denominator if denominator != 0 else 0

    def _print_metrics(self, prefix, color, ece, ece_not_rounded, mse, 
                      asce, asce_not_rounded, acc):
        """Print all calculated metrics with color formatting."""
        print(f"{colored(prefix, color)} ECE: {ece}")
        print(f"{colored(prefix, color)} ECE (not rounded): {ece_not_rounded}")
        print(f"{colored(prefix, color)} MSE: {mse}")
        print(f"{colored(prefix, color)} ASCE: {asce}")
        print(f"{colored(prefix, color)} ASCE (not rounded): {asce_not_rounded}")
        print(f"{colored(prefix, color)} ACC: {acc}")    