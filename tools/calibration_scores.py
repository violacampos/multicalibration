import numpy as np
import math
from termcolor import colored
from tabulate import tabulate
from tools import binning


class score:
    def __init__(self, grid, outputs, debug):
        """
        Initatlization of the score class

        :param grid: grid to calculate scores on
        :param outputs: controls possible optional outputs
        :param debug: controls possible debug outputs
        """
        self.debug = debug
        self.outputs = outputs
        self.grid = grid
        # self.score_grid = np.arange(0.0, 1+(1/m_score), 1/m_score)
        self.p_r = 0
        self.brier_ref_score = 0

        self.score_table = []
        self.printable_table = []

    def ece(self, correctness_per_bin, confidence_per_bin, total_per_bin, num_samples):
        """
        Calculates the expected calibration error. Weighted average of the absolut deviation from the
        fraction of predictions that are correct and the average estimated probability.

        :param correctness_per_bin: Probality for the correctnes per bin
        :param confidence_per_bin: Average confidence of the model per bin
        :param total_bin_count: Count of samples per bin
        :param num_samples: Total count of samples in the dataset

        :return: ECE score
        """
        ece = 0
        for corr_s_i, conf_s_i, s_i_count in zip(
            correctness_per_bin, confidence_per_bin, total_per_bin
        ):
            ece += (abs(s_i_count) / abs(num_samples)) * abs(corr_s_i - conf_s_i)

        return ece

    def ece_not_rounded(self, labels: np.array, confidences: np.array) -> float:
        n_bins = len(self.grid)
        n = len(labels)
        bin_indices = (
            np.digitize(confidences, self.grid) - 1
        )  # Bin index for each prediction

        # abs_errors = []
        cummulative_error = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if np.any(mask):
                bin_count = np.sum(mask)
                prob_avg = np.mean(confidences[mask])
                true_avg = np.mean(labels[mask])
                # abs_errors.append(abs(prob_avg - true_avg))
                cummulative_error += (bin_count / n) * abs(prob_avg - true_avg)

        # test = np.mean(abs_errors)
        return cummulative_error

    def asce(
        self, correctness_per_bin, confidence_per_bin, total_bin_count, num_samples
    ):
        """
        Calculates the average squared calibration error. Weighted average of the squared deviation from the
        fraction of predictions that are correct and the average estimated probability.

        :param correctness_per_bin: Probality for the correctnes per bin
        :param confidence_per_bin: Average confidence of the model per bin
        :param total_bin_count: Count of samples per bin
        :param num_samples: Total count of samples in the dataset

        :return: ASCE score
        """
        asce = 0
        for corr_s_i, conf_s_i, bin_count in zip(
            correctness_per_bin, confidence_per_bin, total_bin_count
        ):
            asce += (bin_count / num_samples) * (corr_s_i - conf_s_i) ** 2
        return asce

    def asce_not_rounded(self, labels: np.array, confidences: np.array) -> float:
        n_bins = len(self.grid)
        n = len(labels)
        bin_indices = (
            np.digitize(confidences, self.grid) - 1
        )  # Bin index for each prediction
        asce = 0.0
        squared_errors = []
        for i in range(n_bins):
            mask = bin_indices == i
            if np.any(mask):
                bin_count = np.sum(mask)
                prob_avg = np.mean(confidences[mask])
                true_avg = np.mean(labels[mask])
                squared_errors.append((prob_avg - true_avg) ** 2)
                asce += (bin_count / n) * (prob_avg - true_avg) ** 2
        test = np.mean(squared_errors)
        return asce

    def brier_ref(self, correct_sample_count, num_samples):
        """
        Calculates the baseline score of the naive estimator, where every prediction is put into one bin.
        p_r is the average correctness in this bin.

        :param correct_sample_count: Correct samples in the dataset
        :param num_samples: Total count of samples in the dataset

        :return: p_r, brier_ref
        """
        p_r = correct_sample_count / num_samples
        return p_r, p_r * (1 - p_r)

    def mse(self, confidences, labels, num_problems):
        """
        Caculates the actual brier score for the given data.
        Also known as the MSE

        :param confidences: List of all prediction probailities
        :param label: list of label if the given sample is correct
        :param num_problems: Total number of samples

        :return: MSE score
        """
        brier_score_actual = 0
        for conf, label in zip(confidences, labels):
            brier_score_actual += (label - conf) ** 2
        return brier_score_actual / num_problems

    def skill_score(self, brier_ref, brier_actual):
        """
        Caculates the skill score. Perfect score is 1.0. Negativ mean worse than the baseline. Small positiv values indicate good skill

        :param brier_ref: Brier baseline score
        :param brier_actual: Actual brier score of the dataset

        :return: skill score
        """
        return (brier_ref - brier_actual) / brier_ref

    def gcu(self, label, confidence, groups):
        """
        Calculates group conditional unbiasedness

        :param label: List of labels
        :param confidence: List of probabilities
        :param groups: 2D Array of assigned groups

        :return: gcu
        """
        gcu = np.array(
            [
                np.mean(abs(label[(col == 1)] - confidence[(col == 1)]))
                for col in groups.T
            ]
        )  # VIOLA: unused? needs abs()
        gcu[np.isnan(gcu)] = 0
        return gcu

    def expected_variance(self, probs, label, bin_assignement, grid, num_samples):
        """
        Calculates expected variance

        :param probs: List of probabilities
        :param label: List of labels
        :param bin_assignement: Assigned bins for each probability
        :param grid: List of grid points
        :param num_samples: Total of samples

        :return: expected variance -> VIOLA: where do we use it? Depends only on bin weights (and grid size) not the actual probabilities
        """
        expec_var = 0.0
        for i in grid:
            bin_probs = probs[bin_assignement == i]
            bin_labels = label[bin_assignement == i]
            if len(bin_probs) == 0:
                continue

            E = np.mean(
                bin_probs
            )  # VIOLA: useless, probs are already rounded to grid -> expected behaviour?
            if math.isnan(E):
                E = 0

            variance = E * (1 - E) ** 2

            weight = len(bin_labels) / num_samples
            expec_var += weight * variance

        return np.round(expec_var, 3)

    def gasce(self, assigned_bins, labels, groups, grid=None):
        """
        Calculates group average squared calibration error

        :param assigned_bins: Assigned bins for each probability
        :param labels: List of labels
        :param groups: List of probabilities
        :param grid: List of grid points

        :return: List of GASCE
        """
        if grid is None:
            grid = self.grid
        num_samples = len(assigned_bins)
        n_per_group = np.sum(groups, axis=0)

        # calculate the total correct per bin
        correct_per_bin_group = np.array(
            [
                [
                    np.divide(
                        len(
                            assigned_bins[
                                (np.round(assigned_bins, 3) == np.round(i, 3)) & (labels == 1) & (g == 1)
                            ]
                        ),
                        len(assigned_bins[(np.round(assigned_bins, 3) == np.round(i, 3)) & (g == 1)]),
                    )
                    for g in groups.T
                ]
                for i in grid
            ]
        )
        # correct_per_bin_group[np.isnan(correct_per_bin_group)] = 0
        correct_per_bin_group = np.nan_to_num(correct_per_bin_group)

        # calculate the total count per bin
        total_per_bin_group = np.array(
            [
                [len(assigned_bins[(np.round(assigned_bins, 3) == np.round(i, 3)) & (g == 1)]) for g in groups.T]
                for i in grid
            ]
        )
        total_per_bin_group[np.isnan(total_per_bin_group)] = 0

        # sum the probabilities per bin
        bin_sums_group = np.array(
            [
                [assigned_bins[(np.round(assigned_bins, 3) == np.round(i, 3)) & (g == 1)].sum() for g in groups.T]
                for i in grid
            ]
        )

        # calculate the average confidence per bin
        average_bin_group_confidence = np.divide(
            bin_sums_group,
            total_per_bin_group,
            where=np.array(total_per_bin_group) != 0,
        )

        gasce = 0
        for corr_bin_group, conf_bin_group, bin_group_count in zip(
            correct_per_bin_group, average_bin_group_confidence, total_per_bin_group
        ):
            gasce += (bin_group_count / n_per_group) * (
                (corr_bin_group - conf_bin_group) ** 2
            )  # VIOLA: fix weighted sum

        return np.array(gasce)

    def gasce_not_rounded(
        self, confidences: np.array, labels: np.array, groups: np.array
    ) -> np.array:
        n_bins = len(self.grid)
        n_per_group = np.sum(groups, axis=0)
        bin_indices = (
            np.digitize(confidences, self.grid) - 1
        )  # Bin index for each prediction
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
                    gasce[g] += (bin_count / n_per_group[g]) * (
                        prob_avg - true_avg
                    ) ** 2

        return gasce

    def calc_all(self, confidences, labels, groups=None, set_brier_ref=False):
        """
        Calculates every score for the given data and returns a dict with all values

        :param confidences: List of probability
        :param labels: List of labels
        :param groups: List of probabilities
        :param set_brier_ref: Flag to set the brier reference score initially

        :return: Result dict
        """
        if set_brier_ref:
            prefix = "Uncalib"
            color = "red"
        else:
            prefix = "Calib"
            color = "green"

        # Assign the values in X to the corresponding bin (discretize values)
        assigned_bins = binning.round_model_to_grid(confidences, self.grid)

        # Calculate total, correctness and confidence per bin
        # [VIOLA] confidence_per_bin is rounded -> expected behaviour?
        total_per_bin, correctness_per_bin, confidence_per_bin = (
            self.bin_round_probabilities_discret(assigned_bins, labels, self.grid)
        )

        num_samples = len(assigned_bins)
        num_correct = sum(labels)

        # Get Expected Calibration Error
        ece = self.ece(
            correctness_per_bin, confidence_per_bin, total_per_bin, num_samples
        )
        ece_not_rounded = self.ece_not_rounded(labels, confidences)

        # Get Mean Squared Error
        mse = self.mse(
            confidences, labels, num_samples
        )  # VIOLA use original confidences instead of discretized bin values

        # Get Average Squared Error
        asce = self.asce(
            correctness_per_bin, confidence_per_bin, total_per_bin, num_samples
        )

        asce_not_rounded = self.asce_not_rounded(labels, confidences)

        # Get expected Variance
        expected_variance = self.expected_variance(
            assigned_bins, labels, assigned_bins, self.grid, num_samples
        )

        # Output results if set
        if self.outputs:
            print(f"{colored(prefix, color)} ECE: {ece}")
            print(f"{colored(prefix, color)} ECE: {ece_not_rounded} (not rounded)")
            print(f"{colored(prefix, color)} MSE: {mse}")
            print(f"{colored(prefix, color)} ASCE: {asce}")
            print(f"{colored(prefix, color)} ASCE: {asce_not_rounded} (not rounded)")
            print(f"{colored(prefix, color)} Expected Variance: {expected_variance}")

        # Get Group conditional unbiasedness VIOLA: unused?
        #if groups is not None:
        #    gcu = self.gcu(labels, confidences, groups)
        #    if self.outputs:
        #        print(f"{colored(prefix, color)} GCU: {gcu}")

        # Set the reference Score for the Skill Score calculation
        if set_brier_ref:
            self.p_r, self.brier_ref_score = self.brier_ref(num_correct, num_samples)
            if self.outputs:
                print(f"{colored(prefix, color)} Base rate (p(correct)): {self.p_r}")
                print(f"{colored(prefix, color)} Brier ref: {self.brier_ref_score}")

        # Calculate the Skill score
        skill_score = self.skill_score(self.brier_ref_score, mse)
        if self.outputs:
            print(f"{colored(prefix, color)} Skill Score: {skill_score}")

        # create dict for better overview
        results = {
            prefix: {
                "ECE": ece,
                "ASCE": asce,
                "MSE": mse,
                "Brier ref": self.brier_ref_score,
                "Skill Score": skill_score,
            }
        }

        # Calculate GASCE and add to dict
        gasce = self.gasce(assigned_bins, labels, groups)
        gasce_not_rounded = self.gasce_not_rounded(confidences, labels, groups)
        if self.outputs:
            print(f"{colored(prefix, color)} GASCE: {np.round(gasce, 3)}")
            print(
                f"{colored(prefix, color)} GASCE: {np.round(gasce_not_rounded, 3)} (not rounded)"
            )
        results[prefix]["GASCE"] = np.round(gasce, 3)

        return results

    def bin_round_probabilities_discret(self, assigned_bins, is_correct, grid):
        """
        Calculates the bin probabilities with the discretized values

        :param assigned_bins: List of discretized probabilities
        :param is_correct: List of labels
        :param grid: List of grid points

        :return: total per bin, correctness per bin, average confidence per bin
        """
        # calculate the total correct per bin
        correct_per_bin = np.array(
            [
                np.divide(
                    len(
                        assigned_bins[
                            (np.round(assigned_bins, 2) == np.round(i, 2))
                            & (is_correct == 1)
                        ]
                    ),
                    len(assigned_bins[(np.round(assigned_bins, 2) == np.round(i, 2))]),
                )
                for i in grid
            ]
        )
        correct_per_bin[np.isnan(correct_per_bin)] = 0

        # calculate the total count per bin
        total_per_bin = np.array(
            [
                len(assigned_bins[(np.round(assigned_bins, 2) == np.round(i, 2))])
                for i in grid
            ]
        )
        total_per_bin[np.isnan(total_per_bin)] = 0

        # sum the probabilities per bin
        bin_sums = np.array(
            [
                assigned_bins[np.round(assigned_bins, 2) == np.round(i, 2)].sum()
                for i in grid
            ]
        )

        # calculate the average confidence per bin
        average_bin_confidence = np.divide(
            bin_sums, total_per_bin, where=np.array(total_per_bin) != 0
        )

        return total_per_bin, correct_per_bin, average_bin_confidence

    def get_total_and_correctness(self, confidences, labels, groups):
        """
        Calculates the total and correctness values per bin and per group

        :param confidences: List of probabilities
        :param labels: List of labels
        :param groups: 2D array of assigned groups

        :return: total per group, correctness per group, total per bin, correctness per bin
        """
        # Assign the values in X to the corresponding bin (discretize values)
        assigned_bins = binning.round_model_to_grid(confidences, self.grid)

        # fraction of total samples per bin and group
        correctness_group = np.array(
            [
                [
                    np.divide(
                        len(labels[(np.round(assigned_bins, 3) == np.round(i, 3)) & (labels == 1) & (g == 1)]),
                        len(labels[(np.round(assigned_bins, 3) == np.round(i, 3)) & (g == 1)]),
                    )
                    for g in groups.T
                ]
                for i in self.grid
            ]
        )
        correctness_group[np.isnan(correctness_group)] = 0

        total_group = np.array(
            [
                [len(assigned_bins[(np.round(assigned_bins, 3) == np.round(i, 3)) & (g == 1)]) for g in groups.T]
                for i in self.grid
            ]
        )
        total_group[np.isnan(total_group)] = 0

        # fraction of correct samples per bin
        correctness_bin = np.array(
            [
                np.divide(
                    len(labels[(np.round(assigned_bins, 3) == np.round(i, 3)) & (labels == 1)]),
                    len(labels[(np.round(assigned_bins, 3) == np.round(i, 3))]),
                )
                for i in self.grid
            ]
        )
        correctness_bin[np.isnan(correctness_bin)] = 0

        total_bin = np.array(
            [len(assigned_bins[(np.round(assigned_bins, 3) == np.round(i, 3))]) for i in self.grid]
        )
        total_bin[np.isnan(total_bin)] = 0

        return total_group, correctness_group, total_bin, correctness_bin

    def get_correctness_per_group(self, confidences, labels, groups):
        """
        Calculates the total, correctness and average confidence per group

        :param confidences: List of probabilities
        :param labels: List of labels
        :param groups: 2D array of assigned groups

        :return: total per group, correctness per group, average confidence per group
        """
        correctness_group = np.array(
            [
                np.divide(len(labels[(labels == 1) & (g == 1)]), len(labels[(g == 1)]))
                for g in groups.T
            ]
        )
        correctness_group[np.isnan(correctness_group)] = 0

        total_group = np.array([len(confidences[(g == 1)]) for g in groups.T])
        total_group[np.isnan(total_group)] = 0

        # sum the probabilities per bin
        bin_sums_group = np.array([confidences[(g == 1)].sum() for g in groups.T])

        # calculate the average confidence per bin
        average_group_confidence = np.divide(
            bin_sums_group, total_group, where=np.array(total_group) != 0
        )

        return total_group, correctness_group, average_group_confidence

    def add_to_score_table(self, run, uncalib_scores, calib_scores, baseline=False):
        """
        Adds given scores to a printable table in the score obj

        :param run: Name of the run for the scores
        :param uncalib_scores: Dict of scores
        :param calib_scores: Dict of scores
        :param baseline: Flag to only use uncalib scores for the baseline
        """
        uncalib_scores = list(list(uncalib_scores.values())[0].values())
        uncalib_gasce = uncalib_scores[-1]
        uncalib_scores = uncalib_scores[:-1]

        if not baseline:
            calib_scores = list(list(calib_scores.values())[0].values())
            calib_gasce = calib_scores[-1]
            calib_scores = calib_scores[:-1]

            score_difference = list(
                np.round(np.array(calib_scores) - np.array(uncalib_scores), 4)
            )

            gasce_diff = np.round(np.array(calib_gasce) - np.array(uncalib_gasce), 4)

            score_difference.append(gasce_diff)

            calib_scores.append(calib_gasce)

        uncalib_scores.append(uncalib_gasce)
        self.add_entry(run, "Uncalib", uncalib_scores)

        if not baseline:
            self.add_entry(run, "Calib", calib_scores)
            self.add_entry(run, "Diff", score_difference)

    def add_entry(self, run, type, scores):
        """
        Adds given entry to score table

        :param run: Name of the run for the scores
        :param type: Type of score (Uncalib, Calib, Diff)
        :param scores: List of scores
        """
        entry = []

        if type == "Uncalib":
            entry.append(run)
        else:
            entry.append("")
        entry.append(type)
        for s in scores:
            entry.append(s)

        self.score_table.append(entry)

    def display_score_table(self):
        """
        Prints the score table of the score class obj
        """
        self.printable_table = tabulate(
            self.score_table,
            headers=[
                "Run",
                "Type",
                "ECE",
                "ASCE",
                "MSE",
                "brier_ref",
                "skill_score",
                "GASCE",
            ],
            tablefmt="orgtbl",
        )
        print(self.printable_table)
