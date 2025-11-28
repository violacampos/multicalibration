import numpy as np
from tools.calibration_scores import Score


class IGHB_calibration:

    def __init__(self, bins, args):
        """
        Initializes an iterative group histogram binning object

        :param bins: used grid for calibration
        :param args: passed commandline parameter
        """
        self.bins = bins
        self.m = len(bins.grid) - 1
        self.alpha = 1 / self.m
        
        self.debug = args.debug
        self.outputs = args.print_info
        self.score_obj = Score(bins, args)

        # Store calibration steps for reproducibility
        self.calibration_steps = []
        self.is_fitted = False

    def fit(self, X_train, y_train, groups_train):
        """
        Learns calibration transformations iteratively on training data.

        :param X_train: Training probabilities
        :param y_train: Training labels
        :param groups_train: Training group matrix
        :return: self
        """
        # Reset calibration steps
        self.calibration_steps = []
        
        # Work on copies to avoid modifying original data
        train_probs = X_train.copy()
        
        iteration = 0
        while True:
            if self.debug:
                print(f"\n=== Iteration {iteration} ===")
            
            # Calculate deltas and error metrics
            deltas = self._get_deltas(train_probs, y_train, groups_train)
            deltas_square = deltas ** 2
            
            # Calculate probability of each bin-group combination
            counts = self._get_bin_group_counts(train_probs, groups_train)
            total_counts = len(train_probs)
            P_S_p_g = counts / total_counts
            
            # Calculate GASCE (Group-Aware Squared Calibration Error)
            gasce = self.score_obj.gasce_not_rounded(train_probs, y_train, groups_train)
            
            if self.debug:
                print(f"GASCE: {gasce}, sum: {gasce.sum()}")
            
            # Calculate group membership probabilities
            group_counts = groups_train.sum(axis=0)
            p_group = group_counts / len(groups_train)
            
            if self.debug:
                print(f"P(g(X)=1): {p_group}")
            
            # Calculate effect of calibration error for each group
            c = gasce * p_group
            
            if self.debug:
                print(f"gASCE * P(g(X)=1): {c}")
            
            # Get max error for stop condition
            max_error = c[np.argmax(c)]
            
            if self.debug:
                print(f"Max error: {max_error}, alpha threshold: {self.alpha}")
            
            # Stopping criterion: max error below threshold
            if max_error <= self.alpha:
                if self.debug:
                    print(f"Stopping: max_error ({max_error:.6f}) <= alpha ({self.alpha:.6f})")
                break
            
            # Find bin-group combination with maximum weighted error
            bin_idx, group_idx = np.unravel_index(
                (P_S_p_g * deltas_square).argmax(),
                deltas.shape
            )
            
            delta_value = deltas[bin_idx, group_idx]
            
            if self.debug:
                print(f"Max delta in: Bin {bin_idx}, Group {group_idx}")
                print(f"Delta value: {delta_value}")
            
            # Apply transformation to training data
            train_probs = self._apply_transformation(
                train_probs, groups_train, bin_idx, group_idx, delta_value
            )
            
            # Save this calibration step
            step = {
                'bin_idx': bin_idx,
                'group_idx': group_idx,
                'delta': delta_value,
                'bin_edges': (self.bins.grid[bin_idx], self.bins.grid[bin_idx + 1])
            }
            self.calibration_steps.append(step)
            
            if self.debug:
                print(f"Step saved: {step}")
            
            iteration += 1
        
        self.is_fitted = True
        if self.debug:
            print(f"\nCalibration complete after {len(self.calibration_steps)} steps")
        
        return self

    def transform(self, X, groups):
        """
        Apply learned calibration transformations to new data.

        :param X: Probabilities to calibrate
        :param groups: Group matrix
        :return: Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform. Call fit() first.")
        
        calibrated_probs = X.copy()
        
        # Apply each saved calibration step in order
        for step in self.calibration_steps:
            calibrated_probs = self._apply_transformation(
                calibrated_probs,
                groups,
                step['bin_idx'],
                step['group_idx'],
                step['delta']
            )
        
        return calibrated_probs

    def fit_transform(self, X_train, y_train, groups_train):
        """
        Fit on training data and return calibrated training probabilities.

        :param X_train: Training probabilities
        :param y_train: Training labels
        :param groups_train: Training group matrix
        :return: Calibrated training probabilities
        """
        self.fit(X_train, y_train, groups_train)
        return self.transform(X_train, groups_train)

    def _apply_transformation(self, X, groups, bin_idx, group_idx, delta):
        """
        Apply a single calibration transformation.

        :param X: Probabilities
        :param groups: Group matrix
        :param bin_idx: Bin index
        :param group_idx: Group index
        :param delta: Delta value to add
        :return: Transformed probabilities
        """
        X_new = X.copy()
        
        # Assign samples to bins
        bin_indices = np.digitize(X, self.bins.grid) - 1
        
        # Create mask for samples in the target bin and group
        mask = (bin_indices == bin_idx) & (groups[:, group_idx] == 1)
        
        # Apply delta adjustment
        X_new[mask] += delta
        
        # Optionally clip to [0, 1] to ensure valid probabilities
        X_new = np.clip(X_new, 0.0, 1.0)
        
        return X_new

    def _get_bin_group_counts(self, X, groups):
        """
        Returns an array of shape (n_bins, n_groups) with the absolute number of samples
        from X that fall into each bin and are in each group.

        :param X: Probabilities
        :param groups: Group matrix
        :return: 2D array of counts
        """
        bin_indices = np.digitize(X, self.bins.grid) - 1
        n_groups = groups.shape[1]
        counts = np.zeros((self.m, n_groups), dtype=int)

        for i in range(self.m):
            for j in range(n_groups):
                mask = (bin_indices == i) & (groups[:, j] == 1)
                counts[i, j] = np.sum(mask)
        
        return counts

    def _get_correct_per_bin_group_counts(self, X, y, groups):
        """
        Returns an array of shape (n_bins, n_groups) with the absolute number of correct samples
        from X that fall into each bin and are in each group.

        :param X: Probabilities
        :param y: Labels of correctness
        :param groups: Group matrix
        :return: 2D array of counts of correct samples
        """
        bin_indices = np.digitize(X, self.bins.grid) - 1
        n_groups = groups.shape[1]
        counts = np.zeros((self.m, n_groups), dtype=int)

        for i in range(self.m):
            for j in range(n_groups):
                mask = (bin_indices == i) & (y == 1) & (groups[:, j] == 1)
                counts[i, j] = np.sum(mask)
        
        return counts

    def _get_mean_per_bin_group(self, X, groups):
        """
        Returns an array of shape (n_bins, n_groups) with the mean value of X
        for each bin and group.

        :param X: Probabilities
        :param groups: Group matrix
        :return: 2D array of mean values
        """
        bin_indices = np.digitize(X, self.bins.grid) - 1
        n_groups = groups.shape[1]
        means = np.zeros((self.m, n_groups), dtype=float)

        for i in range(self.m):
            for j in range(n_groups):
                mask = (bin_indices == i) & (groups[:, j] == 1)
                if np.any(mask):
                    means[i, j] = X[mask].mean()
                else:
                    means[i, j] = 0.0
        
        return means

    def _get_deltas(self, X, y, groups):
        """
        Calculates the deltas for each bin-group combination.

        :param X: Probabilities
        :param y: Labels of correctness
        :param groups: Group matrix
        :return: 2D Array of deltas (between correctness and likelihood) for each bin-group combination
        """
        counts = self._get_bin_group_counts(X, groups)
        correct_counts = self._get_correct_per_bin_group_counts(X, y, groups)

        # Correctness per bin and group
        correct_per_bin_group = np.divide(
            correct_counts,
            counts,
            out=np.zeros_like(correct_counts, dtype=float),
            where=(counts > 0),
        )
        
        probs_per_bin_group = self._get_mean_per_bin_group(X, groups)

        deltas = correct_per_bin_group - probs_per_bin_group
        
        return deltas