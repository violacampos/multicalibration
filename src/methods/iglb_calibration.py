import numpy as np

from scipy.special import logit, expit
from scipy.optimize import minimize

from tools.calibration_scores import Score


class IGLB_calibration:

    def __init__(self, bins, args):
        """
        Initializes an iterative group linear binning object

        :param bins: used bins for calibration
        :param args: passed commandline parameter
        """
        self.bins = bins
        self.epsilon = args.epsilon
        self.debug = args.debug
        self.m = args.bin_count
        self.outputs = args.print_info
        self.score_obj = Score(bins, args)

        # Store calibration steps for reproducibility
        self.calibration_steps = []
        self.is_fitted = False

    def fit(self, X_train, y_train, groups_train, X_val, y_val, groups_val):
        """
        Learns calibration transformations iteratively on training data,
        using validation data for early stopping.

        :param X_train: Training probabilities
        :param y_train: Training labels
        :param groups_train: Training group matrix
        :param X_val: Validation probabilities
        :param y_val: Validation labels
        :param groups_val: Validation group matrix
        :return: self
        """
        # Reset calibration steps
        self.calibration_steps = []
        
        # Work on copies to avoid modifying original data
        train_probs = X_train.copy()
        val_probs = X_val.copy()
        
        iteration = 0
        while True:
            if self.debug:
                print(f"\n=== Iteration {iteration} ===")
            
            # Calculate MSE on validation set before this step
            mse_before = self._calculate_mse(val_probs, y_val)
            
            # Calculate deltas on current training probabilities
            deltas = self._get_deltas(train_probs, y_train, groups_train)
            deltas_square = deltas ** 2
            
            # Calculate probability of each tau-bin-group combination
            P_S_p_g = self._get_P_S_p_g(train_probs, groups_train)
            
            # Find the tau, bin, group combination with maximum weighted error
            tau, bin_idx, group_idx = np.unravel_index(
                (P_S_p_g * deltas_square).argmax(), 
                deltas.shape
            )
            
            if self.debug:
                print(f"Max delta in: Tau {tau}, Bin {bin_idx}, Group {group_idx}")
                print(f"P_S_p_g value: {P_S_p_g[tau, bin_idx, group_idx]}")
            
            # Stopping criterion 1: probability threshold
            if P_S_p_g[tau, bin_idx, group_idx] < self.epsilon:
                if self.debug:
                    print(f"Stopping: P_S_p_g < epsilon ({self.epsilon})")
                break
            
            # Calculate linear scaling parameters for this subset
            alpha, beta = self._get_linear_scaling_params(
                train_probs, y_train, groups_train, tau, bin_idx, group_idx
            )
            
            # Apply transformation to training data
            train_probs = self._apply_transformation(
                train_probs, groups_train, tau, bin_idx, group_idx, alpha, beta
            )
            
            # Apply transformation to validation data
            val_probs = self._apply_transformation(
                val_probs, groups_val, tau, bin_idx, group_idx, alpha, beta
            )
            
            # Calculate MSE on validation set after this step
            mse_after = self._calculate_mse(val_probs, y_val)
            
            if self.debug:
                print(f"MSE before: {mse_before:.6f}, MSE after: {mse_after:.6f}")
            
            # Stopping criterion 2: MSE increase
            if mse_after >= mse_before:
                if self.debug:
                    print(f"Stopping: MSE increased from {mse_before:.6f} to {mse_after:.6f}")
                break
            
            # Save this calibration step
            step = {
                'tau': tau,
                'bin_idx': bin_idx,
                'group_idx': group_idx,
                'alpha': alpha,
                'beta': beta,
                'bin_threshold': self.bins.grid[bin_idx]
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
                step['tau'],
                step['bin_idx'],
                step['group_idx'],
                step['alpha'],
                step['beta']
            )
        
        return calibrated_probs

    def fit_transform(self, X_train, y_train, groups_train, X_val, y_val, groups_val):
        """
        Fit on training data and return calibrated training probabilities.

        :param X_train: Training probabilities
        :param y_train: Training labels
        :param groups_train: Training group matrix
        :param X_val: Validation probabilities
        :param y_val: Validation labels
        :param groups_val: Validation group matrix
        :return: Calibrated training probabilities
        """
        self.fit(X_train, y_train, groups_train, X_val, y_val, groups_val)
        return self.transform(X_train, groups_train)

    def _get_deltas(self, X, y, groups):
        """
        Calculates the deltas for different tau-bin-group combinations.

        :param X: Probabilities
        :param y: Labels
        :param groups: Group matrix
        :return: 3D delta array
        """
        # Calculate correctness bias for <= threshold
        deltas_smaller = [
            [
                np.mean(y[(X <= threshold) & (g == 1)] - X[(X <= threshold) & (g == 1)])
                if np.any((X <= threshold) & (g == 1)) else 0.0
                for g in groups.T
            ]
            for threshold in self.bins.grid
        ]

        # Calculate correctness bias for >= threshold
        deltas_greater = [
            [
                np.mean(y[(X >= threshold) & (g == 1)] - X[(X >= threshold) & (g == 1)])
                if np.any((X >= threshold) & (g == 1)) else 0.0
                for g in groups.T
            ]
            for threshold in self.bins.grid
        ]

        # Stack: index 0 is <=, index 1 is >=
        deltas = np.stack([np.array(deltas_smaller), np.array(deltas_greater)])
        deltas[np.isnan(deltas)] = 0

        return deltas

    def _get_P_S_p_g(self, probs, groups):
        """
        Calculates the probability that a sample is in different tau-bin-group combinations.

        :param probs: Probabilities
        :param groups: Group matrix
        :return: 3D probability array
        """
        total_samples = len(probs)
        
        # Probability for <= threshold
        P_S_p_g_smaller = [
            [
                np.sum((probs <= threshold) & (g == 1)) / total_samples
                for g in groups.T
            ]
            for threshold in self.bins.grid
        ]

        # Probability for >= threshold
        P_S_p_g_greater = [
            [
                np.sum((probs >= threshold) & (g == 1)) / total_samples
                for g in groups.T
            ]
            for threshold in self.bins.grid
        ]

        P_S_p_g = np.stack([np.array(P_S_p_g_smaller), np.array(P_S_p_g_greater)])
        P_S_p_g[np.isnan(P_S_p_g)] = 0

        return P_S_p_g

    def _get_linear_scaling_params(self, X, y, groups, tau, bin_idx, group_idx):
        """
        Calculate linear scaling parameters for a specific subset.

        :param X: Probabilities
        :param y: Labels
        :param groups: Group matrix
        :param tau: Direction (0 for <=, 1 for >=)
        :param bin_idx: Bin index
        :param group_idx: Group index
        :return: (alpha, beta) tuple
        """
        threshold = self.bins.grid[bin_idx]
        group_mask = groups[:, group_idx] == 1
        
        if tau == 0:
            mask = (X <= threshold) & group_mask
        else:
            mask = (X >= threshold) & group_mask
        
        if not np.any(mask):
            return 0.0, 1.0
        
        subset_X = X[mask]
        subset_y = y[mask]
        
        return self._linear_scaling(subset_X, subset_y)

    def _linear_scaling(self, X, y):
        """
        Learn alpha and beta for linear scaling in logit space.

        :param X: Probabilities
        :param y: Labels
        :return: (alpha, beta) tuple
        """
        if len(X) == 0:
            return 0.0, 1.0
        
        # Clip to avoid infinite logits
        X_clipped = np.clip(X, 1e-10, 1 - 1e-10)
        logit_X = logit(X_clipped)

        def mse(params):
            alpha, beta = params
            transformed = expit(alpha + beta * logit_X)
            return np.mean((transformed - y) ** 2)

        result = minimize(mse, x0=[0, 1], method='BFGS')
        alpha, beta = result.x

        return alpha, beta

    def _apply_transformation(self, X, groups, tau, bin_idx, group_idx, alpha, beta):
        """
        Apply a single calibration transformation.

        :param X: Probabilities
        :param groups: Group matrix
        :param tau: Direction (0 for <=, 1 for >=)
        :param bin_idx: Bin index
        :param group_idx: Group index
        :param alpha: Linear scaling parameter
        :param beta: Linear scaling parameter
        :return: Transformed probabilities
        """
        threshold = self.bins.grid[bin_idx]
        X_new = X.copy()
        
        # Clip to avoid infinite logits
        X_clipped = np.clip(X, 1e-10, 1 - 1e-10)
        
        for idx in range(len(X)):
            group_match = groups[idx, group_idx] == 1
            
            if tau == 0:
                threshold_match = X[idx] <= threshold
            else:
                threshold_match = X[idx] >= threshold
            
            if group_match and threshold_match:
                X_new[idx] = expit(alpha + beta * logit(X_clipped[idx]))
        
        return X_new

    def _calculate_mse(self, X, y):
        """
        Calculate mean squared error.

        :param X: Probabilities
        :param y: Labels
        :return: MSE value
        """
        return np.mean((X - y) ** 2)