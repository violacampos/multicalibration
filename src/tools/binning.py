import numpy as np

class Binning:

    def __init__(self, num_bins, binning_type='linear'):
        """
        Initializes a binning object with the specified grid.

        :param grid: The grid points for binning
        """
        self.grid = self.init_grid(num_bins, binning_type)
        

   

    def init_grid(self, num_bins:int, bin_type:str) -> np.array:
        """
        Initialize the grid for the specified binning type.

        :param num_bins: Number of bins
        :param bin_type: Type of binning ('linear' or 'quantile') TODO check quantiles

        :return: np.array of grid points
        """
        if bin_type == 'linear':
            return self.create_uniform_grid(num_bins)
        else:
            raise ValueError(f"Unknown binning type: {bin_type}")

    def round_probabilities_to_grid(self, probs):
        """
        Discretize probabilities by mapping each to the nearest grid point.
        
        Args:
            probs: List of probabilities
            grid: List of grid points
        
        Returns:
            np.array: Discretized probabilities mapped to grid points
        """
        bin_assignments = []
        
        for prob in probs:
            # Find the nearest grid point
            nearest_idx = np.argmin(np.abs(prob - self.grid))
            bin_assignments.append(self.grid[nearest_idx])
        
        return np.array(bin_assignments)


    @staticmethod
    def create_uniform_grid(num_bins):
        """
        Create a uniform grid with the specified number of bins.
        
        Args:
            num_bins: Number of grid points
        
        Returns:
            np.array: Uniform grid points from 0.0 to 1.0
        """
        step_size = 1 / num_bins
        decimal_str = str(step_size)
        decimal_places = len(decimal_str) - 2  # Subtract "0."
        
        if decimal_places > 10:
            return np.linspace(0.0, 1.0, num_bins + 1)
        else:
            # Round to avoid floating point precision issues
            return np.round(np.arange(0.0, 1.0 + step_size, step_size), decimal_places)


    

