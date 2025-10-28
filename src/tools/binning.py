import numpy as np
from tools.create_charts import Charts

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


    


def get_grid_and_chartmaker(run, args, save_dir, probs=None):
    """
    Create a grid and chart object for the selected binning method.
    
    Args:
        run: Name of the calibration run
        args: Command line arguments containing binning_type and bin_count
        save_dir: Save directory for charts and scores
        probs: List of probabilities (required for quantile binning)
    
    Returns:
        tuple: (grid, chartmaker) objects for calibration
    """
    binning_step_size = 1 / args.bin_count
    
    if args.binning_type == 'linear':
        grid = create_uniform_grid(args.bin_count)
        chartmaker = Charts(run, args.binning_type, grid, save_dir)
        
    elif args.binning_type == 'quantil':
        bin_edges = create_quantile_grid(probs, binning_step_size)
        # Calculate bin centers
        grid = (bin_edges[1:] + bin_edges[:-1]) / 2
        chartmaker = Charts(run, args.binning_type, grid, save_dir, bin_edges=bin_edges)
    
    return grid, chartmaker

