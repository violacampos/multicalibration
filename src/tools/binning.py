import numpy as np
from tools.create_charts import charts

def get_grid_and_chartmaker(run, args, save_dir, extern, probs=None):
    """
        Creates a grid and chart object for the selected binning method.

        :param run: name of the calibration run
        :param binning_type: binning type to use
        :param save_dir: save directory for charts and scores
        :param extern: is the method called from an external method (comparison)
        :param probs: list of probabilities

        :return: grid, chartmaker
    """
    binning_step_size=1/args.bin_count

    if args.binning_type == 'linear':
        # uniform grid 1/m
        grid = create_unform_grid(args.bin_count)

        # only create chart object for direct usage of a calibration method
        if not extern:
            chartmaker = charts(run, args.binning_type, grid, save_dir)
    elif args.binning_type == 'quantil':
        # get quantils for step size n
        bin_edges = create_qunatil_grid(probs, binning_step_size)
        # get the middle of the bins for hb
        grid = np.array(((bin_edges[1:]-bin_edges[:-1])/2)+bin_edges[:-1]) 
        if not extern:
            chartmaker = charts(run, args.binning_type, grid, save_dir, bin_edges=bin_edges)
    
    if extern:
        return grid, None
    else:
        return grid, chartmaker

def create_unform_grid(m):
    """
        Creates a unform grid with the number m.

        :param m: number of grid points

        :return: list of uniform grid points
    """
    d = str(1/m)
    round_to = len(d)-2
    if round_to > 10:
        return np.linspace(0.0, 1.0, m + 1)
    else:
        return np.round(np.arange(0.0, 1+(1/m), 1/m), round_to) # VIOLA: extra bin for p=1.0? somehow weird
    #

def create_qunatil_grid(probs, m):
    """
        Creates a grid for given probabilties and the number of qunatils.

        :param probs: List of probailities
        :param m: number of quantils

        :return: list of quantil grid points
    """
    return np.array([(np.quantile(probs, i) if (i != 0) and (i != 1) else i) for i in np.arange(0, 1+m, m)])  
            

def round_model_to_grid(probs, grid):
    """
        Calculates the closest grid point for every probability and assigns the probability ot the
        selecte grid point.

        :param probs: List of probailities
        :param grid: list of grid points

        :return: list of disctreized probabilities
    """
    bin_assignment = []    

    for f_x in probs:             
        bin_assignment.append(grid[np.argmin(np.abs(f_x - grid))])         
    
    return np.array(bin_assignment)