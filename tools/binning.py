import numpy as np
from tools.create_charts import chart_creator
import decimal

def get_grid_and_chartmaker(run, binning_type, save_dir, m, extern, probs=None, binning_step_size=None):
    """
        Creates a grid and chart object for the selected binning method.

        :param run: name of the calibration run
        :param binning_type: binning type to use
        :param save_dir: save directory for charts and scores
        :param extern: is the method called from an external method (comparison)
        :param probs: list of probabilities
        :param binning_step_size: size of a bin

        :return: grid, chartmaker
    """
    if binning_type == 'linear':
        # uniform grid 1/m
        grid = create_unform_grid(m)

        # only create chart object for direct usage of a calibration method
        if not extern:
            chartmaker = chart_creator(run, binning_type, grid, save_dir, m)
    elif binning_type == 'quantil':
        # get quantils for step size n
        bin_edges = create_qunatil_grid(probs, binning_step_size)
        # get the middle of the bins for hb
        grid = np.array(((bin_edges[1:]-bin_edges[:-1])/2)+bin_edges[:-1]) 
        if not extern:
            chartmaker = chart_creator(run, binning_type, grid, save_dir, bin_edges=bin_edges)
    
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
    return np.round(np.arange(0.0, 1+(1/m), 1/m), round_to) 

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
        Calculates the closest grid point for every probabaility and assigns the probabaility ot the
        selecte grid point.

        :param probs: List of probailities
        :param grid: list of grid points

        :return: list of disctreized probabilities
    """
    bin_assignment = []    

    for f_x in probs:             
        bin_assignment.append(grid[np.argmin(np.abs(f_x - grid))])         
    
    return np.array(bin_assignment)


"""
    Calculates the needed values for a given bin ranges and assigns the value to the bin if it lies between
"""
"""def bin_range_probabilities(bin_ranges, probs, is_correct):

    # convert bin edges to np array
    bin_ranges = np.array(bin_ranges)

    # calculate the middle of the bin for the bar chart diagramm
    chart_range = np.array(((bin_ranges[1:]-bin_ranges[:-1])/2)+bin_ranges[:-1])

    # calculate the bar width for graphical purpose
    bar_width = np.array(bin_ranges[1:] - bin_ranges[:-1])

    # calculate the total count per bin
    total_per_bin, _ = np.histogram(probs, bin_ranges)

    # sum the probabilities per bin
    # if we reach the last bin, we include the upper edge
    bin_sums = np.array([probs[(probs >= bin_ranges[i]) & (probs < bin_ranges[i + 1] if i < len(bin_ranges) - 1 else (probs <= bin_ranges[i + 1]))].sum() for i in range(len(bin_ranges) - 1)])

    if np.round(bin_sums.sum(), 2) != np.round(probs.sum(), 2):
        exit("Error while Binning. Sums dont match!")

    # calculate the total correct per bin
    correct_per_bin, _ = np.histogram(probs[is_correct == 1], bin_ranges)

    # calculate the average confidence per bin
    average_bin_confidence = np.divide(bin_sums, total_per_bin, where=np.array(total_per_bin)!=0)

    return total_per_bin, correct_per_bin, average_bin_confidence, chart_range, bar_width"""

"""
    Calculates the bin probabilities with a list of assigned bins
"""
"""def bin_round_probabilities(assigend_bins, probs, is_correct, grid):
    
    # calculate the total correct per bin
    correct_per_bin = np.array([np.divide(len(probs[(assigend_bins == i) & (is_correct == 1)]), len(probs[(assigend_bins == i)])) for i in grid])
    correct_per_bin[np.isnan(correct_per_bin)] = 0
    
    # calculate the total count per bin
    total_per_bin = np.array([len(probs[(assigend_bins == i)]) for i in grid])
    total_per_bin[np.isnan(total_per_bin)] = 0
    
    # sum the probabilities per bin
    bin_sums = np.array([probs[assigend_bins == i].sum() for i in grid])

    # calculate the average confidence per bin
    average_bin_confidence = np.divide(bin_sums, total_per_bin, where=np.array(total_per_bin)!=0)

    return total_per_bin, correct_per_bin, average_bin_confidence"""


