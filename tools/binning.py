import numpy as np

def create_unform_grid(m):
    return np.round(np.arange(0.0, 1+(1/m), 1/m), 2)   


def round_model_to_grid(probs, grid):
    # Round model to the grid (assign values to bin edges)
    bin_assignment = []    

    for f_x in probs:             
        bin_assignment.append(np.round(grid[np.argmin(np.abs(f_x - grid))], 2))         
    
    return np.array(bin_assignment)

"""
    Calculates the needed values for a given bin ranges and assigns the value to the bin if it lies between
"""
def bin_range_probabilities(bin_ranges, probs, is_correct):

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

    return total_per_bin, correct_per_bin, average_bin_confidence, chart_range, bar_width

"""
    Calculates the bin probabilities with a list of assigned bins
"""
def bin_round_probabilities(assigend_bins, probs, is_correct, grid):
    
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

    return total_per_bin, correct_per_bin, average_bin_confidence