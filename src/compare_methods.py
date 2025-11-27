import os

from tabulate import tabulate


import methods.run_hb as run_hb
import methods.run_lr as run_lr
import methods.run_ighb as run_ighb
import methods.run_iglb as run_iglb
import tools.compute_baseline as compute_baseline
import methods.run_platt as run_platt
from tools import binning, cmd_input
from tools.create_charts import CalibrationCharts
from data.dataset import CalibrationDataset, GroupConfig


def get_benchmark_configs():
    """Define configuration for each benchmark."""
    return {
        "livecodebench": GroupConfig(
            add_counter=False,
            language=False,
            larger_than_median_loc=True,
            larger_than_median_prompt=True,
            larger_than_median_output=True,
            difficulty_easy=True,
            difficulty_medium=True,
            difficulty_hard=True
        ),
        "mceval": GroupConfig(
            add_counter=False,
            language=True,
            larger_than_median_loc=True,
            larger_than_median_prompt=True,
            larger_than_median_output=True,
            difficulty_easy=True,
            difficulty_medium=True,
            difficulty_hard=True
        ),
        "multipl-e": GroupConfig(
            add_counter=True,
            larger_than_median_loc=True,
            larger_than_median_prompt=True,
            larger_than_median_output=False,
            difficulty_easy=False,
            difficulty_medium=False,
            difficulty_hard=False,
            language=True
        )
    }


def run_calibration_methods(dataset, bins, plots):
    """Execute all calibration approaches and return results."""
    methods = {
        "Baseline": lambda: compute_baseline.main(
            data_provider=dataset, bins=bins),
        "Platt scaling": lambda: run_platt.main(
            extern=True, data_provider=dataset, bins=bins, plots=plots
        ),
        "Histogram binning": lambda: run_hb.main(
            extern=True, data_provider=dataset, bins=bins, plots=plots
        ),
        "Linear regression": lambda: run_lr.main(
            type='linear', extern=True, data_provider=dataset, bins=bins, plots=plots
        ),
        "Logistic regression": lambda: run_lr.main(
            type='logistic', extern=True, data_provider=dataset, bins=bins, plots=plots
        ),
        "Iterative group histogram binning": lambda: run_ighb.main(
            extern=True, data_provider=dataset, bins=bins, plots=plots
        ),
        "Iterative group linear binning": lambda: run_iglb.main(
            extern=True, data_provider=dataset, bins=bins, plots=plots
        )
    }
    
    results = {}
    for name, method in methods.items():
        print(f"{name}:")
        results[name] = method()
    
    return results

def create_results_table(results):
    """Create a formatted table of calibration results."""
    
    table_rows = [
        ["Uncalib", list(results["Baseline"]["scores_uncalibrated"].values())[0].values()],
        ["Platt", list(results["Platt scaling"]["scores_calibrated"].values())[0].values()],
        ["HB", list(results["Histogram binning"]["scores_calibrated"].values())[0].values()],
        ["LR", list(results["Linear regression"]["scores_calibrated"].values())[0].values()],
        ["LOGR", list(results["Logistic regression"]["scores_calibrated"].values())[0].values()],
        ["IGHB", list(results["Iterative group histogram binning"]["scores_calibrated"].values())[0].values()],
        ["IGLB", list(results["Iterative group linear binning"]["scores_calibrated"].values())[0].values()]
    ]
    
    # Flatten the values for each row
    table_rows = [[row[0]] + list(row[1]) for row in table_rows]
    
    headers = ['Method', 'ECE', 'ASCE', 'MSE', 'brier_ref', 'skill_score', 'accuracy', 'GASCE']
    return tabulate(table_rows, headers=headers, tablefmt='orgtbl')


def save_table(table, save_dir, prob_method):
    """Save the results table to a file."""
    file_path = os.path.join(save_dir, f'scores_{prob_method}.txt')
    with open(file_path, 'w') as f:
        f.write(table)



def main():
    """Main execution function."""
    args = cmd_input.load_parser()
    
    # Load benchmark configuration
    configs = get_benchmark_configs()
    config = configs[args.benchmark]
    
    # Load dataset
    dataset = CalibrationDataset(
            benchmark=args.benchmark,
            model=args.model,
            group_config=config,
            args=args
        )
    
    # Set up bins and plotting
    bins = binning.Binning(args.bin_count, args.binning_type)
    plots = CalibrationCharts(dataset.run, args.binning_type, bins.grid, dataset.save_dir)
    
    # Run all calibration methods
    results = run_calibration_methods(dataset, bins, plots)
    
    # Create and display results table
    table = create_results_table(results)
    print(table)
    
    # Save outputs based on args
    if args.save_table:
        save_table(table, dataset.save_dir, args.prob_method)
    
    if args.save_charts:
        plots.create_charts(results, args)




if __name__ == "__main__":
    main()
    
