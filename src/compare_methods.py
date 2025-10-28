import os

from tabulate import tabulate


import run_hb
import run_lr
import run_ighb
import run_iglb
import compute_baseline
import run_platt
from tools import binning, cmd_input
from tools.create_charts import Charts
from tools.dataset import GroupConfig, HumanEvalDataset, LiveCodeBenchDataset


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
        "humaneval": GroupConfig(
            add_counter=True,
            larger_than_median_loc=True,
            larger_than_median_prompt=True,
            difficulty_easy=False,
            difficulty_medium=False,
            difficulty_hard=False,
            language=True
        )
    }

def load_dataset(args, config):
    """Load the appropriate dataset based on benchmark type."""
    if args.benchmark in ["livecodebench", "mceval"]:
        return LiveCodeBenchDataset(
            jsonl_path=args.data_path,
            split='train',
            benchmark=args.benchmark,
            group_config=config,
            args=args
        )
    else:
        run_dirs = sorted([x[0] for x in os.walk(args.dir[0])])
        run_dir = run_dirs[0]
        return HumanEvalDataset(
            jsonl_path=args.data_path,
            run_dir=run_dir,
            group_config=config
        )


def run_calibration_methods(split_obj, grid, chartmaker):
    """Execute all calibration approaches and return results."""
    methods = {
        "Baseline": lambda: compute_baseline.main(
            data_provider=split_obj, bins=grid),
        "Platt scaling": lambda: run_platt.main(
            extern=True, data_provider=split_obj, grid=grid, chartmaker=chartmaker
        ),
        "Histogram binning": lambda: run_hb.main(
            extern=True, data_provider=split_obj, grid=grid, chartmaker=chartmaker
        ),
        "Linear regression": lambda: run_lr.main(
            type='linear', extern=True, data_provider=split_obj, grid=grid, chartmaker=chartmaker
        ),
        "Logistic regression": lambda: run_lr.main(
            type='logistic', extern=True, data_provider=split_obj, grid=grid, chartmaker=chartmaker
        ),
        "Iterative group histogram binning": lambda: run_ighb.main(
            extern=True, data_provider=split_obj, grid=grid, chartmaker=chartmaker
        ),
        "Iterative group linear binning": lambda: run_iglb.main(
            extern=True, data_provider=split_obj, grid=grid, chartmaker=chartmaker
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
    split_obj = load_dataset(args, config)
    
    # Set up grid and plotting
    bins = binning.Binning(args.bin_count, args.binning_type)
    plots = Charts(split_obj.run, args.binning_type, bins.grid, split_obj.save_dir)
    
    # Run all calibration methods
    results = run_calibration_methods(split_obj, bins, plots)
    
    # Create and display results table
    table = create_results_table(results)
    print(table)
    
    # Save outputs based on args
    if args.save_table:
        save_table(table, split_obj.save_dir, args.prob_method)
    
    if args.save_charts:
        plots.create_charts(results, args)




if __name__ == "__main__":
    main()
    
