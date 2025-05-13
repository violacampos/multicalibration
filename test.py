from tools import data, cmd_input
args = cmd_input.load_parser()

data.generate_save_dir("test", "comparison", args.binning_type, args.prob_method, args.split, args.model)