import argparse


def load_parser():
    """
    Creates a parser for command line inputs

    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", type=str, help="Directory with results.", nargs="+")

    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to jsonlines-file with results.",
        default=None,
    )

    parser.add_argument(
        "--benchmark",
        choices=["livecodebench", "humaneval", "mceval"],
        default="humaneval",
        help="The evaluation benchmark.",
    )
    
    parser.add_argument(
        "--model",
        choices=["qwen3", "gpt-oss", "r1-distil"],
        default="qwen3",
        help="The LLM used for sample generation.",
    )

    parser.add_argument(
        "--save-table", action="store_true", help="Flag to save the result table."
    )

    parser.add_argument(
        "--prob-method",
        choices=[
            "avg_prob",
            "qualitativ",
            "quantitativ",
            "code_prob",
            "tail_prob",
            "code_top20_prob",
            "tail_top20_prob",
        ],
        default="avg_prob",
        help="Choose which probability to use.",
    )

    parser.add_argument(
        "--binning-type",
        choices=["linear"],
        default="linear",
        help="Choose which binning type to use.",
    )

    parser.add_argument(
        "--bin-count", type=int, default=20, help="Number of bins for calibration."
    )

    parser.add_argument(
        "--save-charts",
        action="store_true",
        default=False,
        help="Save the charts for the method.",
    )

    parser.add_argument(
        "--save-data",
        action="store_true",
        default=False,
        help="Save the output data of all methods",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="Epsilon value only for IGLB method.",
    )

    parser.add_argument(
        "--debug", action="store_true", default=False, help="Print debug information."
    )

    parser.add_argument(
        "--print-info",
        action="store_true",
        default=False,
        help="Print intermediate states to command line.",
    )

    args = parser.parse_args()

    return args
