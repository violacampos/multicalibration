import numpy as np
from tools import binning, cmd_input
from methods.iglb_calibration import IGLB_calibration
from tools.create_charts import CalibrationCharts


def main(data_provider, extern=False, bins=None, plots=None):
    """
    Main function to run IGLB calibration.
    
    :param data_provider: Data provider object
    :param extern: Whether to return results externally
    :param bins: Optional binning object
    :param plots: Optional plotting object
    :return: Results dictionary if extern=True
    """
    # Load command line parameters
    args = cmd_input.load_parser()

    # Initialize binning if not provided
    if bins is None:
        bins = binning.Binning(args.bin_count, args.binning_type)

    # Load training data
    train_probs = data_provider.get_train_probs(args.prob_method)
    train_is_correct = data_provider.get_train_is_correct()
    train_groups = data_provider.get_train_groups()

    # Load validation data
    val_probs = data_provider.get_val_probs(args.prob_method)
    val_is_correct = data_provider.get_val_is_correct()
    val_groups = data_provider.get_val_groups()

    # Load test data
    test_probs = data_provider.get_test_probs(args.prob_method)
    test_is_correct = data_provider.get_test_is_correct()
    test_groups = data_provider.get_test_groups()

    # Create calibration object
    iglb = IGLB_calibration(bins, args)

    # Calculate uncalibrated scores on test data
    scores_uncalibrated = iglb.score_obj.calc_all(
        test_probs,
        test_is_correct,
        groups=test_groups,
        set_brier_ref=True
    )

    _, correctness_group_uncalibrated, total_bin_uncalibrated, correctness_bin_uncalibrated = (
        iglb.score_obj.get_total_and_correctness(
            test_probs,
            test_is_correct,
            test_groups
        )
    )

    # Fit calibration model on training data with validation for early stopping
    iglb.fit(
        train_probs,
        train_is_correct,
        train_groups,
        val_probs,
        val_is_correct,
        val_groups
    )

    # Transform test data using learned calibration
    test_probs_calibrated = iglb.transform(test_probs, test_groups)

    # Calculate calibrated scores
    scores_calibrated = iglb.score_obj.calc_all(
        test_probs_calibrated,
        test_is_correct,
        groups=test_groups
    )

    _, _, total_bin_calibrated, correctness_bin_calibrated = (
        iglb.score_obj.get_total_and_correctness(
            test_probs_calibrated,
            test_is_correct,
            test_groups
        )
    )

    total_group, correctness_group, average_group_confidence = (
        iglb.score_obj.get_correctness_per_group(
            test_probs_calibrated,
            test_is_correct,
            test_groups
        )
    )


    # Return results or save/display
    if extern:
        return {
            "total_bin_calibrated": total_bin_calibrated,
            "correctness_bin_calibrated": correctness_bin_calibrated,
            "correctness_group": correctness_group,
            "average_group_confidence": average_group_confidence,
            "total_group": total_group,
            "scores_calibrated": scores_calibrated,
            "calibrated_probs": test_probs_calibrated,
            "group_names": data_provider.group_names,
            "calibration_steps": iglb.calibration_steps
        }
    else:
        # Create charts if requested
        if args.save_charts:
            if plots is None:
                plots = CalibrationCharts(
                    data_provider.run,
                    args.binning_type,
                    bins.grid,
                    data_provider.save_dir
                )
            plots.calibration_info(
                total_bin_uncalibrated,
                correctness_bin_uncalibrated,
                total_bin_calibrated,
                correctness_bin_calibrated
            )
