from typing import Optional
import numpy as np
import os
from tools import binning, cmd_input

from tools.create_charts import CalibrationCharts
from methods.platt_calibration import Platt_calibration


def main(
    data_provider,
    extern: bool = False,
    bins: Optional[binning.Binning] = None,
    plots: Optional[CalibrationCharts]=None,
):
    args = cmd_input.load_parser()

    # training data
    y = data_provider.get_train_is_correct()
    X = data_provider.get_train_probs(args.prob_method)

    if bins is None:
        bins = binning.Binning(args.bin_count, args.binning_type)


    # Train platt calibration
    platt = Platt_calibration(bins, args).fit(X, y)

    # Calculate uncalibrated scores and per-bin/group stats for test set
    test_probs = data_provider.get_test_probs(args.prob_method)
    test_is_correct = data_provider.get_test_is_correct()
    test_groups = data_provider.get_test_groups()
    
    scores_uncalibrated = platt.score_obj.calc_all(
        test_probs, test_is_correct, groups=test_groups, set_brier_ref=True
    )
    
    (
        total_group_uncalibrated,
        correctness_group_uncalibrated,
        total_bin_uncalibrated,
        correctness_bin_uncalibrated,
    ) = platt.score_obj.get_total_and_correctness(test_probs, test_is_correct, test_groups)

    # Predict for test set
    calibrated_predictions = platt.predict(test_probs)


    # Calculate scores on uncalibrated test set
    scores_calibrated = platt.score_obj.calc_all(
        calibrated_predictions, test_is_correct, groups=test_groups
    )

    (
        total_group_calibrated,
        correctness_group_calibrated,
        total_bin_calibrated,
        correctness_bin_calibrated,
    ) = platt.score_obj.get_total_and_correctness(
        calibrated_predictions, test_is_correct, test_groups
    )

    total_group, correctness_group, average_group_confidence = (
        platt.score_obj.get_correctness_per_group(
            calibrated_predictions, test_is_correct, test_groups
        )
    )

    if getattr(args, "print_info", False):
        print(f"coef: {platt.platt.coef_}")
        print(f"bias: {platt.platt.intercept_}")


    # only return values when called from other script
    if extern:
        return {
            "total_bin_calibrated": total_bin_calibrated,
            "correctness_bin_calibrated": correctness_bin_calibrated,
            "correctness_group": correctness_group,
            "average_group_confidence": average_group_confidence,
            "total_group": total_group,
            "scores_calibrated": scores_calibrated,
            "calibrated_probs": calibrated_predictions,
            "group_names": data_provider.group_names,
        }
    else:
        if getattr(args, "save_charts", False):
            if plots is None:
        
                plots = CalibrationCharts(
                    data_provider.run, args.binning_type, bins.grid, data_provider.save_dir
                )
            plots.calibration_info(
                total_bin_uncalibrated,
                correctness_bin_uncalibrated,
                total_bin_calibrated,
                correctness_bin_calibrated,
            )




