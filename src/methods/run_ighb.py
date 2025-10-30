import numpy as np
import os
from tools import binning, cmd_input
from methods.ighb_calibration import IGHB_calibration
import pickle
from sklearn.model_selection import KFold
from tools.create_charts import CalibrationCharts


DEBUG = True
OUTPUTS = True

np.seterr(divide="ignore", invalid="ignore")


def main(data_provider, extern=False, bins=None, plots=None):
    # loads commandline parameter
    args = cmd_input.load_parser()

    if bins is None:
        bins = binning.Binning(args.bin_count, args.binning_type)

    train_probs = data_provider.get_train_probs(args.prob_method)
    train_is_correct = data_provider.get_train_is_correct()
    train_groups = data_provider.get_train_groups()

    ighb = IGHB_calibration(bins, args).fit(
        train_probs,
        train_is_correct,
        train_groups,
    )

    # Calculate values for uncalibrated test set

    test_probs = data_provider.get_test_probs(args.prob_method)
    test_is_correct = data_provider.get_test_is_correct()
    test_groups = data_provider.get_test_groups()

    scores_uncalibrated = ighb.score_obj.calc_all(
        test_probs,
        test_is_correct,
        groups=test_groups,
        set_brier_ref=True,
    )

    (
        _,
        correctness_group_uncalibrated,
        total_bin_uncalibrated,
        correctness_bin_uncalibrated,
    ) = ighb.score_obj.get_total_and_correctness(
        test_probs,
        test_is_correct,
        test_groups,
    )

    # set the conficence to calibrate on
    train_conf = train_probs.copy()
    test_conf = test_probs.copy()

    # calibration
    while ighb.max_error > ighb.alpha:

        train_conf = ighb.predict(train_conf, train_groups)

        # calculate corrected values for the test set
        test_conf = ighb.predict(test_conf, test_groups)

        # Calculate some metrics on the values
        curr_group_total, curr_group_correctness, _, curr_bin_correctness = (
            ighb.score_obj.get_total_and_correctness(
                test_conf,
                test_is_correct,
                test_groups,
            )
        )
       
        # fit the model on the updated train confidences
        ighb = ighb.fit(
            train_conf,
            train_is_correct,
            train_groups,
        )

    # Calculate values for calibrated test set
    scores_calibrated = ighb.score_obj.calc_all(
        test_conf,
        test_is_correct,
        groups=test_groups,
    )

    _, _, total_bin_calibrated, correctness_bin_calibrated = (
        ighb.score_obj.get_total_and_correctness(
            test_conf,
            test_is_correct,
            test_groups,
        )
    )

    total_group, correctness_group, average_group_confidence = (
        ighb.score_obj.get_correctness_per_group(
            test_conf,
            test_is_correct,
            test_groups,
        )
    )

    # Add to score table
    ighb.score_obj.add_to_score_table(
        data_provider.run, scores_uncalibrated, scores_calibrated
    )

    if extern:
        return {
            "total_bin_calibrated": total_bin_calibrated,
            "correctness_bin_calibrated": correctness_bin_calibrated,
            "correctness_group": correctness_group,
            "average_group_confidence": average_group_confidence,
            "total_group": total_group,
            "scores_calibrated": scores_calibrated,
            "calibrated_probs": test_probs,
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
    
    # display score table for all runs
    ighb.score_obj.display_score_table()

    # saves the score table
    if args.save_table:
        with open(data_provider.save_dir + "scores.txt", "w") as f:
            f.write(ighb.score_obj.printable_table)



