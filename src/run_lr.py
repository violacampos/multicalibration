import numpy as np
import os
from tools import binning, cmd_input
from tools.create_charts import Charts
from tools.lr_calibration import LR_calibration


def main(data_provider, type="linear", extern=False, bins=None, plots=None):
    args = cmd_input.load_parser()

    # training data
    y = data_provider.get_train_is_correct()
    X = np.hstack(
        [
            data_provider.get_train_probs(args.prob_method).values.reshape(-1, 1),
            data_provider.get_train_groups(),
        ]
    )

    if bins is None:
        bins = binning.Binning(args.bin_count, args.binning_type)

    # Train linear regression on training data
    lr = LR_calibration(bins, args, type).fit(X, y)

    # Calculate uncalibrated scores on test set
    test_probs = data_provider.get_test_probs(args.prob_method)
    test_is_correct = data_provider.get_test_is_correct()
    test_groups = data_provider.get_test_groups()

    scores_uncalibrated = lr.score_obj.calc_all(
        test_probs, test_is_correct, groups=test_groups, set_brier_ref=True
    )
    (
        total_group_uncalibrated,
        correctness_group_uncalibrated,
        total_bin_uncalibrated,
        correctness_bin_uncalibrated,
    ) = lr.score_obj.get_total_and_correctness(test_probs, test_is_correct, test_groups)

    # Predict for test data
    calibrated_predictions = lr.predict(np.hstack([test_probs.values.reshape(-1, 1), test_groups]))


    # Calculate calibrated scores on test set
    scores_calibrated = lr.score_obj.calc_all(
        calibrated_predictions, test_is_correct, groups=test_groups
    )

    (
        total_group_calibrated,
        correctness_group_calibrated,
        total_bin_calibrated,
        correctness_bin_calibrated,
    ) = lr.score_obj.get_total_and_correctness(
        calibrated_predictions, test_is_correct, test_groups
    )

    total_group, correctness_group, average_group_confidence = (
        lr.score_obj.get_correctness_per_group(
            calibrated_predictions, test_is_correct, test_groups
        )
    )

    if getattr(args, "print_info", False):

        print(f"Group weights: {lr.reg.coef_}")
        print(f"bias: {lr.reg.intercept_}")

    # Add result to score table
    lr.score_obj.add_to_score_table(
        data_provider.run, scores_uncalibrated, scores_calibrated
    )

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
        if args.save_charts:
            if plots is None:
                plots = Charts(
                    data_provider.run, args.binning_type, bins.grid, data_provider.save_dir
                )
            plots.calibration_info(
                total_bin_uncalibrated,
                correctness_bin_uncalibrated,
                total_bin_calibrated,
                correctness_bin_calibrated,
            )

    # display score table 
    lr.score_obj.display_score_table()

    if getattr(args, "save_table", False):
        out_path = os.path.join(data_provider.save_dir, "scores.txt")
        with open(out_path, "w") as f:
            f.write(lr.score_obj.printable_table)


if __name__ == "__main__":
    main()
