import numpy as np
import os
from tools import binning, cmd_input
from tools.ighb_calibration import IGHB_calibration
import pickle
from sklearn.model_selection import KFold
from tools.data import data_loader
from tools.split import split

DEBUG = True
OUTPUTS = True

np.seterr(divide="ignore", invalid="ignore")


def main(data_provider, extern=False, grid=None, chartmaker=None):
    # loads commandline parameter
    args = cmd_input.load_parser()

    m = args.bin_count

    if OUTPUTS:
        print(f"Run: {data_provider.run}")
    if OUTPUTS:
        print(f"Gruppen Anzahl: {data_provider.get_train_groups().sum(axis=0)}")

    # Possible k_fold VIOLA: did not check this yet
    if args.k_fold:
        if args.split:
            exit("Can't use split while using K-Fold!")

        kf = KFold(n_splits=5)
        kf.get_n_splits(data_provider.get_train_probs(args.prob_method))

        history = {}
        for i, (train_index, test_index) in enumerate(
            kf.split(data_provider.get_train_probs(args.prob_method))
        ):
            print(f"Fold {i}:")

            train_X = data_provider.get_train_probs(args.prob_method)[train_index]
            test_X = data_provider.get_train_probs(args.prob_method)[test_index]
            train_y = data_provider.get_train_is_correct()[train_index]
            test_y = data_provider.get_train_is_correct()[test_index]
            train_groups = data_provider.get_train_groups()[train_index]
            test_groups = data_provider.get_train_groups()[test_index]

            # get the grid for binning type and the chartmaker obj
            grid, chartmaker = binning.get_grid_and_chartmaker(
                data_provider.run, args, data_provider.save_dir, extern, probs=train_X
            )

            # Fit calibrator
            ighb = IGHB_calibration(grid, args).fit(
                train_X, train_y, train_groups
            )

            # Calculate values for uncalibrated test set
            scores_uncalibrated = ighb.score_obj.calc_all(
                test_X, test_y, groups=test_groups, set_brier_ref=True
            )

            (
                _,
                correctness_group_uncalibrated,
                total_bin_uncalibrated,
                correctness_bin_uncalibrated,
            ) = ighb.score_obj.get_total_and_correctness(test_X, test_y, test_groups)

            temp_group_correctness = correctness_group_uncalibrated
            temp_bin_correctness = correctness_bin_uncalibrated

            # set the conf to calibrate on
            calibrated_conf = train_X
            history_item = {}
            while ighb.max_error > ighb.alpha:
                if DEBUG:
                    print(f"Max Error: {ighb.max_error}")
                # get new better calibrated confidences
                calibrated_conf = ighb.predict(
                    calibrated_conf, train_groups, is_correct=train_y
                )

                # calculate the corrected values for the test set
                test_X = ighb.predict(test_X, test_groups, test=True, is_correct=test_y)

                # Calculate some metrics on the UNcorrected values
                curr_group_total, curr_group_correctness, _, curr_bin_correctness = (
                    ighb.score_obj.get_total_and_correctness(
                        test_X, test_y, test_groups
                    )
                )
                # history_item[len(ighb.changes)] = [
                #     temp_group_correctness,
                #     curr_group_correctness,
                #     ighb.changes[-1],
                #     curr_group_total,
                #     temp_bin_correctness,
                #     curr_bin_correctness,
                # ]
                temp_group_correctness = curr_group_correctness
                temp_bin_correctness = curr_bin_correctness

                # fit the model on the corrected confidences
                ighb = ighb.fit(calibrated_conf, train_y, train_groups)

            # Calculate values for calibrated test set
            scores_calibrated = ighb.score_obj.calc_all(
                test_X, test_y, groups=test_groups
            )

            _, _, total_bin_calibrated, correctness_bin_calibrated = (
                ighb.score_obj.get_total_and_correctness(test_X, test_y, test_groups)
            )

            # Add entry for the run in the score table
            ighb.score_obj.add_to_score_table(
                data_provider.run, scores_uncalibrated, scores_calibrated
            )
            ighb.score_obj.display_score_table()

            history_item["score"] = ighb.score_obj.score_table
            history[i] = history_item
            ighb = None
        if args.save_history:
            with open(
                data_provider.save_dir + "history_data/ighb_history_kfold.pkl", "wb"
            ) as f:
                pickle.dump(history, f)
    else:
        # get the grid for binning type and the chartmaker obj
        if grid is None or chartmaker is None:
            grid, chartmaker = binning.get_grid_and_chartmaker(
                data_provider.run,
                args,
                data_provider.save_dir,
                extern,
                probs=data_provider.get_train_probs(args.prob_method),
            )

        ighb = IGHB_calibration(grid, args).fit(
            data_provider.get_train_probs(args.prob_method),
            data_provider.get_train_is_correct(),
            data_provider.get_train_groups(),
        )

        # Calculate values for uncalibrated test set
        scores_uncalibrated = ighb.score_obj.calc_all(
            data_provider.get_test_probs(args.prob_method),
            data_provider.get_test_is_correct(),
            groups=data_provider.get_test_groups(),
            set_brier_ref=True,
        )

        (
            _,
            correctness_group_uncalibrated,
            total_bin_uncalibrated,
            correctness_bin_uncalibrated,
        ) = ighb.score_obj.get_total_and_correctness(
            data_provider.get_test_probs(args.prob_method),
            data_provider.get_test_is_correct(),
            data_provider.get_test_groups(),
        )

        temp_group_correctness = correctness_group_uncalibrated
        temp_bin_correctness = correctness_bin_uncalibrated

        # set the conf to calibrate on
        train_conf = data_provider.get_train_probs(args.prob_method).copy()
        test_conf = data_provider.get_test_probs(args.prob_method).copy()
        history = {}
        while ighb.max_error > ighb.alpha:
            # print(f"Max Error: {ighb.max_error}")
            # get new calibrated confidences
            if args.split:
                train_conf = ighb.predict(
                    train_conf,
                    data_provider.get_train_groups(),
                    is_correct=data_provider.get_train_is_correct(),
                )
            else:
                train_conf = ighb.predict(
                    train_conf,
                    data_provider.get_train_groups(),
                    test=True,
                    is_correct=data_provider.get_train_is_correct(),
                )

            # calculate the corrected values for the test set VIOLA check this!!
            if args.split:
                test_conf = ighb.predict(
                    test_conf,
                    data_provider.get_test_groups(),
                    test=True,
                    is_correct=data_provider.get_test_is_correct(),
                )
            else:
                test_conf = train_conf

            # Calculate some metrics on the values
            curr_group_total, curr_group_correctness, _, curr_bin_correctness = (
                ighb.score_obj.get_total_and_correctness(
                    test_conf,
                    data_provider.get_test_is_correct(),
                    data_provider.get_test_groups(),
                )
            )
            # Storing history
            # history[len(ighb.changes)] = [
            #     temp_group_correctness,
            #     curr_group_correctness,
            #     ighb.changes[-1],
            #     curr_group_total,
            #     temp_bin_correctness,
            #     curr_bin_correctness,
            # ]  # chartmaker.map_correctness_to_eleven_bins(
            temp_group_correctness = curr_group_correctness
            temp_bin_correctness = curr_bin_correctness

            # fit the model on the corrected confidences
            ighb = ighb.fit(
                train_conf,
                data_provider.get_train_is_correct(),
                data_provider.get_train_groups(),
            )

        # Calculate values for calibrated test set
        scores_calibrated = ighb.score_obj.calc_all(
            test_conf,
            data_provider.get_test_is_correct(),
            groups=data_provider.get_test_groups(),
        )

        _, _, total_bin_calibrated, correctness_bin_calibrated = (
            ighb.score_obj.get_total_and_correctness(
                test_conf,
                data_provider.get_test_is_correct(),
                data_provider.get_test_groups(),
            )
        )

        total_group, correctness_group, average_group_confidence = (
            ighb.score_obj.get_correctness_per_group(
                test_conf,
                data_provider.get_test_is_correct(),
                data_provider.get_test_groups(),
            )
        )

        # Add entry for the run in the score table
        ighb.score_obj.add_to_score_table(
            data_provider.run, scores_uncalibrated, scores_calibrated
        )

        history["score"] = ighb.score_obj.score_table

        if extern:
            return {
                "total_bin_calibrated": total_bin_calibrated,
                "correctness_bin_calibrated": correctness_bin_calibrated,
                "correctness_group": correctness_group,
                "average_group_confidence": average_group_confidence,
                "total_group": total_group,
                "scores_calibrated": scores_calibrated,
                "calibrated_probs": data_provider.get_test_probs(args.prob_method),
                "group_names": data_provider.group_names,
            }
        else:
            if args.save_charts:
                chartmaker.calibration_info(
                    total_bin_uncalibrated,
                    correctness_bin_uncalibrated,
                    total_bin_calibrated,
                    correctness_bin_calibrated,
                )
        if args.save_history:
            with open(
                data_provider.save_dir + "history_data/ighb_history.pkl", "wb"
            ) as f:
                pickle.dump(history, f)

        # display score table for all runs
        ighb.score_obj.display_score_table()

        # saves the score table
        if args.save_table:
            with open(data_provider.save_dir + "scores.txt", "w") as f:
                f.write(ighb.score_obj.printable_table)


if __name__ == "__main__":
    main()
