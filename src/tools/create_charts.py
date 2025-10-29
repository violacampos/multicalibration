import os
import matplotlib.pyplot as plt
import numpy as np


class Charts:

    def __init__(self, run, binning_type, grid, save_dir, bin_edges=None):
        """
        Initialization of chart class.

        :param run: Name of the run. Used for titles
        :param binning_type: Type of binning
        :param grid: List of grid points. Used for the location of each bar and bar size
        :param save_dir: Directory to save the charts to.
        :param bin_edges: Used only for quantil binning.
        """

        self.run = run
        self.debug = binning_type
        self.save_dir = save_dir
        self.grid = grid

        cmaps = [plt.cm.tab20, plt.cm.tab20b, plt.cm.tab20c]

        self.colors = [
            item for sublist in [cm.colors for cm in cmaps] for item in sublist
        ]

        # define chart ranges for display reasons
        if binning_type == "linear":
            self.chart_range = grid
            self.bar_width = 1 / (len(grid) - 1)
        elif binning_type == "quantil":
            if not bin_edges:
                exit("bin_edges needs to be set with quantil binning")
            self.chart_range = self.grid
            self.bar_width = np.array(bin_edges[1:] - bin_edges[:-1])

    def get_bar_colors(self, total, orange=False):
        """
        Calculates the shade of blue for each bar, based on the amount of samples in each bin.

        :param total: List of total values for each bin of the data

        :return: List of color tuples
        """
        colors = []
        total_bin_count_norm = (total - np.min(total)) / (np.max(total) - np.min(total))

        for x in total_bin_count_norm:
            if orange:
                colors.append(("tab:orange", x))
            else:
                colors.append(("tab:blue", x))

        return colors

    def calibration_method_comp_bar_chart(
        self,
        scoring_method: str,
        y1,
        y2,
        y3,
        y4,
        y5,
        y6,
        y7,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7,
    ):
        """
        Creates a bar chart for the baseline and each calibration method for comparison

        :param y1: Grid points where the total is not null for uncalibrated data
        :param y2: Grid points where the total is not null for HB
        :param y3: Grid points where the total is not null for LR
        :param y4: Grid points where the total is not null for IGHB
        :param y5: Grid points where the total is not null for IGLB
        :param x1: List of correctness values for each bin of the uncalibrated data
        :param x2: List of correctness values for each bin of the HB
        :param x3: List of correctness values for each bin of the LR
        :param x4: List of correctness values for each bin of the IGHB
        :param x5: List of correctness values for each bin of the IGLB
        """

        plt.rcParams["axes.labelsize"] = 14
        plt.rcParams["xtick.labelsize"] = 12
        plt.rcParams["ytick.labelsize"] = 12
        fig, axs = plt.subplots(1, 7, figsize=(24, 4), sharey=True)

        self.calibration_bar_chart(
            axs[0],
            "Uncalibrated",
            x1,
            self.get_bar_colors(y1, orange=True),
            ylabel=True,
        )
        self.calibration_bar_chart(
            axs[1], "Platt", x2, self.get_bar_colors(y2)
        )  # , y2, x2=x1, bar_colors2=self.get_bar_colors(y1, orange=True))
        self.calibration_bar_chart(
            axs[2], "HB", x3, self.get_bar_colors(y3)
        )  # , y3, x2=x1, bar_colors2=self.get_bar_colors(y1, orange=True))
        self.calibration_bar_chart(
            axs[3], "LINR", x4, self.get_bar_colors(y4)
        )  # , y4, x2=x1, bar_colors2=self.get_bar_colors(y1, orange=True))
        self.calibration_bar_chart(
            axs[4], "LOGR", x5, self.get_bar_colors(y5)
        )  # , y5, x2=x1, bar_colors2=self.get_bar_colors(y1, orange=True))
        self.calibration_bar_chart(
            axs[5], "IGHB", x6, self.get_bar_colors(y6)
        )  # , y6, x2=x1, bar_colors2=self.get_bar_colors(y1, orange=True))
        self.calibration_bar_chart(
            axs[6], "IGLB", x7, self.get_bar_colors(y7)
        )  # , y7, x2=x1, bar_colors2=self.get_bar_colors(y1, orange=True))

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{scoring_method}_calibration_comparison_bar.pdf"))
        plt.close()

    def calibration_bar_scatter_chart(
        self,
        scoring_method: str,
        y1,
        y2,
        y3,
        y4,
        y5,
        y6,
        y7,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7,
        uncalib_corr,
        uncalib_conf,
        uncalib_total,
        platt_corr,
        platt_conf,
        platt_total,
        hb_corr,
        hb_conf,
        hb_total,
        lr_corr,
        lr_conf,
        lr_total,
        logr_corr,
        logr_conf,
        logr_total,
        ighb_corr,
        ighb_conf,
        ighb_total,
        iglb_corr,
        iglb_conf,
        iglb_total,
    ):
        """
        Creates a bar chart for the baseline and each caliibration method for comparison

        :param y1: Grid points where the total is not null for uncalibrated data
        :param y2: Grid points where the total is not null for HB
        :param y3: Grid points where the total is not null for LR
        :param y4: Grid points where the total is not null for IGHB
        :param y5: Grid points where the total is not null for IGLB
        :param x1: List of correctness values for each bin of the uncalibrated data
        :param x2: List of correctness values for each bin of the HB
        :param x3: List of correctness values for each bin of the LR
        :param x4: List of correctness values for each bin of the IGHB
        :param x5: List of correctness values for each bin of the IGLB
        """

        plt.rcParams["axes.labelsize"] = 14
        plt.rcParams["xtick.labelsize"] = 12
        plt.rcParams["ytick.labelsize"] = 12
        fig, axs = plt.subplots(2, 7, figsize=(24, 7), sharey=True)

        self.calibration_bar_chart(
            axs[0, 0],
            "Uncalibrated",
            x1,
            self.get_bar_colors(y1, orange=True),
            ylabel=True,
        )
        self.calibration_bar_chart(
            axs[0, 1], "Platt", x2, self.get_bar_colors(y2)
        )  # , y2, x2=x1, bar_colors2=self.get_bar_colors(y1, orange=True))
        self.calibration_bar_chart(
            axs[0, 2], "HB", x3, self.get_bar_colors(y3)
        )  # , y3, x2=x1, bar_colors2=self.get_bar_colors(y1, orange=True))
        self.calibration_bar_chart(
            axs[0, 3], "LINR", x4, self.get_bar_colors(y4)
        )  # , y4, x2=x1, bar_colors2=self.get_bar_colors(y1, orange=True))
        self.calibration_bar_chart(
            axs[0, 4], "LOGR", x5, self.get_bar_colors(y5)
        )  # , y5, x2=x1, bar_colors2=self.get_bar_colors(y1, orange=True))
        self.calibration_bar_chart(
            axs[0, 5], "IGHB", x6, self.get_bar_colors(y6)
        )  # , y6, x2=x1, bar_colors2=self.get_bar_colors(y1, orange=True))
        self.calibration_bar_chart(
            axs[0, 6], "IGLB", x7, self.get_bar_colors(y7)
        )  # , y7, x2=x1, bar_colors2=self.get_bar_colors(y1, orange=True))

        self.scatter_plot(
            axs[1, 0], "Uncalibrated", uncalib_conf, uncalib_corr, uncalib_total / 2
        )
        self.scatter_plot(axs[1, 1], "Platt", platt_conf, platt_corr, platt_total / 2)
        self.scatter_plot(axs[1, 2], "HB", hb_conf, hb_corr, hb_total / 2)
        self.scatter_plot(axs[1, 3], "LINR", lr_conf, lr_corr, lr_total / 2)
        self.scatter_plot(axs[1, 4], "LOGR", logr_conf, logr_corr, logr_total / 2)
        self.scatter_plot(axs[1, 5], "IGHB", ighb_conf, ighb_corr, ighb_total / 2)
        self.scatter_plot(axs[1, 6], "IGLB", iglb_conf, iglb_corr, iglb_total / 2)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{scoring_method}_calibration_bar_scatter.pdf"))
        plt.close()

    """def histogram(self, data, path, typ):
        fig, ax = plt.subplots()  
        ax.hist(data, range=(0, 1.0))
        ax.plot([0, 1], [0, 1], transform=ax.transAxes)
        plt.title(self.run+' # '+typ, fontsize=7)
        plt.savefig(path)
        print(f"Histogram saved: {path}")
        plt.close()"""

    def count_distribution(self, ax, sub_title, totals):
        """
        Creates a bar chart with the totals to display the distribution over the bins.

        :param ax: ax object to create chart on
        :param sub_title: Title of the subplot
        :param totals: List of total values for each bin
        """
        ax.set_title(sub_title, fontsize=12)
        bars = ax.bar(
            self.grid,
            totals,
            width=self.bar_width,
            color=["tab:blue"],
            edgecolor="black",
        )
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set(xlabel="Confidence")
        ax.set(ylabel="Count")

    def calibration_bar_chart(
        self,
        ax,
        sub_title,
        x,
        bar_colors,
        totals=None,
        x2=None,
        bar_colors2=None,
        ylabel: bool = None,
    ):
        """
        Creates a bar chart with the correctness values to display the calibration over the bins.

        :param ax: ax object to create chart on
        :param sub_title: Title of the subplot
        :param x: List of correctness values for each bin
        :param bar_colors: List of colors for the bars
        :param totals: Optional to add totals to each bar
        """
        ax.set_title(sub_title, fontsize=18, fontweight="bold")

        if bar_colors2 != None and any(x2 != None):
            bars = ax.bar(
                self.grid,
                x2,
                width=self.bar_width,
                color=bar_colors2,
                edgecolor="black",
            )
        bars = ax.bar(
            self.grid, x, width=self.bar_width, color=bar_colors, edgecolor="black"
        )
        ax.plot([0, 1], [0, 1], linestyle="--")
        if totals is not None:
            ax.bar_label(bars, totals, fontsize=6)
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.set(xlabel="Confidence")
        if ylabel:
            ax.set(ylabel="Correctness")

    def scatter_plot(self, ax, method, x, y, area):
        """
        Creates a scatter plot to display the calibration within each group for a calibration approach.

        :param ax: ax object to create chart on
        :param method: Name of the use calibration method
        :param x: List of confidence values for each group
        :param y: List of correctness values for each group
        :param area: Size of the dot. Represents the amount of samples in the group.
        """
        colors = self.colors[: len(x)]

        scatter = ax.scatter(
            x, y, s=area, c=colors, alpha=0.9, marker="o"
        )  # r'$\odot$')
        # ax.set_title(method, fontsize=18, fontweight="bold")
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--")
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.set(xlabel="Confidence")
        ax.set(ylabel="Correctness")

    def group_calibration_scatter(
        self,
        scoring_method: str,
        uncalib_corr,
        uncalib_conf,
        uncalib_total,
        platt_corr,
        platt_conf,
        platt_total,
        hb_corr,
        hb_conf,
        hb_total,
        lr_corr,
        lr_conf,
        lr_total,
        logr_corr,
        logr_conf,
        logr_total,
        ighb_corr,
        ighb_conf,
        ighb_total,
        iglb_corr,
        iglb_conf,
        iglb_total,
    ):
        """
        Creates a plot to display the calibration within each group for all calibration approaches.

        :param ..._corr: correctness for each group of the given method
        :param ..._conf: confidence for each group of the given method
        :param ..._total: total for each group of the given method
        """
        fig, axs = plt.subplots(1, 7, figsize=(24, 4), sharey=True)

        self.scatter_plot(
            axs[0], "Uncalibrated", uncalib_conf, uncalib_corr, uncalib_total / 2
        )
        self.scatter_plot(axs[1], "Platt", platt_conf, platt_corr, platt_total / 2)
        self.scatter_plot(axs[2], "HB", hb_conf, hb_corr, hb_total / 2)
        self.scatter_plot(axs[3], "LINR", lr_conf, lr_corr, lr_total / 2)
        self.scatter_plot(axs[4], "LOGR", logr_conf, logr_corr, logr_total / 2)
        self.scatter_plot(axs[5], "IGHB", ighb_conf, ighb_corr, ighb_total / 2)
        self.scatter_plot(axs[6], "IGLB", iglb_conf, iglb_corr, iglb_total / 2)
        # axs[0,1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{scoring_method}_group_calibration.pdf"))
        plt.close()

    def calibration_info(
        self,
        total_uncalibrated,
        correctness_uncalibrated,
        total_calibrated,
        correctness_calibrated,
    ):
        """
        Creates multiple subplots to visualize the calibration changes of a method

        :param total_uncalibrated: total values for the uncalibrated data
        :param correctness_uncalibrated: correcntess values for the uncalibrated data
        :param total_calibrated: total values for the calibrated data
        :param correctness_calibrated: correcntess values for the calibrated data
        """
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        self.calibration_bar_chart(
            axs[0, 0],
            "Test uncalibrated",
            correctness_uncalibrated,
            self.get_bar_colors(total_uncalibrated),
        )
        self.calibration_bar_chart(
            axs[0, 1],
            "Test calibrated",
            correctness_calibrated,
            self.get_bar_colors(total_calibrated),
        )

        self.count_distribution(
            axs[1, 0], "Test uncalibrated distribution", total_uncalibrated
        )
        self.count_distribution(
            axs[1, 1], "Test calibrated distribution", total_calibrated
        )

        plt.savefig(os.path.join(self.save_dir, "calibration_infos.pdf"))
        plt.close()
        
    def create_charts(self, results, args):
        """Generate calibration charts."""
        baseline = results["Baseline"]
        platt = results["Platt scaling"]
        hb = results["Histogram binning"]
        lr = results["Linear regression"]
        logr = results["Logistic regression"]
        ighb = results["Iterative group histogram binning"]
        iglb = results["Iterative group linear binning"]
        
        self.calibration_method_comp_bar_chart(
            args.prob_method,
            baseline["total_bin_uncalibrated"],
            platt["total_bin_calibrated"],
            hb["total_bin_calibrated"],
            lr["total_bin_calibrated"],
            logr["total_bin_calibrated"],
            ighb["total_bin_calibrated"],
            iglb["total_bin_calibrated"],
            baseline["correctness_bin_uncalibrated"],
            platt["correctness_bin_calibrated"],
            hb["correctness_bin_calibrated"],
            lr["correctness_bin_calibrated"],
            logr["correctness_bin_calibrated"],
            ighb["correctness_bin_calibrated"],
            iglb["correctness_bin_calibrated"]
        )
        
        self.group_calibration_scatter(
            args.prob_method,
            baseline["correctness_group_uncalib"],
            baseline["average_group_confidence_uncalib"],
            baseline["total_group_uncalib"],
            platt["correctness_group"],
            platt["average_group_confidence"],
            platt["total_group"],
            hb["correctness_group"],
            hb["average_group_confidence"],
            hb["total_group"],
            lr["correctness_group"],
            lr["average_group_confidence"],
            lr["total_group"],
            logr["correctness_group"],
            logr["average_group_confidence"],
            logr["total_group"],
            ighb["correctness_group"],
            ighb["average_group_confidence"],
            ighb["total_group"],
            iglb["correctness_group"],
            iglb["average_group_confidence"],
            iglb["total_group"]
        )
