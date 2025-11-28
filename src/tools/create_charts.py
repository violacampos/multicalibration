import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import NDArray


class CalibrationCharts:
    
    # Constants
    COLOR_UNCALIBRATED = 'tab:orange'
    COLOR_CALIBRATED = 'tab:blue'
    CALIBRATION_METHODS = [
        'Uncalibrated', 'Platt', 'HB', 'LINR', 'LOGR', 'IGHB', 'IGLB'
    ]

    def __init__(
        self,
        run: str,
        binning_type: str,
        grid: NDArray[np.floating],
        save_dir: str,
        bin_edges: Optional[NDArray[np.floating]] = None
    ):
        """
        Initialize the charts class.

        Args:
            run: Name of the run (used for titles)
            binning_type: Type of binning ('linear' or 'quantil')
            grid: Array of grid points for bar locations and sizing
            save_dir: Directory path to save the charts
            bin_edges: Required for quantile binning
            
        Raises:
            ValueError: If binning_type is 'quantil' and bin_edges is not provided
        """
        self.run = run
        self.binning_type = binning_type
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.grid = grid

        # Generate color palette for scatter plots
        cmaps = [plt.cm.get_cmap("tab20"), 
                 plt.cm.get_cmap("tab20b"), 
                 plt.cm.get_cmap("tab20c")]
        self.colors = [
            color for cmap in cmaps for color in cmap.colors
        ]
        
        # Configure chart parameters based on binning type
        if binning_type == "linear":
            self.chart_range = grid
            self.bar_width = 1 / (len(grid) - 1)
        elif binning_type == "quantil":
            if bin_edges is None:
                raise ValueError("bin_edges must be provided for quantile binning")
            self.chart_range = grid
            self.bar_width = np.diff(bin_edges)
        else:
            raise ValueError(f"Unknown binning_type: {binning_type}")

    
    def _get_bar_colors(
        self, 
        total: NDArray[np.floating], 
        use_orange: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Calculate color shades for bars based on sample counts in each bin.

        Args:
            total: Array of total values for each bin
            use_orange: If True, use orange color scheme; otherwise use blue

        Returns:
            List of (color_name, intensity) tuples
        """
        colors = []
        
        # Normalize to [0, 1] range
        total_min, total_max = np.min(total), np.max(total)
        if total_max > total_min:
            normalized = (total - total_min) / (total_max - total_min)
        else:
            normalized = np.ones_like(total)

        color_name = self.COLOR_UNCALIBRATED if use_orange else self.COLOR_CALIBRATED
        
        for intensity in normalized:
            colors.append((color_name, intensity))

        return colors
    
    def _setup_plot_style(self):
        """Configure matplotlib plot styling."""
        plt.rcParams["axes.labelsize"] = 14
        plt.rcParams["xtick.labelsize"] = 12
        plt.rcParams["ytick.labelsize"] = 12
    
    def _save_and_close(self, filename: str):
        """Save figure and close to free memory."""
        filepath = self.save_dir / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
    def calibration_bar_chart(
        self,
        ax: Axes,
        title: str,
        correctness: NDArray[np.floating],
        bar_colors: List[Tuple[str, float]],
        totals: Optional[NDArray[np.floating]] = None,
        show_ylabel: bool = False
    ):
        """
        Create a bar chart showing calibration correctness over confidence bins.

        Args:
            ax: Matplotlib axes object
            title: Subplot title
            correctness: Correctness values for each bin
            bar_colors: List of colors for bars
            totals: Optional sample counts to label bars
            show_ylabel: Whether to show y-axis label
        """
        ax.set_title(title, fontsize=18, fontweight="bold")

        # Create bars
        bars = ax.bar(
            self.grid,
            correctness,
            width=self.bar_width,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.5
        )
        
        # Add perfect calibration reference line
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7, linewidth=1)
        
        # Add sample count labels if provided
        if totals is not None:
            ax.bar_label(bars, totals.astype(int), fontsize=6)
        
        # Configure axes
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.set_xlabel("Confidence")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        if show_ylabel:
            ax.set_ylabel("Correctness")
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.set_axisbelow(True)    
    
    def scatter_plot(
        self,
        ax: Axes,
        confidence: NDArray[np.floating],
        correctness: NDArray[np.floating],
        sizes: NDArray[np.floating],
        title: Optional[str] = None,
    ):
        """
        Create a scatter plot showing group-level calibration.

        Args:
            ax: Matplotlib axes object
            method: Calibration method name
            confidence: Confidence values for each group
            correctness: Correctness values for each group
            sizes: Dot sizes representing sample counts
        """
        
        if title: 
            ax.set_title(title, fontsize=18, fontweight="bold")
        
        colors = self.colors[:len(confidence)]

        ax.scatter(
            confidence,
            correctness,
            s=sizes,
            c=colors,
            alpha=0.7,
            marker="o",
            edgecolors='black',
            linewidths=0.5
        )
        
        # Add perfect calibration reference line
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", alpha=0.7, linewidth=1)
        
        # Configure axes
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Correctness")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.set_axisbelow(True)


    def calibration_method_comparison_bar_chart(
        self,
        scoring_method: str,
        totals: List[NDArray[np.floating]],
        correctness: List[NDArray[np.floating]]
    ):
        """
        Create comparison bar chart for all calibration methods.

        Args:
            scoring_method: Name of the scoring method used
            totals: List of total arrays for each method
            correctness: List of correctness arrays for each method
        """
        self._setup_plot_style()
        fig, axs = plt.subplots(1, 7, figsize=(24, 4), sharey=True)

        for idx, (ax, method, total, corr) in enumerate(
            zip(axs, self.CALIBRATION_METHODS, totals, correctness)
        ):
            is_uncalibrated = idx == 0
            colors = self._get_bar_colors(total, use_orange=is_uncalibrated)
            self.calibration_bar_chart(
                ax, method, corr, colors, show_ylabel=(idx == 0)
            )

        self._save_and_close(f"{scoring_method}_reliability_plot.pdf")



    def calibration_bar_scatter_chart(
        self,
        scoring_method: str,
        totals: List[NDArray[np.floating]],
        correctness_bins: List[NDArray[np.floating]],
        correctness_groups: List[NDArray[np.floating]],
        confidence_groups: List[NDArray[np.floating]],
        total_groups: List[NDArray[np.floating]]
    ):
        """
        Create combined bar and scatter plot comparison chart.

        Args:
            scoring_method: Name of the scoring method
            totals: List of bin totals for each method
            correctness_bins: List of bin correctness for each method
            correctness_groups: List of group correctness for each method
            confidence_groups: List of group confidence for each method
            total_groups: List of group totals for each method
        """
        self._setup_plot_style()
        fig, axs = plt.subplots(2, 7, figsize=(24, 8), sharey='row')

        # Top row: bar charts
        for idx, (ax, method, total, corr) in enumerate(
            zip(axs[0], self.CALIBRATION_METHODS, totals, correctness_bins)
        ):
            is_uncalibrated = idx == 0
            colors = self._get_bar_colors(total, use_orange=is_uncalibrated)
            self.calibration_bar_chart(
                ax, method, corr, colors, show_ylabel=(idx == 0)
            )

        # Bottom row: scatter plots
        for ax, conf, corr, total in zip(
            axs[1], confidence_groups, correctness_groups, total_groups
        ):
            self.scatter_plot(ax, conf, corr, total / 2)

        self._save_and_close(f"{scoring_method}_combined_plots.pdf") 


    
    def group_calibration_scatter(
        self,
        scoring_method: str,
        correctness_groups: List[NDArray[np.floating]],
        confidence_groups: List[NDArray[np.floating]],
        total_groups: List[NDArray[np.floating]]
    ):
        """
        Create scatter plot comparison for group-level calibration.

        Args:
            scoring_method: Name of the scoring method
            correctness_groups: List of group correctness for each method
            confidence_groups: List of group confidence for each method
            total_groups: List of group totals for each method
        """
        self._setup_plot_style()
        fig, axs = plt.subplots(1, 7, figsize=(24, 4), sharey=True)

        for ax, method, conf, corr, total in zip(
            axs, self.CALIBRATION_METHODS, confidence_groups, 
            correctness_groups, total_groups
        ):
            self.scatter_plot(ax, conf, corr, total / 2, title=method)

        self._save_and_close(f"{scoring_method}_group_calibration.pdf")
        
    
    def count_distribution(
        self,
        ax: Axes,
        title: str,
        totals: NDArray[np.floating]
    ):
        """
        Create a bar chart showing sample distribution across bins.

        Args:
            ax: Matplotlib axes object
            title: Subplot title
            totals: Array of sample counts per bin
        """
        ax.set_title(title, fontsize=12)
        ax.bar(
            self.grid,
            totals,
            width=self.bar_width,
            color=self.COLOR_CALIBRATED,
            edgecolor="black",
            linewidth=0.5
        )
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3, axis='y')    
    

    def calibration_info(
        self,
        total_uncalibrated: NDArray[np.floating],
        correctness_uncalibrated: NDArray[np.floating],
        total_calibrated: NDArray[np.floating],
        correctness_calibrated: NDArray[np.floating]
    ):
        """
        Create comprehensive calibration visualization with before/after comparison.

        Args:
            total_uncalibrated: Bin totals for uncalibrated data
            correctness_uncalibrated: Bin correctness for uncalibrated data
            total_calibrated: Bin totals for calibrated data
            correctness_calibrated: Bin correctness for calibrated data
        """
        self._setup_plot_style()
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        self.calibration_bar_chart(
            axs[0, 0],
            "Test Uncalibrated",
            correctness_uncalibrated,
            self._get_bar_colors(total_uncalibrated, use_orange=True),
            show_ylabel=True
        )
        self.calibration_bar_chart(
            axs[0, 1],
            "Test Calibrated",
            correctness_calibrated,
            self._get_bar_colors(total_calibrated),
            show_ylabel=True
        )

        self.count_distribution(
            axs[1, 0], "Test Uncalibrated Distribution", total_uncalibrated
        )
        self.count_distribution(
            axs[1, 1], "Test Calibrated Distribution", total_calibrated
        )

        self._save_and_close("calibration_infos.pdf")

    
    
    
    def create_charts(self, results: Dict[str, Dict[str, Any]], args):
        """
        Generate calibration bar charts and group scatter plots from results dictionary.

        Args:
            results: Dictionary containing results for all calibration methods
            args: Arguments object containing configuration (e.g., prob_method)
        """
        # Extract method results
        methods = [
            "Baseline",
            "Platt scaling",
            "Histogram binning",
            "Linear regression",
            "Logistic regression",
            "Iterative group histogram binning",
            "Iterative group linear binning"
        ]
        
        method_data = [results[method] for method in methods]
        
        # Extract data for bar charts
        totals = [method_data[0]["total_bin_uncalibrated"]] + [
            data["total_bin_calibrated"] for data in method_data[1:]
        ]
        correctness_bins = [method_data[0]["correctness_bin_uncalibrated"]] + [
            data["correctness_bin_calibrated"] for data in method_data[1:]
        ]
        
        # Extract data for scatter plots
        correctness_groups = [method_data[0]["correctness_group_uncalib"]] + [
            data["correctness_group"] for data in method_data[1:]
        ]
        confidence_groups = [method_data[0]["average_group_confidence_uncalib"]] + [
            data["average_group_confidence"] for data in method_data[1:]
        ]
        total_groups = [method_data[0]["total_group_uncalib"]] + [
            data["total_group"] for data in method_data[1:]
        ]
        
        # Generate charts
        
        self.calibration_bar_scatter_chart(
            args.prob_method,
            totals,
            correctness_bins,
            correctness_groups,
            confidence_groups,
            total_groups
        )
        
        
        self.calibration_method_comparison_bar_chart(
            args.prob_method,
            totals,
            correctness_bins
        )
        
        self.group_calibration_scatter(
            args.prob_method,
            correctness_groups,
            confidence_groups,
            total_groups
        )
    
        
    
