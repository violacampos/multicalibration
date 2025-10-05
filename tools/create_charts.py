import matplotlib.pyplot as plt
import numpy as np

class charts():

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
        
        self.colors = [item for sublist in [cm.colors for cm in cmaps] for item in sublist]
        

        # self.colors = [  
        #                 'tab:blue',
        #                 'tab:orange',
        #                 'tab:green',
        #                 'tab:red',
        #                 'tab:purple',
        #                 'tab:brown',
        #                 'tab:pink',
        #                 'tab:gray',
        #                 'tab:olive',
        #                 'tab:cyan',
        #                 'yellow', 
        #                 'indigo', 
        #                 'violet', 
        #                 'navy', 
        #                 'teal', 
        #                 'maroon', 
        #                 'silver', 
        #                 'tan', 
        #                 'gold', 
        #                 'purple',
        #                 'moccasin', 
        #                 'bisque', 
        #                 'wheat', 
        #                 'peachpuff', 
        #                 'navajowhite', 
        #                 'salmon', 
        #                 'crimson'
        #             ]

        # define chart ranges for display reasons
        if binning_type == 'linear':
            self.chart_range = grid
            self.bar_width = 1/(len(grid)-1)
        elif binning_type == 'quantil':
            if not bin_edges: exit("bin_edges needs to be set with quantil binning")
            self.chart_range = self.grid
            self.bar_width = np.array(bin_edges[1:] - bin_edges[:-1])            
            self.colors_uncalibrated = ['tab:blue']
            self.colors_calibrated = ['tab:blue']

    def get_bar_colors(self, total):
        """
            Calculates the shade of blue for each bar, based on the amount of samples in each bin.

            :param total: List of total values for each bin of the data

            :return: List of color tuples
        """  
        colors = [] 
        total_bin_count_norm = (total-np.min(total))/(np.max(total)-np.min(total))
        
        for x in total_bin_count_norm:
           colors.append((0.0, 0.0, 1.0, x))

        return colors

    def calibration_comparision_chart(self, y1, y2, x1, x2):
        """
            Creates a line chart with the uncalibrated data and the calibrated data.

            :param y1: Grid points where the total is not null for uncalibrated data
            :param y2: Grid points where the total is not null for calibrated data
            :param x1: List of correctness values for each bin of the data
            :param x2: List of correctness values for each bin of the data
        """  
        plt.title(self.run+' # Reliability chart on test set', fontsize=7)
        plt.plot(y1, x1, color="green")
        plt.plot(y2, x2, color="red")
        plt.legend(["Calibrated", "Uncalibrated"])
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('Confidence')
        plt.ylabel('Correct')
        plt.savefig(self.save_dir+"calibration_comparison.png")
        plt.close()

    def calibration_method_comp_chart(self, y1, y2, y3, y4, y5, x1, x2, x3, x4, x5):
        """
            Creates a line plot with all calibration methods for comparison

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
        plt.plot(y1, x1, color="r")
        plt.plot(y2, x2, color="b")
        plt.plot(y3, x3, color="g")
        plt.plot(y4, x4, color="c")
        plt.plot(y5, x5, color="m")
        plt.legend(["Uncalibrated", "HB", "LR", "IGHB", "IGLB"])
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('Confidence')
        plt.ylabel('Correct')
        plt.savefig(self.save_dir+"calibration_method_comparison.pdf")
        plt.close()

    def calibration_method_comp_bar_chart(self, y1, y2, y3, y4, y5, x1, x2, x3, x4, x5):
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
        fig, axs = plt.subplots(3, 2, figsize=(10, 15))
        
        self.calibration_bar_chart(axs[0, 0], 'Uncalibrated', x1, self.get_bar_colors(y1), y1)        
        self.calibration_bar_chart(axs[1, 0], 'HB', x2, self.get_bar_colors(y2), y2)    
        self.calibration_bar_chart(axs[1, 1], 'LR', x3, self.get_bar_colors(y3), y3)    
        self.calibration_bar_chart(axs[2, 0], 'IGHB', x4, self.get_bar_colors(y4), y4)    
        self.calibration_bar_chart(axs[2, 1], 'IGLB', x5, self.get_bar_colors(y5), y5)
        axs[0,1].axis('off')
        plt.savefig(self.save_dir+"calibration_comparison_bar.pdf")
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
        bars = ax.bar(self.grid, totals, width = self.bar_width, color=['tab:blue'], edgecolor='black')
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set(xlabel ='Confidence')
        ax.set(ylabel ='Count')

    def calibration_bar_chart(self, ax, sub_title, x, bar_colors, totals=None):
        """
            Creates a bar chart with the correctness values to display the calibration over the bins.

            :param ax: ax object to create chart on
            :param sub_title: Title of the subplot
            :param x: List of correctness values for each bin
            :param bar_colors: List of colors for the bars
            :param totals: Optional to add totals to each bar
        """  
        ax.set_title(sub_title, fontsize=12, fontweight="bold")
        bars = ax.bar(self.grid, x, width = self.bar_width, color=bar_colors, edgecolor='black')
        ax.plot([0, 1], [0, 1], linestyle='--')
        if totals is not None:
            ax.bar_label(bars, totals, fontsize=6)
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set(xlabel ='Confidence')
        ax.set(ylabel ='Correct')

    def scatter_plot(self, ax, method, x, y, area):
        """
            Creates a scatter plot to display the calibration within each group for a calibration approach.

            :param ax: ax object to create chart on
            :param method: Name of the use calibration method
            :param x: List of confidence values for each group
            :param y: List of correctness values for each group
            :param area: Size of the dot. Represents the amount of samples in the group.
        """  
        colors = self.colors[:len(x)]

        scatter = ax.scatter(x, y, s=area, c=colors, alpha=0.7, marker=r'$\odot$')
        ax.set_title(method, fontsize=12, fontweight="bold")
        ax.plot([0.150, 0.8], [0.150, 0.8], linestyle='--')
        ax.set(xlabel ='Confidence')
        ax.set(ylabel ='Correct')   

    def group_calibration_scatter(self, 
                                  uncalib_corr,
                                  uncalib_conf,
                                  uncalib_total, 
                                  hb_corr,
                                  hb_conf,
                                  hb_total, 
                                  lr_corr,
                                  lr_conf,
                                  lr_total, 
                                  ighb_corr,
                                  ighb_conf,
                                  ighb_total, 
                                  iglb_corr,
                                  iglb_conf,
                                  iglb_total):
        """
            Creates a plot to display the calibration within each group for all calibration approaches.

            :param ..._corr: correctness for each group of the given method
            :param ..._conf: confidence for each group of the given method
            :param ..._total: total for each group of the given method
        """          
        fig, axs = plt.subplots(3, 2, figsize=(10, 15))

        self.scatter_plot(axs[0, 0], 'Uncalibrated', uncalib_conf, uncalib_corr, uncalib_total)        
        self.scatter_plot(axs[1, 0], 'HB', hb_conf, hb_corr, hb_total)    
        self.scatter_plot(axs[1, 1], 'LR', lr_conf, lr_corr, lr_total)    
        self.scatter_plot(axs[2, 0], 'IGHB', ighb_conf, ighb_corr, ighb_total)    
        self.scatter_plot(axs[2, 1], 'IGLB', iglb_conf, iglb_corr, iglb_total)
        axs[0,1].axis('off')
        plt.savefig(self.save_dir+"group_calibration.pdf")
        plt.close()

    def calibration_info(self, total_uncalibrated, correctness_uncalibrated, total_calibrated, correctness_calibrated):
        """
            Creates multiple subplots to visualize the calibration changes of a method

            :param total_uncalibrated: total values for the uncalibrated data
            :param correctness_uncalibrated: correcntess values for the uncalibrated data
            :param total_calibrated: total values for the calibrated data
            :param correctness_calibrated: correcntess values for the calibrated data
        """    
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        self.calibration_bar_chart(axs[0, 0], 'Test uncalibrated', correctness_uncalibrated, self.get_bar_colors(total_uncalibrated))
        self.calibration_bar_chart(axs[0, 1], 'Test calibrated', correctness_calibrated, self.get_bar_colors(total_calibrated))
        
        self.count_distribution(axs[1, 0], 'Test uncalibrated distribution', total_uncalibrated)
        self.count_distribution(axs[1, 1], 'Test calibrated distribution', total_calibrated)
        
        plt.savefig(self.save_dir+"calibration_infos.pdf")
        plt.close() 

        #self.calibration_comparision_chart(self.chart_range[total_calibrated != 0], self.chart_range[total_uncalibrated != 0], correctness_calibrated[total_calibrated != 0], correctness_uncalibrated[total_uncalibrated != 0])