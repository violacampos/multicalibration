import matplotlib.pyplot as plt
import numpy as np

class chart_creator():

    def __init__(self, run, binning_type,  grid, save_dir, m=None, bin_edges=None):
        self.run = run
        self.debug = binning_type
        self.save_dir = save_dir
        self.grid = grid

        self.colors_uncalibrated = []
        self.colors_calibrated = []

        # define chart ranges for display reasons
        if binning_type == 'linear':
            if not m: exit("m needs to be set with linear binning")
            self.chart_range = grid
            self.bar_width = 1/m 
        elif binning_type == 'quantil':
            if not bin_edges: exit("bin_edges needs to be set with quantil binning")
            self.chart_range = self.grid
            self.bar_width = np.array(bin_edges[1:] - bin_edges[:-1])            
            self.colors_uncalibrated = ['tab:blue']
            self.colors_calibrated = ['tab:blue']

    def set_bar_colors(self, total_uncalibrated, total_calibrated):
        total_bin_count_norm = (total_uncalibrated-np.min(total_uncalibrated))/(np.max(total_uncalibrated)-np.min(total_uncalibrated))
        for x in total_bin_count_norm:
            self.colors_uncalibrated.append((0.0, 0.0, 1.0, x))
            
        total_bin_count_norm = (total_calibrated-np.min(total_calibrated))/(np.max(total_calibrated)-np.min(total_calibrated))
        for x in total_bin_count_norm:
            self.colors_calibrated.append((0.0, 0.0, 1.0, x))

    def calibration_comparision_chart(self, y1, y2, x1, x2):
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
        plt.title(self.run+' # Reliability chart on test set', fontsize=7)
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
        plt.savefig(self.save_dir+"calibration_method_comparison.png")
        plt.close()

    def histogram(self, data, path, typ):
        fig, ax = plt.subplots()  
        ax.hist(data, range=(0, 1.0))
        ax.plot([0, 1], [0, 1], transform=ax.transAxes)
        plt.title(self.run+' # '+typ, fontsize=7)
        plt.savefig(path)
        print(f"Histogram saved: {path}")
        plt.close()

    def count_distribution(self, ax, sub_title, totals):
        ax.set_title(sub_title, fontsize=12)
        bars = ax.bar(self.chart_range, totals, width = self.bar_width, color=['tab:blue'], edgecolor='black')
        ax.bar_label(bars, totals)
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set(xlabel ='Confidence')
        ax.set(ylabel ='Count')

    def calibration_bar_chart(self, ax, sub_title, x, bar_colors):
        ax.set_title(sub_title, fontsize=12)
        bars = ax.bar(self.chart_range, x, width = self.bar_width, color=bar_colors, edgecolor='black')
        ax.plot([0, 1], [0, 1], linestyle='--')
        #ax.bar_label(bars, totals)
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set(xlabel ='Confidence')
        ax.set(ylabel ='Correct')

    def stacked_bar_plot(self, x, y1, y2, path, width):
        plt.bar(x, y1, color='g', width = width, edgecolor='black')
        plt.bar(x, y2, bottom=y1, color='r', width = width, edgecolor='black')
        plt.xlabel("Confidence")
        plt.ylabel("Anzahl")
        plt.legend(["Pass", "Fail"])
        plt.title(self.run, fontsize=7)
        plt.savefig(path)
        plt.close()

    def calibration_info(self, total_uncalibrated, correctness_uncalibrated, total_calibrated, correctness_calibrated):
        if len(correctness_calibrated) > 11:
            # auf 11 bins runterbrechen
            total_uncalibrated, total_calibrated, correctness_uncalibrated, correctness_calibrated = self.map_values_to_eleven_bins(total_uncalibrated, total_calibrated, correctness_uncalibrated, correctness_calibrated)

        self.set_bar_colors(total_uncalibrated, total_calibrated)

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle(self.run+' # Calibration Charts', fontsize=14)
        self.calibration_bar_chart(axs[0, 0], 'Test uncalibrated', correctness_uncalibrated, self.colors_uncalibrated)
        self.calibration_bar_chart(axs[0, 1], 'Test calibrated', correctness_calibrated, self.colors_calibrated)
        
        self.count_distribution(axs[1, 0], 'Test uncalibrated distribution', total_uncalibrated)
        self.count_distribution(axs[1, 1], 'Test calibrated distribution', total_calibrated)
        
        plt.savefig(self.save_dir+"calibration_infos.png")
        plt.close() 

        self.calibration_comparision_chart(self.chart_range[total_calibrated != 0], self.chart_range[total_uncalibrated != 0], correctness_calibrated[total_calibrated != 0], correctness_uncalibrated[total_uncalibrated != 0])

        self.reset_chart_range()

    def plot_correctness_change(self, total_uncalibrated, total_calibrated, correctness_uncalibrated, correctness_calibrated, change, num):
        if len(correctness_calibrated) > 11:
            # auf 11 bins runterbrechen
            total_uncalibrated, total_calibrated, correctness_uncalibrated, correctness_calibrated = self.map_values_to_eleven_bins(total_uncalibrated, total_calibrated, correctness_uncalibrated, correctness_calibrated)

        self.set_bar_colors(total_uncalibrated, total_calibrated)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(np.round(change,4), fontsize=14)
        self.calibration_bar_chart(axs[0], 'Before', correctness_uncalibrated, self.colors_uncalibrated)
        self.calibration_bar_chart(axs[1], 'After', correctness_calibrated, self.colors_calibrated)

        plt.savefig(self.save_dir+"changes/"+str(num)+"_calibration_changes.png")
        plt.close() 
        
        self.reset_chart_range()

    def map_values_to_eleven_bins(self, total_uncalibrated, total_calibrated, correctness_uncalibrated, correctness_calibrated):
        # auf 11 bins runterbrechen
        new_bins = np.arange(0, 1.1, 0.1)
        bin_assignment = np.array([new_bins[np.argmin(np.abs(old_bin - new_bins))] for old_bin in self.chart_range])

        total_calibrated = np.array([np.sum(total_calibrated[bin_assignment == nb]) for nb in new_bins])
        
        total_uncalibrated = np.array([np.sum(total_uncalibrated[bin_assignment == nb]) for nb in new_bins])

        correctness_uncalibrated = np.array([np.median(correctness_uncalibrated[bin_assignment == nb]) for nb in new_bins])
        
        correctness_calibrated = np.array([np.median(correctness_calibrated[bin_assignment == nb]) for nb in new_bins])

        self.chart_range = new_bins
        self.bar_width = 0.1

        return total_calibrated, total_uncalibrated, correctness_calibrated, correctness_uncalibrated
    
    def map_correctness_to_eleven_bins(self, correctness):
        # auf 11 bins runterbrechen
        new_bins = np.arange(0, 1.1, 0.1)
        bin_assignment = np.array([new_bins[np.argmin(np.abs(old_bin - new_bins))] for old_bin in self.chart_range])

        correctness = np.array([np.median(correctness[bin_assignment == nb], axis=0) for nb in new_bins])
        return correctness
    
    def map_total_to_eleven_bins(self, total):
        # auf 11 bins runterbrechen
        new_bins = np.arange(0, 1.1, 0.1)
        bin_assignment = np.array([new_bins[np.argmin(np.abs(old_bin - new_bins))] for old_bin in self.chart_range])

        total = np.array([np.sum(total[bin_assignment == nb]) for nb in new_bins])
        return total
    
    def map_total_to_eleven_bins(self, total, group=False):
        # auf 11 bins runterbrechen
        new_bins = np.arange(0, 1.1, 0.1)
        bin_assignment = np.array([new_bins[np.argmin(np.abs(old_bin - new_bins))] for old_bin in self.chart_range])

        total = np.array([np.sum(total[(bin_assignment == nb)], axis=0) if group else np.sum(total[(bin_assignment == nb)]) for nb in new_bins])

        return total
      
    def reset_chart_range(self):
        self.chart_range = self.grid

