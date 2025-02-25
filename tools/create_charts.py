import matplotlib.pyplot as plt
import numpy as np

def calibration_comparision_chart(y, x1, x2 ,path, run):
    plt.title(run+' # Reliability chart', fontsize=7)
    plt.plot(y, x1, color="green")
    plt.plot(y, x2, color="red")
    plt.legend(["With HB", "Without Calibration"])
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('Confidence')
    plt.ylabel('Correct')
    plt.savefig(path)
    plt.close()

def histogram(data, path, run, typ):
    fig, ax = plt.subplots()  
    ax.hist(data, range=(0, 1.0))
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    plt.title(run+' # '+typ, fontsize=7)
    plt.savefig(path)
    print(f"Histogram saved: {path}")
    plt.close()



def calibration_bar_chart(y, x, path, run, width, bar_colors, totals):
    plt.title(run+' # Reliability chart', fontsize=7)
    bars = plt.bar(y, x, width = width, color=bar_colors, edgecolor='black')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.bar_label(bars, totals)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('Confidence')
    plt.ylabel('Correct')
    plt.savefig(path)
    plt.close()

def stacked_bar_plot(x, y1, y2, path, run, width):
    plt.bar(x, y1, color='g', width = width, edgecolor='black')
    plt.bar(x, y2, bottom=y1, color='r', width = width, edgecolor='black')
    plt.xlabel("Confidence")
    plt.ylabel("Anzahl")
    plt.legend(["Pass", "Fail"])
    plt.title(run, fontsize=7)
    plt.savefig(path)
    plt.close()