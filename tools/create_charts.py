import matplotlib.pyplot as plt
import numpy as np

def calibration_comparision_chart(y1, y2, x1, x2 ,path, run):
    plt.title(run+' # Reliability chart', fontsize=7)
    plt.plot(y1, x1, color="green")
    plt.plot(y2, x2, color="red")
    plt.legend(["Test", "Train"])
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

def count_distribution(ax, sub_title, y, totals, width):
    #fig, ax = plt.subplots() 
    ax.set_title(sub_title, fontsize=12)
    bars = ax.bar(y, totals, width = width, color=['tab:blue'], edgecolor='black')
    ax.bar_label(bars, totals)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set(xlabel ='Confidence')
    ax.set(ylabel ='Count')
    #return ax
    #plt.savefig(path)
    #plt.close()  

def calibration_bar_chart(ax, sub_title, y, x, width, bar_colors, totals):
    #fig, ax = plt.subplots() 
    ax.set_title(sub_title, fontsize=12)
    bars = ax.bar(y, x, width = width, color=bar_colors, edgecolor='black')
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.bar_label(bars, totals)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set(xlabel ='Confidence')
    ax.set(ylabel ='Correct')
    #return ax
    #plt.savefig(path)
    #plt.close()

def stacked_bar_plot(x, y1, y2, path, run, width):
    plt.bar(x, y1, color='g', width = width, edgecolor='black')
    plt.bar(x, y2, bottom=y1, color='r', width = width, edgecolor='black')
    plt.xlabel("Confidence")
    plt.ylabel("Anzahl")
    plt.legend(["Pass", "Fail"])
    plt.title(run, fontsize=7)
    plt.savefig(path)
    plt.close()
