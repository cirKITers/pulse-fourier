import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import stats
import random


def plot_fx(x_data, y_data, title="Plot",
            xlabel="Input x", ylabel="Predicted f(x)",
            uncertainty=None, statistical_analysis=False):
    """
    Plots f(x) with advanced features, including uncertainty visualization and
    optional statistical analysis.

    Args:
        x_data (array-like): The x-values.
        y_data (array-like): The y-values.
        title (str): Title of the plot.
        xlabel (str, optional): Label for the x-axis. Defaults to "Input x".
        ylabel (str, optional): Label for the y-axis. Defaults to "Predicted f(x)".
        uncertainty (array-like, optional): Uncertainty (e.g., standard deviation) for y_data.
        statistical_analysis (bool, optional): Perform and display basic statistical analysis. Defaults to False.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, label=title, color='blue', linewidth=2)

    if uncertainty is not None:
        plt.fill_between(x_data, y_data - uncertainty, y_data + uncertainty,
                         color='lightblue', alpha=0.5, label="Uncertainty")

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    if statistical_analysis:
        mean = np.mean(y_data)
        std = np.std(y_data)
        min_val = np.min(y_data)
        max_val = np.max(y_data)

        stats_text = f"Mean: {mean:.3f}\nStd: {std:.3f}\nMin: {min_val:.3f}\nMax: {max_val:.3f}"
        plt.text(0.05, 0.95, stats_text,
                 transform=plt.gca().transAxes, verticalalignment='top', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.7))

    plt.show()


# KIT green hexcode: #2e9481
kit_green = '#2e9481'
# KIT yellow hexcode: #f9e200
kit_yellow = '#f9e200'
kit_purple = '#9b187b'
kit_blue = '#3fa0e0'
kit_red = '#ca000e'
kit_orange = '#d99808'
kit_light_green = '#8db339'

def plot_2fx(x, fx1, fx2, label1="Pulse-Based", label2="Gate-Based",
             title="One Period of Expectation Values for Circuit 15",       # Hardware-Efficient
             xlabel="Sequence of Discrete Points {X}", ylabel="Quantum Fourier Expectation Value f",
             uncertainty1=None, uncertainty2=None,
             statistical_test=True):

    plt.figure(figsize=(12, 8))
    plt.plot(x, fx1, label=label1, color=kit_green, linewidth=2.5)
    plt.plot(x, fx2, label=label2, color=kit_yellow, linewidth=2.5)

    if uncertainty1 is not None:
        plt.fill_between(x, fx1 - uncertainty1, fx1 + uncertainty1, color='lightblue', alpha=0.5, label=f"Uncertainty {label1}")
    if uncertainty2 is not None:
        plt.fill_between(x, fx2 - uncertainty2, fx2 + uncertainty2, color='lightcoral', alpha=0.5, label=f"Uncertainty {label2}")

    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()

    if statistical_test and len(fx1) == len(fx2):
        t_stat, p_value = stats.ttest_ind(fx1, fx2)
        plt.text(0.778, 0.05, f"t-statistic: {t_stat:.3f}",
                 transform=plt.gca().transAxes, verticalalignment='bottom', fontsize=16,
                 bbox=dict(facecolor='white', alpha=0.7))

    plt.show()



# def plot_2fx_new(x, fx_gate, fx_pulse):
#     """
#     Plots two function sequences with specified KIT colors and a legend.
#     """
#
#     plt.figure(figsize=(12, 8))
#
#     plt.plot(x, fx_gate, color=kit_green, label='gate level')
#     plt.plot(x, fx_pulse, color=kit_yellow, label='pulse level')
#
#     plt.title("Circuit HEA", fontsize=16)
#     plt.xlabel("X-axis", fontsize=14)
#     plt.ylabel("Y-axis", fontsize=14)
#     plt.legend(fontsize=12)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.tight_layout()
#     plt.show()


def plot_nfx(x, fx_list, random_color=False):
    """
    Plots a list of functions with gradually different colors.
    """
    num_fx = len(fx_list)
    if num_fx == 0:
        return

    plt.figure(figsize=(12, 8))

    # gradual
    cmap = plt.cm.get_cmap('viridis', num_fx)

    # random
    colors = [(random.random(), random.random(), random.random()) for _ in range(num_fx)]

    for i, fx in enumerate(fx_list):
        if random_color:
            color = colors[i]
        else:
            color = cmap(i / num_fx)
        plt.plot(x, fx, color=color)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_fx_old(x_data, y_data, title):
    plt.figure(figsize=(8, 6))
    plt.plot(x_data, y_data, label=title, color='b')
    plt.title(title)
    plt.xlabel("Input x")
    plt.ylabel("Predicted f(x)")
    plt.grid(True)
    plt.legend()
    plt.show()