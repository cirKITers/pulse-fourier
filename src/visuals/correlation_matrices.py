import numpy as np
from matplotlib import pyplot as plt

from utils.helpers import custom_grey_colormap, custom_scientific_formatter


def corr_matr(correlation_mar, num_coeffs, num_params, print_every):
    plt.figure(figsize=(12, 10))
    plt.imshow(correlation_mar, cmap="viridis", aspect="auto")
    cbar = plt.colorbar()

    plt.xticks(np.arange(num_coeffs), [f"$c_{i + 1}$" for i in range(num_coeffs)], fontsize=25)
    y_ticks_indices = np.arange(0, num_params, print_every)
    y_tick_labels = [f"$\\theta_{{{i + 1}}}$" for i in y_ticks_indices]
    plt.yticks(ticks=y_ticks_indices, labels=y_tick_labels, fontsize=25)

    # --- MODIFICATION FOR SCIENTIFIC COLORBAR TICKS ---
    from matplotlib import ticker
    formatter = ticker.FuncFormatter(custom_scientific_formatter)
    cbar.ax.yaxis.set_major_formatter(formatter)

    cbar.ax.tick_params(labelsize=25)
    plt.tight_layout(pad=3.0)
    plt.show()

def custom_log_formatter(x, pos):
    # x is the log10 value. We want to display 10^x
    return f'$10^{{{int(x)}}}$' if x % 1 == 0 else f'$10^{{{x:.1f}}}$'

def diff_corr(correlation_differences, num_coeffs, num_params, print_every=2, log=False):
    correlation_differences = abs(correlation_differences)
    from matplotlib import ticker
    if log:
        correlation_differences = np.log10(correlation_differences)
        formatter = ticker.FuncFormatter(custom_log_formatter)
    else:
        formatter = ticker.FuncFormatter(custom_scientific_formatter)

    print(correlation_differences)
    # print("Element-wise difference in correlations:\n", correlation_differences)
    vmax_abs = np.max(np.abs(correlation_differences))
    vmin_abs = np.min(correlation_differences)

    plt.figure(figsize=(12, 10))
    plt.imshow(correlation_differences, cmap=custom_grey_colormap(), aspect='auto', vmin=vmin_abs, vmax=vmax_abs)
    plt.xticks(np.arange(num_coeffs), [f"$c_{i+1}$" for i in range(num_coeffs)], fontsize=25)
    y_ticks_indices = np.arange(0, num_params, print_every)
    y_tick_labels = [f"$\\theta_{{{i+1}}}$" for i in y_ticks_indices]
    plt.yticks(ticks=y_ticks_indices, labels=y_tick_labels, fontsize=25)
    cbar = plt.colorbar()

    # --- MODIFICATION FOR SCIENTIFIC COLORBAR TICKS ---
    cbar.ax.yaxis.set_major_formatter(formatter)

    cbar.ax.tick_params(labelsize=25)
    plt.tight_layout(pad=3.0)
    plt.show()


