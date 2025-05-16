from matplotlib import pyplot as plt

from fft.coefficients import coefficient_set, order_coefficients_sets
from qft_models.circuit_15 import Circuit15
from qft_models.circuit_hea import CircuitHE
from utils.data_handler import load, load_and_combine_fx_sets
from utils.helpers import random_parameter_set2, combine_parameter_sets
from visuals.coefficients import subplot
from visuals.fx import plot_2fx

import pandas as pd
import numpy as np


# const
# num_samples = 5000


# # PLOTTING OF FOURIER SERIES
# num_qubits = 4
# # file_to_load_single_sample = "../results/hea_exp/pulse/PulseHEA_Random_Parallel_Joblib_4qubits_1layers_1samples_0start_6.283185307179586stop_101points_42seed.json"
# file_to_load_single_sample = "../results/c15_exp/pulse/Pulse15_Random_Parallel_Joblib_4qubits_1layers_1samples_0start_6.283185307179586stop_101points_15seed.json"
# loaded_x_copy, loaded_fx_set_hea = load(file_to_load_single_sample)
#
# # Model
# model = Circuit15(num_qubits)
# # model = CircuitHE(num_qubits)
#
# # Parameter
# # parameter_set = random_parameter_set2(1, 2, num_qubits, len(["RY", "RZ", "RY"]), seed=42)
# parameter_set = random_parameter_set2(1, 2, num_qubits, len(["RY", "RY"]), seed=15)
# # MODEL RUN
# fx_set = model.sample_fourier(loaded_x_copy, parameter_set, 1)
#
#
# print(loaded_x_copy)
# print(loaded_fx_set_hea)
# plot_2fx(loaded_x_copy, loaded_fx_set_hea[0], fx_set[0])




# COMPOSING OF MULTIPLE FILES
num_qubits = 4
num_layers = 1
num_samples = 50
start = 0
stop = 6.283185307179586
points = 9


directory_path_gate = "../results/c9_exp/gate/"
model_name_gate = "Circuit9_Random"
fx_set_gate = load_and_combine_fx_sets(directory_path_gate, model_name_gate, num_qubits, num_layers, num_samples, start, stop, points)
# if combined_fx_set_gate is not None:
#     print("Combined fx_set shape:", combined_fx_set_gate.shape)


directory_path_pulse = "../results/c9_exp/pulse/"
model_name_pulse = "Pulse9_Random_Parallel_Joblib"
fx_set_pulse = load_and_combine_fx_sets(directory_path_pulse, model_name_pulse, num_qubits, num_layers, num_samples, start, stop, points)
# if combined_fx_set_pulse is not None:
#     print("Combined fx_set shape:", combined_fx_set_pulse.shape)


# --- Data Preparation --- DEFINE NUMBER OF PARAMS AND SEED CORRECTLY DEPENDING ON CIRCUIT
num_params = 8
# parameter_set = random_parameter_set2(num_samples, 2, num_qubits, len(["RX"]), seed=9)
# parameter_set = random_parameter_set2(num_samples, 2, num_qubits, len(["RY", "RY"]), seed=15)
# parameter_set = random_parameter_set2(num_samples, 2, num_qubits, len(["RY", "RZ", "RY"]), seed=42)

seeds = list(range(10, 110))  # Seeds from 10 to 109
parameter_set = combine_parameter_sets(num_samples, 2, num_qubits, len(["RX"]), seeds)
# print("Combined parameter set length:", len(combined_parameter_set))
# print("Shape of each parameter set:", combined_parameter_set[0].shape)


# Flatten out ansatz
param_array = np.array([p.flatten() for p in parameter_set])


# --- Data Preparation --- DECIDE NUMBER COEFFS
num_coeffs = 5
a_pulse, b_pulse, c_pulse = coefficient_set(fx_set_pulse, num_coeff=num_coeffs)
a_gate, b_gate, c_gate = coefficient_set(fx_set_gate, num_coeff=num_coeffs)

# subplot(a_gate, b_gate)
# subplot(a_pulse, b_pulse)


order_coefficients_sets(c_pulse)
order_coefficients_sets(c_gate)

magnitude_pulse = np.abs(c_pulse)
magnitude_gate = np.abs(c_gate)


approximation = magnitude_pulse
label = magnitude_gate
mae = np.mean(np.abs(approximation - label))
print(f"Mean Absolute Error (MAE): {mae}")

frequencies = np.arange(5)
relative_error_spectrum = np.abs(approximation - label) / (np.abs(label) + 1e-8)






# import seaborn as sns  # For KDE plots
#
# for i in range(5):
#     plt.figure(figsize=(8, 5))
#     sns.histplot(label[:, i], label="Gate-based", kde=True)
#     sns.histplot(approximation[:, i], label="Pule-based", kde=True)
#     plt.xlabel("Magnitude")
#     plt.title(f"Distribution for Correlation {i + 1}")
#     plt.legend()
#     plt.show()



# --- Analysis ---
from matplotlib.colors import LinearSegmentedColormap
def custom_grey_colormap(levels=256):
    """
    Creates a custom colormap with white in the middle and light grey
    moving towards positive and negative extremes.

    Args:
        levels (int): Number of color levels in the colormap.

    Returns:
        matplotlib.colors.LinearSegmentedColormap: The custom colormap.
    """
    midpoint = 0.5
    cdict = {
        'red':   [(0.0, 0.9, 0.9),  # Start near white (adjust as needed)
                  (midpoint, 1.0, 1.0),  # White at the midpoint
                  (1.0, 0.9, 0.9)],  # End near white
        'green': [(0.0, 0.9, 0.9),
                  (midpoint, 1.0, 1.0),
                  (1.0, 0.9, 0.9)],
        'blue':  [(0.0, 0.9, 0.9),
                  (midpoint, 1.0, 1.0),
                  (1.0, 0.9, 0.9)],
        'alpha': [(0.0, 1.0, 1.0),
                  (midpoint, 1.0, 1.0),
                  (1.0, 1.0, 1.0)]
    }
    cmap = LinearSegmentedColormap('custom_grey', cdict, N=levels)
    return cmap

# 1. Correlation Analysis
correlations_pulse = np.corrcoef(param_array.T, magnitude_pulse.T)[
    :num_params, num_params:
]  # Shape: (params, coeffs)
# print("Correlation between parameters and coefficient magnitudes:\n", correlations)
correlations_gate = np.corrcoef(param_array.T, magnitude_gate.T)[
    :num_params, num_params:
]  # Shape: (params, coeffs)

# DIFFERENCE
# correlation_differences = correlations_pulse - correlations_gate
# print("Element-wise difference in correlations:\n", correlation_differences)
# vmax_abs = np.max(np.abs(correlation_differences))
# plt.figure(figsize=(8, 6))
# plt.imshow(correlation_differences, cmap=custom_grey_colormap(), aspect='auto', vmin=-vmax_abs, vmax=vmax_abs)
# plt.title('Difference in Correlations (Pulse - Gate)', fontsize=16)
# plt.xticks(np.arange(num_coeffs), [f"Coeff {i}" for i in range(num_coeffs)], fontsize=16)
# plt.yticks(np.arange(num_params), [f"Param {i+1}" for i in range(num_params)], fontsize=16)
# cbar = plt.colorbar(label='Correlation Difference')
# cbar.ax.tick_params(labelsize=16)
# cbar.set_label('Correlation Difference', fontsize=16)
# plt.tight_layout()
# plt.show()


# 2. Visualization
#   - Scatter Plots
# for coeff_index in range(num_coeffs):
#     plt.figure(figsize=(12, 8))
#     for i in range(num_params):
#         plt.subplot(2, int(num_params/2), i + 1)  # Assuming 2x4 parameter grid
#         plt.scatter(param_array[:, i], magnitude_gate[:, coeff_index], alpha=0.05)
#         plt.xlabel(f"Param {i+1}")
#         # plt.ylabel(f"|Coeff {coeff_index+1}|")
#     plt.tight_layout()
#     plt.show()

#   - Heatmap of Correlations
plt.figure(figsize=(8, 6))
plt.imshow(correlations_pulse, cmap="coolwarm", aspect="auto")
cbar = plt.colorbar(label="Correlation")
cbar.ax.tick_params(labelsize=16)
cbar.set_label("Correlation", fontsize=16)
plt.xticks(range(num_coeffs), [f"Coeff {i}" for i in range(num_coeffs)], fontsize=16)
plt.yticks(range(num_params), [f"Param {i+1}" for i in range(num_params)], fontsize=16)
plt.title("Circuit 9 Gate-Based Correlation Heatmap", fontsize=16)
plt.tight_layout()
plt.show()

# Second
plt.figure(figsize=(8, 6))
plt.imshow(correlations_gate, cmap="coolwarm", aspect="auto")
cbar = plt.colorbar(label="Correlation")
cbar.ax.tick_params(labelsize=16)
cbar.set_label("Correlation", fontsize=16)
plt.xticks(range(num_coeffs), [f"Coeff {i}" for i in range(num_coeffs)], fontsize=16)
plt.yticks(range(num_params), [f"Param {i+1}" for i in range(num_params)], fontsize=16)
plt.title("Circuit 9 Pulse-Based Correlation Heatmap", fontsize=16)
plt.tight_layout()
plt.show()

