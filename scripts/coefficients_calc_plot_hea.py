from matplotlib import pyplot as plt

from fft.coefficients import coefficient_set, order_coefficients_sets
from qft_models.circuit_15 import Circuit15
from qft_models.circuit_hea import CircuitHE
from utils.data_handler import load, load_and_combine_fx_sets
from utils.helpers import random_parameter_set2, combine_parameter_sets, custom_grey_colormap, custom_scientific_formatter
from visuals.coefficients import subplot
from visuals.correlation_matrices import diff_corr, corr_matr
from visuals.fx import plot_2fx

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('TkAgg')

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
num_samples = 10
start = 0
stop = 6.283185307179586
points = 9
seed_start = 1
seed_stop = 501


directory_path_gate = "../results/hea_exp/gate/"
model_name_gate = "CircuitHE_Random"
fx_set_gate = load_and_combine_fx_sets(directory_path_gate, model_name_gate, num_qubits, num_layers, num_samples, start, stop, points, seed_start, seed_stop)
if fx_set_gate is not None:
    print("Combined fx_set shape:", fx_set_gate.shape)


directory_path_pulse = "../results/hea_exp/pulse/"
model_name_pulse = "PulseHEA_Random_Parallel_Joblib"
fx_set_pulse = load_and_combine_fx_sets(directory_path_pulse, model_name_pulse, num_qubits, num_layers, num_samples, start, stop, points, seed_start, seed_stop)
if fx_set_pulse is not None:
    print("Combined fx_set shape:", fx_set_pulse.shape)


# --- Data Preparation --- DEFINE NUMBER OF PARAMS AND SEED CORRECTLY DEPENDING ON CIRCUIT
num_params = 24
# parameter_set = random_parameter_set2(num_samples, 2, num_qubits, len(["RX"]), seed=9)
# parameter_set = random_parameter_set2(num_samples, 2, num_qubits, len(["RY", "RY"]), seed=15)
# parameter_set = random_parameter_set2(num_samples, 2, num_qubits, len(["RY", "RZ", "RY"]), seed=42)

seeds = list(range(1, 501))  # Seeds from 10 to 109
parameter_set = combine_parameter_sets(num_samples, 2, num_qubits, len(["RY", "RZ", "RY"]), seeds)
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
# print(param_array.T.shape)
# print(magnitude_pulse.T.shape)
# print(magnitude_gate.T.shape)

# 1. Correlation Analysis
correlations_pulse = np.corrcoef(param_array.T, magnitude_pulse.T)[
    :num_params, num_params:
]  # Shape: (params, coeffs)
# print("Correlation between parameters and coefficient magnitudes:\n", correlations)
correlations_gate = np.corrcoef(param_array.T, magnitude_gate.T)[
    :num_params, num_params:
]  # Shape: (params, coeffs)

correlations_pulse = np.abs(correlations_pulse)
correlations_gate = np.abs(correlations_gate)

# Mean Absolute Error of Correlations
mae_correlation = np.mean(np.abs(correlations_pulse - correlations_gate))
print(f"Mean Absolute Error (MAE) of Correlations: {mae_correlation}")

# DIFFERENCE
correlation_differences = correlations_pulse - correlations_gate
# diff_corr(correlation_differences, num_coeffs, num_params, 2)

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

corr_matr(correlations_pulse, num_coeffs, num_params, 2)



