from matplotlib import pyplot as plt

from fft.coefficients import coefficient_set, order_coefficients_sets
from utils.data_handler import load
from utils.helpers import random_parameter_set2
from visuals.coefficients import subplot
from visuals.fx import plot_nfx

import pandas as pd
import numpy as np


# const
num_samples = 5000
num_qubits = 4

circuit_name = "Circuit 9"
file_to_load = "../results/c9_exp/gate/Circuit9_Random_4qubits_1layers_5000samples_0start_20stop_1000points.json"
loaded_x, loaded_fx_set = load(file_to_load)


# plot_nfx(loaded_x, loaded_fx_set, random_color=True)


# --- Data Preparation --- DEFINE NUMBER OF PARAMS AND SEED CORRECTLY DEPENDING ON CIRCUIT
num_params = 8
parameter_set = random_parameter_set2(num_samples, 2, num_qubits, len(["RX"]), seed=9)
# parameter_set = random_parameter_set2(num_samples, 2, num_qubits, len(["RY", "RY"]), seed=15)
# parameter_set = random_parameter_set2(num_samples, 2, num_qubits, len(["RY", "RZ", "RY"]), seed=42)

# Flatten out ansatz
param_array = np.array([p.flatten() for p in parameter_set])


# --- Data Preparation --- DECIDE NUMBER COEFFS

num_coeffs = 30
a, b, c = coefficient_set(loaded_fx_set, num_coeff=num_coeffs)
complex_coeffs = c



# X = np.fft.fft(loaded_fx_set[0]) / len(loaded_fx_set[0])
# X_shift = np.fft.fftshift(X)
# X_freq = np.fft.fftfreq(X.size, 1/6)
# X_freq_shift = np.fft.fftshift(X_freq)


# fig, ax = plt.subplots()
# ax.stem(X_freq_shift, np.abs(X_shift)/X_shift.size)
# plt.xlabel('Frequency')
# plt.ylabel('Amplitude')
# plt.show()




order_coefficients_sets(complex_coeffs)


coeff_magnitudes = np.abs(complex_coeffs)  # Calculate magnitudes
# subplot(a, b)



# --- Analysis ---

# 1. Correlation Analysis
correlations = np.corrcoef(param_array.T, coeff_magnitudes.T)[
    :num_params, num_params:
]  # Shape: (8, 7)
# print("Correlation between parameters and coefficient magnitudes:\n", correlations)


# 2. Visualization
#   - Scatter Plots
# for coeff_index in range(num_coeffs):
#     plt.figure(figsize=(12, 8))
#     for i in range(num_params):
#         plt.subplot(2, int(num_params/2), i + 1)  # Assuming 2x4 parameter grid
#         plt.scatter(param_array[:, i], coeff_magnitudes[:, coeff_index], alpha=0.05)
#         plt.xlabel(f"Param {i+1}")
#         # plt.ylabel(f"|Coeff {coeff_index+1}|")
#     plt.tight_layout()
#     plt.show()

#   - Heatmap of Correlations
plt.figure(figsize=(8, 6))
plt.imshow(correlations, cmap="coolwarm", aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(num_coeffs), [f"Coeff {i+1}" for i in range(num_coeffs)])
plt.yticks(range(num_params), [f"Param {i+1}" for i in range(num_params)])
plt.title(circuit_name+" Correlation Heatmap")
plt.tight_layout()
plt.show()


