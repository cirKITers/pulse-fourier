import math

import numpy as np
from matplotlib import pyplot as plt

from fft.coefficients import order_coefficients_sets, coefficient_set
from qft_models.circuit_15 import Circuit15
from qft_models.circuit_9 import Circuit9
from qft_models.circuit_hea import CircuitHE

from utils.helpers import random_parameter_set2
from utils.data_handler import save, load
from visuals.coefficients import subplot
from visuals.fx import plot_nfx

# Function parameter
# start = 0
# stop = 20
# points = 1000  # 1000

# four relevant coeffs
# x = np.linspace(start, stop, points)
excluding_discrete_points = 8  # len(x) is plus one (including interval length value)!
interval_length = 2 * np.pi
delta = interval_length / excluding_discrete_points
x = np.arange(0, interval_length + delta, delta)
# print(len(x))
# print(x)
# Samples
num_samples = 30

# Hyper parameter
num_qubits = 4  # scale
num_ansatz = 2  # const, 1 layer


# --- Data Preparation --- DEFINE NUMBER OF PARAMS AND SEED CORRECTLY DEPENDING ON CIRCUIT

# Model
model_name = "Test C9"
model = Circuit9(num_qubits)

# Parameter
parameter_set = random_parameter_set2(num_samples, 2, num_qubits, len(["RX"]), seed=9)
# parameter_set = random_parameter_set2(num_samples, 2, num_qubits, len(["RY", "RY"]), seed=15)
# parameter_set = random_parameter_set2(num_samples, 2, num_qubits, len(["RY", "RZ", "RY"]), seed=42)

num_params = 8
# p
# Flatten out ansatz
param_array = np.array([p.flatten() for p in parameter_set])


# MODEL RUN
fx_set = model.sample_fourier(x, parameter_set, num_samples)

# Save function
# save("Test_circuit9_Random", num_qubits, 1, num_samples, start, stop, points, x, fx_set, "test/")

# plot_nfx(x, fx_set, random_color=True)


num_qubits = 4


# plot_nfx(x, fx_set, random_color=True)


# --- Data Preparation --- DECIDE NUMBER COEFFS

num_coeffs = math.floor(len(x)/2) + 1

c = coefficient_set(fx_set, num_coeff=num_coeffs)
complex_coeffs = c

ordererd_coefficients = order_coefficients_sets(complex_coeffs)

coeff_magnitudes = np.abs(complex_coeffs)  # Calculate magnitudes


# SUBPLOT MAGNITUDES
a = 2 * np.real(c)  # describes cosinus part
b = -2 * np.imag(c)  # describes sinus part
a[0] = a[0] / 2       # to get rid of the *2 from above, always practically 0
subplot(a, b)

# print(len(coeff_magnitudes))


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
plt.xticks(range(num_coeffs), [f"Coeff {i}" for i in range(num_coeffs)])
plt.yticks(range(num_params), [f"Param {i+1}" for i in range(num_params)])
plt.title(model_name+" Correlation Heatmap")
plt.tight_layout()
plt.show()



