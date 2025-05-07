import numpy as np
from matplotlib import pyplot as plt

from fft.coefficients import order_coefficients_sets, coefficient_set
from qft_models.circuit_9 import Circuit9

from utils.helpers import random_parameter_set2
from utils.data_handler import save, load
from visuals.fx import plot_nfx

# Function parameter
start = 0
stop = 20
points = 1000  # 1000
x = np.linspace(start, stop, points)

# Samples
num_samples = 10

# Hyper parameter
num_qubits = 4  # scale
num_ansatz = 2  # const, 1 layer


# Model
model = Circuit9(num_qubits)

# Parameter
parameter_set = random_parameter_set2(num_samples, 2, num_qubits, len(["RX"]), seed=9)

# MODEL RUN
fx_set = model.sample_fourier(x, parameter_set, num_samples)

# Save function
# save("Test_circuit9_Random", num_qubits, 1, num_samples, start, stop, points, x, fx_set, "test/")

plot_nfx(x, fx_set, random_color=True)



num_qubits = 4




# plot_nfx(loaded_x, loaded_fx_set, random_color=True)


# --- Data Preparation --- DEFINE NUMBER OF PARAMS AND SEED CORRECTLY DEPENDING ON CIRCUIT

num_params = 8
parameter_set = random_parameter_set2(num_samples, 2, num_qubits, len(["RX"]), seed=9)
# parameter_set = random_parameter_set2(num_samples, 2, num_qubits, len(["RY", "RY"]), seed=15)


# Flatten out ansatz
param_array = np.array([p.flatten() for p in parameter_set])


# --- Data Preparation --- DECIDE NUMBER COEFFS

num_coeffs = 30
a, b, c = coefficient_set(fx_set, num_coeff=num_coeffs)
complex_coeffs = c


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
plt.title("Test C9"+" Correlation Heatmap")
plt.tight_layout()
plt.show()



