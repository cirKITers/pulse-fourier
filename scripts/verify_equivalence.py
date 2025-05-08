import numpy as np

from qft_models.circuit_15 import Circuit15
from qft_models.circuit_9 import Circuit9
from qft_models.circuit_hea import CircuitHE
from qft_models.pulse_15 import Pulse15
from qft_models.pulse_9 import Pulse9
from qft_models.pulse_hea import PulseHE

from utils.helpers import random_parameter_set2
from utils.data_handler import save
from visuals.fx import plot_nfx

# Function parameter
# start = 0
# stop = 20
# points = 1000  # 1000
# x = np.linspace(start, stop, points)

excluding_discrete_points = 2  # len(x) is plus one (including interval length value)!
interval_length = 2 * np.pi
delta = interval_length / excluding_discrete_points
x = np.arange(0, interval_length + delta, delta)
start = 0
stop = interval_length
points = excluding_discrete_points + 1

# Samples
num_samples = 1

# Hyper parameter
num_qubits = 4  # scale
num_ansatz = 2  # const, 1 layer


# Parameter
# parameter_set = random_parameter_set2(num_samples, 2, num_qubits, len(["RX"]), seed=9)
# parameter_set = random_parameter_set2(num_samples, 2, num_qubits, len(["RY", "RY"]), seed=15)
parameter_set = random_parameter_set2(num_samples, 2, num_qubits, len(["RY", "RZ", "RY"]), seed=42)


# Model
model_gate = CircuitHE(num_qubits)
# MODEL RUN
fx_set_gate = model_gate.sample_fourier(x, parameter_set, num_samples)


# Model
model = PulseHE(num_qubits)
# MODEL RUN
fx_set_pulse = model.sample_fourier(x, parameter_set, num_samples)

plot_nfx(x, [fx_set_gate[0], fx_set_pulse[0]])

tolerance = 1e-3

difference_matrices = []
for i in range(len(fx_set_pulse)):
    pulse_sequence = fx_set_pulse[i]
    gate_sequence = fx_set_gate[i]

    if pulse_sequence.shape != gate_sequence.shape:
        print(f"Warning: Shapes of sequences at index {i} do not match.")
        difference_matrices.append(None)  # Or handle differently
        continue

    absolute_differences = np.abs(pulse_sequence - gate_sequence)
    difference_matrices.append(absolute_differences)


for i, diff_matrix in enumerate(difference_matrices):
    if diff_matrix is not None:
        print(f"Absolute differences for sequence pair {i}:")
        print(diff_matrix)

        max_diff = np.max(diff_matrix)
        print(f"Maximum absolute difference for sequence pair {i}: {max_diff}")
        within_tolerance = np.all(diff_matrix <= tolerance)
        print(f"All differences within tolerance ({tolerance}): {within_tolerance}")
    else:
        print(f"Comparison skipped for sequence pair {i} due to shape mismatch.")



