import sys
import numpy as np
from joblib import Parallel, delayed

from qft_models.pulse_15 import Pulse15


from utils.helpers import random_parameter_set2
from utils.data_handler import save


excluding_discrete_points = 8  # len(x) is plus one (including interval length value)!
interval_length = 2 * np.pi
delta = interval_length / excluding_discrete_points
x = np.arange(0, interval_length + delta, delta)
start = 0
stop = interval_length
points = excluding_discrete_points + 1

# Samples
num_samples = 10

# Hyper parameter
num_qubits = 4  # scale
num_ansatz = 2  # const, 1 layer


def process_sample(index, params):
    # print(f"Processing sample at index {index} with parameters: {params}")
    return model.sample_fourier(x, np.expand_dims(params, axis=0), 1)[0]


n_jobs = -1

# SEED AS ARG
if len(sys.argv) > 1:
    seed = int(sys.argv[1])
    print(f"Evaluating seed number {seed} on script...")
else:
    print("Error: The provided seed must be an integer.")
    sys.exit(1)


parameter_set = random_parameter_set2(num_samples, 2, num_qubits, len(["RY", "RY"]), seed=seed)
model = Pulse15(num_qubits)


delayed_funcs = [delayed(process_sample)(i, params) for i, params in enumerate(parameter_set)]
results = Parallel(n_jobs=n_jobs)(delayed_funcs)
fx_set = np.array(results)

save("Pulse15_Random_Parallel_Joblib", num_qubits, 1, num_samples, start, stop, points, seed, x, fx_set, "c15_exp/pulse/", cluster=True)


# CHECK DIFFERENCE
# from qft_models.circuit_15 import Circuit15
# model_gate = Circuit15(num_qubits)
# fx_set_gate = model_gate.sample_fourier(x, parameter_set, num_samples)
#
#
# fx_set_pulse = fx_set
# tolerance = 1e-3
#
# difference_matrices = []
# for i in range(len(fx_set_pulse)):
#     pulse_sequence = fx_set_pulse[i]
#     gate_sequence = fx_set_gate[i]
#
#     if pulse_sequence.shape != gate_sequence.shape:
#         print(f"Warning: Shapes of sequences at index {i} do not match.")
#         difference_matrices.append(None)  # Or handle differently
#         continue
#
#     absolute_differences = np.abs(pulse_sequence - gate_sequence)
#     difference_matrices.append(absolute_differences)
#
#
# for i, diff_matrix in enumerate(difference_matrices):
#     if diff_matrix is not None:
#         print(f"Absolute differences for sequence pair {i}:")
#         print(diff_matrix)
#
#         max_diff = np.max(diff_matrix)
#         print(f"Maximum absolute difference for sequence pair {i}: {max_diff}")
#         within_tolerance = np.all(diff_matrix <= tolerance)
#         print(f"All differences within tolerance ({tolerance}): {within_tolerance}")
#     else:
#         print(f"Comparison skipped for sequence pair {i} due to shape mismatch.")


