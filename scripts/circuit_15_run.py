import numpy as np

from qft_models.circuit_15 import Circuit15

from utils.helpers import random_parameter_set2
from utils.data_handler import save

# Function parameter
# start = 0
# stop = 20
# points = 1000  # 1000
# x = np.linspace(start, stop, points)

excluding_discrete_points = 8  # len(x) is plus one (including interval length value)!
interval_length = 2 * np.pi
delta = interval_length / excluding_discrete_points
x = np.arange(0, interval_length + delta, delta)
start = 0
stop = interval_length
points = excluding_discrete_points + 1


# Samples
num_samples = 10  # 5000

# Hyper parameter
num_qubits = 4  # scale
num_ansatz = 2  # const, 1 layer

seeds = list(range(500, 501))

for seed in seeds:
    print(f"Starting seed number {seed}...")


    # Model
    model = Circuit15(num_qubits)

    # Parameter
    parameter_set = random_parameter_set2(num_samples, 2, num_qubits, len(["RY", "RY"]), seed=seed)

    # MODEL RUN
    fx_set = model.sample_fourier(x, parameter_set, num_samples)

    # Save function
    save("Circuit15_Random", num_qubits, 1, num_samples, start, stop, points, seed, x, fx_set, "c15_exp/gate/", cluster=False)






