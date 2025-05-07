import numpy as np

from qft_models.circuit_9 import Circuit9

from utils.helpers import random_parameter_set2
from utils.data_handler import save

# Function parameter
start = 0
stop = 20
points = 1000  # 1000
x = np.linspace(start, stop, points)

# Samples
num_samples = 5000

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
save("Circuit9_Random", num_qubits, 1, num_samples, start, stop, points, x, fx_set, "c9_exp/gate/")






