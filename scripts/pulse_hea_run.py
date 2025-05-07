import numpy as np

from qft_models.pulse_hea import PulseHE

from utils.helpers import random_parameter_set2
from utils.data_handler import save


# Function parameter
# start = 0
# stop = 20
# points = 1000  # 1000
# x = np.linspace(start, stop, points)

num_discrete_points = 20
interval_length = 2 * np.pi
x = np.arange(0, interval_length + interval_length / num_discrete_points, interval_length / num_discrete_points)
start = 0
stop = interval_length
points = num_discrete_points + 1

# Samples
num_samples = 5000

# Hyper parameter
num_qubits = 4  # scale
num_ansatz = 2  # const, 1 layer


# Model
model = PulseHE(num_qubits)

# Parameter
parameter_set = random_parameter_set2(num_samples, 2, num_qubits, len(["RY", "RZ", "RY"]), seed=42)

# MODEL RUN
fx_set = model.sample_fourier(x, parameter_set, num_samples)

# Save function
save("PulseHE_Random", num_qubits, 1, num_samples, start, stop, points, x, fx_set, "hea_exp/pulse/")



