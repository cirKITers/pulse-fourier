import numpy as np

from qft_models.pulse_9 import Pulse9

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
num_samples = 5000

# Hyper parameter
num_qubits = 4  # scale
num_ansatz = 2  # const, 1 layer


# Model
model = Pulse9(num_qubits)

# Parameter
parameter_set = random_parameter_set2(num_samples, 2, num_qubits, len(["RX"]), seed=9)


# MODEL RUN
fx_set = model.sample_fourier(x, parameter_set, num_samples)

# Save function
save("Pulse9_Random", num_qubits, 1, num_samples, start, stop, points, x, fx_set, "c9_exp/pulse/", cluster=True)


# file_to_load = "../results/c9_exp/Circuit9_Random_4qubits_1layers_2samples_0start_20stop_2points.json"
# loaded_x, loaded_fx_set = load(file_to_load)

# plot_nfx(x, fx_set, random_color=True)




