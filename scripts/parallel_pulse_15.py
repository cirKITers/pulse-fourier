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
num_samples = 1000

# Hyper parameter
num_qubits = 4  # scale
num_ansatz = 2  # const, 1 layer


def process_sample(params):
    return model.sample_fourier(x, np.expand_dims(params, axis=0), 1)[0]



n_jobs = 4  # Adjust based on your CPU core count
parameter_set = random_parameter_set2(num_samples, 2, num_qubits, len(["RY", "RY"]), seed=15)
model = Pulse15(num_qubits)

results = Parallel(n_jobs=n_jobs)(delayed(process_sample)(params) for params in parameter_set)
fx_set = np.array(results)

    # Save the complete result
    # save("Pulse15_Random_Parallel_Joblib", num_qubits, 1, num_samples, start, stop, points, x, fx_set, "c15_exp/pulse/", cluster=True)

