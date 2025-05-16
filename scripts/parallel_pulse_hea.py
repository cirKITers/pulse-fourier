import sys

import numpy as np
from joblib import Parallel, delayed


from qft_models.pulse_hea import PulseHE

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


parameter_set = random_parameter_set2(num_samples, 2, num_qubits, len(["RY", "RZ", "RY"]), seed=seed)
model = PulseHE(num_qubits)

delayed_funcs = [delayed(process_sample)(i, params) for i, params in enumerate(parameter_set)]
results = Parallel(n_jobs=n_jobs)(delayed_funcs)
fx_set = np.array(results)

save("PulseHEA_Random_Parallel_Joblib", num_qubits, 1, num_samples, start, stop, points, seed, x, fx_set, "hea_exp/pulse/", cluster=True)

