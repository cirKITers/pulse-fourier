import numpy as np
from matplotlib import pyplot as plt

from gate_coefficients import coefficient_distribution
from gate_fourier_models import GateONEQFourier
from utils.helpers import *
from utils.visualize.fx import *
from data.load import *
from constants import *
from correlation import *

# Current TASKS:
# Wiedmann Fourier Trees
# Kedro
# Correlation params and coefficients numerically
# -> Linear transformation?
# Pulse - level simple QFM


# Hyper
qm = GateONEQFourier   # max 5 coeffs needed
simulator = 'statevector_simulator'
num_repetitions = 1
num_layer = 3
num_qubits = 1
shots = 32768

points = 200
x = np.linspace(0, 1, points)

# Analysis
num_samples = 75
num_coeff = 5

# parameter_set = random_parameter_set(num_repetitions, num_layer, num_qubits, num_samples)

# a, b, c = coefficient_distribution(num_samples, num_coeff, qm, parameter_set, num_layer, num_qubits, simulator, shots, x, save=True, plot=True)

data = load_rows_between(gate_file, 2, 76)

plot_interactive_3d_heatmaps(np.array(data["parameter"]), np.array(data["coeffs_all"]))



