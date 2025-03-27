import numpy as np
from matplotlib import pyplot as plt

from coefficients import coefficient_distribution_fft
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


# GATE
data = load_rows_between(gate_file, 2, 76)

parameter_set = np.array(data["parameter"])
fx_set = np.array(data["fx"])

qm = GateONEQFourier(num_qubits, num_layer, parameter_set[0])   # max 5 coeffs needed

a, b, c = coefficient_distribution_fft(num_samples, num_coeff, x, fx_set, qm, parameter_set, save=False, plot=True)


# PULSE
data = load_rows_between(pulse_file,)

# correlation(np.array(data["parameter"]), np.array(data["coeffs_all"]))



