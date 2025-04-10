import numpy as np

from fft.coefficients import coefficient_distribution_fft
from src.qft_models.gate_fourier_models import GateONEQFourier
from src.utils.load import *
from visuals.correlation import *

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



