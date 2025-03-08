import numpy as np
from matplotlib import pyplot as plt

from gate_coefficients import coefficient_distribution
from gate_fourier_models import GateONEQFourier
from utils.helpers import random_parameter

# Current TASKS:
# Wiedmann Fourier Trees
# Kedro
# Correlation params and coefficients numerically
# -> Linear transformation?
# Pulse - level simple QFM


# Hyper
QuantumModel = GateONEQFourier   # max 5 coeffs needed
simulator = 'statevector_simulator'
num_layer = 3
num_qubits = 1
shots = 32768

interval = [0, 1]
points = 200

# Analysis
num_samples = 75
num_coeff = 5


a, b, c = coefficient_distribution(num_samples, num_coeff, QuantumModel, num_layer, num_qubits, simulator, shots, interval, points)

