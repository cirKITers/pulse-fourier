import numpy as np

from gate_fourier_models import GateONEQFourier
from pulse_fourier_models import PulseONEQFourier
from utils.helpers import *
from data.load import *
from src.utils.visualize.fx import *
from coefficients import *


simulator = 'statevector_simulator'
num_repetitions = 1
num_layer = 4
num_qubits = 1
num_coeffs = 5

samples = 2
shots = 32768
points = 70

x = np.linspace(0, 1, points)

parameter = random_parameter_set(num_repetitions, num_layer, num_qubits, samples)

gate_fx_set = []
pulse_fx_set = []

for _ in range(samples):
    gate_qm = GateONEQFourier(num_qubits, num_layer, np.array(parameter[_]))
    pulse_qm = PulseONEQFourier(num_qubits, num_layer, np.array(parameter[_]))

    gate_fx_set.append(gate_qm.predict_interval(simulator, shots, x, False))
    pulse_fx_set.append(pulse_qm.predict_interval(x, plot=False))
    plot_2fx_advanced(x, pulse_fx_set[_], gate_fx_set[_])

    _, _, pulse_coeffs = coefficient_distribution_fft(samples, num_coeffs, x, pulse_fx_set[_], pulse_qm.model_name, parameter[_], pulse_file, plot=False)
    _, _, gate_coeffs = coefficient_distribution_fft(samples, num_coeffs, x, gate_fx_set[_], gate_qm.model_name, parameter[_], gate_file, plot=False)

