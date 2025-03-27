from src.gate_fourier_models import *
from src.utils.visualize.fx import *
from src.coefficients import *

simulator = 'statevector_simulator'

qm = QiskitFourier1
num_layer = qm.num_layer
num_qubits = qm.num_qubits
num_gates = qm.num_gates

fixed_theta = 2.1

samples = 1

num_coeffs = 5
points = 1000

x = np.linspace(-1, 5, points)


parameter = fixed_parameter_set(num_layer, num_qubits, num_gates, samples, fixed_theta)

gate_fx_set = []

for smpl in range(samples):
    gate_qm = qm(parameter[smpl])
    gate_fx_set.append(predict_interval(gate_qm, simulator, shots, x, False))

    plot_fx(x, gate_fx_set[0], "gate_fx")

_, _, gate_coeffs = coefficient_distribution_fft(samples, num_coeffs, x, gate_fx_set, qm.model_name, parameter, None, plot=False)


