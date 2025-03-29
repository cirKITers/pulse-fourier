from qiskit_aer import Aer

from src.gate_fourier_models import *
from src.utils.visualize.fx import *
from src.coefficients import *

from src.pulse_gates import *

# PULSE LEVEL

theta = np.pi / 2
num_qubits = 5
init_state = GROUND_STATE(num_qubits)

_, _, trajectory = RX_pulseMULT(theta, init_state, False, False)

print("num qubits:", num_qubits)
# print(trajectory[-1])

_, _, trajectory = RX_pulseMULT(theta, trajectory[-1], False, False)
print(trajectory[-1])

# GATE LEVEL:
qm = TestCircuit
num_layer = qm.num_layer
num_qubits = qm.num_qubits
num_gates = qm.num_gates
simulator = 'statevector_simulator'

fixed_theta = np.pi / 2

samples = 1

num_coeffs = 5
points = 1

x = np.linspace(1, 1, points)

parameter = fixed_parameter_set(num_layer, num_qubits, num_gates, samples, fixed_theta)

gate_fx_set = []

for smpl in range(samples):
    gate_qm = qm(parameter[smpl])
    gate_fx_set.append(predict_interval(gate_qm, simulator, shots, x, False))

    # plot_fx(x, gate_fx_set[0], "gate_fx")

# _, _, gate_coeffs = coefficient_distribution_fft(samples, num_coeffs, x, gate_fx_set, qm.model_name, parameter, None, plot=False)

