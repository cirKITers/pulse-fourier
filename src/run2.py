import numpy as np
from qiskit_aer import Aer

from src.gate_fourier_models import *
from src.utils.visualize.fx import *
from src.coefficients import *

from src.pulse_gates import *

# FIXED PARAMS
theta = np.pi / 2


# GATE LEVEL:
qm = TestCircuit
num_layer = qm.num_layer
num_qubits = qm.num_qubits
num_gates = qm.num_gates
simulator = 'statevector_simulator'


samples = 1

num_coeffs = 5
points = 1

x = np.linspace(1, 1, points)
parameter = fixed_parameter_set(num_layer, num_qubits, num_gates, samples, theta)


gate_qm = qm(parameter[0])
_, expected_statevector = predict_single(gate_qm, simulator, shots, None)

print(expected_statevector.data)


# PULSE LEVEL

init_state = GROUND_STATE(num_qubits)

_, _, trajectory_afterRX = RX_pulseMULT(theta, init_state, False, False)

print("num qubits:", num_qubits)


omega_list = [5.0, 4.9]  # Frequencies in GHz (control, target)
g = 0.02  # Coupling strength in GHz

tries = 500

gs = np.random.uniform(0.937, 0.939, tries)
drive_strengths = np.random.uniform(0.288, 0.29, tries)  # Needs calibration


for tryy in range(tries):
    print("###############################################################################")
    _, _, trajectory_afterCNOT = CNOT_pulse(trajectory_afterRX[-1], 0, 1, omega_list, gs[tryy], drive_strengths[tryy])
    # print(round_statevector(trajectory[-1]).data)
    fid = fidelity(trajectory_afterCNOT[-1], expected_statevector.data)
    if fid > 0.6:
        print("##################################################################################################")
        print("found!")
        print(f"fidelity: {fidelity(trajectory_afterCNOT[-1], expected_statevector.data)}, drive strength: {drive_strengths[tryy]:}, g: {gs[tryy]}", )
    else:
        print(f"fidelity: {fidelity(trajectory_afterCNOT[-1], expected_statevector.data)}, drive strength: {drive_strengths[tryy]:}, g: {gs[tryy]}", )
    print(trajectory_afterCNOT[-1])









