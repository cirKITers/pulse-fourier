import qiskit_aer
from qiskit import QuantumCircuit

from src.utils.definitions import *
from src.pulse_gates import *

class QuickCircuit:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits

    def run_quick_circuit(self, initial_state):
        qc = QuantumCircuit(self.num_qubits)

        if initial_state is not None:
            qc.initialize(initial_state, range(self.num_qubits))  # Initialize with the custom state

        # qc.h(0)
        # qc.rx(np.pi / 2, 0)
        # qc.rx(np.pi / 2, 1)
        qc.cx(1, 0)
        simulator = 'statevector_simulator'
        backend = qiskit_aer.Aer.get_backend(simulator)
        result = backend.run(qc).result()
        statevector = result.get_statevector()
        return statevector


num_q = 2
qm = QuickCircuit(num_qubits=num_q)


phi_plus = qm.run_quick_circuit(PHI_PLUS_NO_CNOT)
psi_plus = qm.run_quick_circuit(PSI_PLUS_NO_CNOT)
phi_minus = qm.run_quick_circuit(PHI_MINUS_NO_CNOT)
psi_minus = qm.run_quick_circuit(PSI_MINUS_NO_CNOT)

pulse_control_qubit = 1
pulse_target_qubit = 0

omega_list = [5.0, 4.9]
g = 0.05
ds = 1.0728385125463975
cnot_dur = 239
cnot_p = 1.7554873088999543
cnot_sigma = 1.5

# (current_state, control_qubit, target_qubit, omega_list, g, drive_strength, cnot_duration=120, cnot_phase=0.0, cnot_sigma=15)
_, _, phi_plusEcho = CNOT_pulseEcho(PHI_PLUS_NO_CNOT, pulse_control_qubit, pulse_target_qubit, omega_list, g, ds, cnot_dur, cnot_p, cnot_sigma)
_, _, psi_plusEcho = CNOT_pulseEcho(PSI_PLUS_NO_CNOT, pulse_control_qubit, pulse_target_qubit, omega_list, g, ds, cnot_dur, cnot_p, cnot_sigma)
_, _, phi_minusEcho = CNOT_pulseEcho(PHI_MINUS_NO_CNOT, pulse_control_qubit, pulse_target_qubit, omega_list, g, ds, cnot_dur, cnot_p, cnot_sigma)
_, _, psi_minusEcho = CNOT_pulseEcho(PSI_MINUS_NO_CNOT, pulse_control_qubit, pulse_target_qubit, omega_list, g, ds, cnot_dur, cnot_p, cnot_sigma)


prints(phi_plus)
# prints(PHI_PLUS)
prints(phi_plusEcho)
# print(np.equal(phi_plus, phi_plusEcho))
print("\n")

prints(psi_plus)
# prints(PSI_PLUS)
prints(psi_plusEcho)
# print(np.equal(psi_plus, psi_plusEcho))
print("\n")

prints(phi_minus)
# prints(PHI_MINUS)
prints(phi_minusEcho)
# print(np.equal(phi_minus, phi_minusEcho))
print("\n")

prints(psi_minus)
# prints(PSI_MINUS)
prints(psi_minusEcho)
# print(np.equal(psi_minus, psi_minusEcho))
print("\n")





