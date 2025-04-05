import pennylane as qml
import numpy as np

from src.utils.definitions import *
from src.pulse_gates import *


class Circuit:

    def __init__(self, num_qubits):

        self.num_qubits = num_qubits

    def run_quick_circuit(self, init_state=None):
        dev = qml.device('default.qubit', wires=self.num_qubits)

        @qml.qnode(dev)
        def general_circuit():
            if init_state is not None:
                qml.StatePrep(init_state, wires=range(self.num_qubits))
            # qml.CNOT(wires=[0, 1])  # big endian
            qml.RX(theta, 0)
            qml.RX(theta, 1)
            # qml.RZ(theta, 0)
            qml.RZ(theta, 1)
            return qml.state()

        return general_circuit()


theta = np.pi / 2
num_q = 2
c = Circuit(num_q)

penny_state = c.run_quick_circuit()
prints(penny_state)


current_state = GROUND_STATE(num_q)
_, _, current_state = RX_pulseSPEC(theta, current_state, "all")
# prints(current_state[-1])

_, _, current_state = RZ_pulseSPEC(theta, current_state[-1], 1)
prints(current_state[-1])

print("\n Similarity:")
print(statevector_similarity(penny_state, current_state[-1]))

# num_q = 2
# c = Circuit(num_q)
#
# PHI_PLUS_NO_CNOT = Statevector([1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0])        # H(0)
# PSI_PLUS_NO_CNOT = Statevector([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)])        # X(1) H(0)
# PHI_MINUS_NO_CNOT = Statevector([1 / np.sqrt(2), 0, -1 / np.sqrt(2), 0])      # H(0) Z(0)
# PSI_MINUS_NO_CNOT = Statevector([0, 1 / np.sqrt(2), 0, -1 / np.sqrt(2)])      # X(1) H(0)
#
# phi_plus = c.run_quick_circuit(PHI_PLUS_NO_CNOT.data)
# psi_plus = c.run_quick_circuit(PSI_PLUS_NO_CNOT.data)
# phi_minus = c.run_quick_circuit(PHI_MINUS_NO_CNOT.data)
# psi_minus = c.run_quick_circuit(PSI_MINUS_NO_CNOT.data)
#
#
# print(phi_plus)
# prints(PHI_PLUS)
# print(np.equal(phi_plus, PHI_PLUS.data))
# print("\n")
#
# print(psi_plus)
# prints(PSI_PLUS)
# print(np.equal(psi_plus, PSI_PLUS.data))
# print("\n")
#
# print(phi_minus)
# prints(PHI_MINUS)
# print(np.equal(phi_minus, PHI_MINUS.data))
# print("\n")
#
# print(psi_minus)
# prints(PSI_MINUS)
# print(np.equal(psi_minus, PSI_MINUS.data))
# print("\n")


# ∣Φ+⟩, H(0) CNOT(0,1)
# |Ψ⁺⟩, X(1) H(0) CNOT(0,1)
# ∣Φ-⟩, H(0) Z(0) CNOT(0,1)
# |Ψ-⟩, X(1) H(0) Z(0) CNOT(0,1)
class BellStateCircuits:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits

    def run_quick_circuit(self, bell_state_type="phi_plus"):
        dev = qml.device('default.qubit', wires=self.num_qubits)

        @qml.qnode(dev)
        def bell_circuits():
            if bell_state_type == "phi_plus":
                qml.Hadamard(0)
                # qml.CNOT(wires=[0, 1])
            elif bell_state_type == "psi_plus":
                qml.PauliX(1)
                qml.Hadamard(0)
                # qml.CNOT(wires=[0, 1])
            elif bell_state_type == "phi_minus":
                qml.Hadamard(0)
                qml.PauliZ(0)
                # qml.CNOT(wires=[0, 1])
            elif bell_state_type == "psi_minus":
                qml.PauliX(1)
                qml.Hadamard(0)
                qml.PauliZ(0)
                # qml.CNOT(wires=[0, 1])
            else:
                raise ValueError("Invalid bell_state_type. Choose from 'phi_plus', 'psi_plus', 'phi_minus', or 'psi_minus'.")

            return qml.state()

        return bell_circuits()


# num_q = 2
# bell_circuit = BellStateCircuits(num_q)
#
# phi_plus = bell_circuit.run_quick_circuit("phi_plus")
# print(phi_plus)
# # print(PHI_PLUS.data)
# # print(np.equal(phi_plus, PHI_PLUS.data))
# print("\n")
#
# psi_plus = bell_circuit.run_quick_circuit("psi_plus")
# print(psi_plus)
# # print(PSI_PLUS.data)
# # print(np.equal(psi_plus, PSI_PLUS.data))
# print("\n")
#
# phi_minus = bell_circuit.run_quick_circuit("phi_minus")
# print(phi_minus)
# # print(PHI_MINUS.data)
# # print(np.equal(phi_minus, PHI_MINUS.data))
# print("\n")
#
# psi_minus = bell_circuit.run_quick_circuit("psi_minus")
# print(psi_minus)
# # print(PSI_MINUS.data)
# # print(np.equal(psi_minus, PSI_MINUS.data))
# print("\n")
