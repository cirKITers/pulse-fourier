import numpy as np
import pennylane as qml

class PennyCircuit:

    def __init__(self, num_qubits):

        self.num_qubits = num_qubits

    def run_quick_circuit(self, init_state=None):
        dev = qml.device('default.qubit', wires=self.num_qubits)

        @qml.qnode(dev)
        def general_circuit():
            if init_state is not None:
                qml.StatePrep(init_state, wires=range(self.num_qubits))

            qml.Hadamard(1)
            # qml.CNOT(wires=[0, 1])  # big endian
            # qml.RX(np.pi/2, 0)
            # qml.RX(np.pi/2, 1)
            # qml.CZ([0, 1])
            # qml.Hadamard(1)
            return qml.state()

        return general_circuit()


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


# TESTING BellState Pennylane Circuit Class

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


# TESTING BellState circuit 2.0

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
