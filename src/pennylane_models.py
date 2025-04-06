import numpy as np
import pennylane as qml

# import numpy as np
# from skopt import gp_minimize
# from skopt.space import Real
#
# from src.utils.definitions import *
from src.pulse_gates import RZ_pulseSPEC, RX_pulseSPEC
from src.utils.definitions import *
from src.utils.helpers import *

from qiskit.quantum_info import Statevector

class PennyCircuit:

    def __init__(self, num_qubits):

        self.num_qubits = num_qubits

    def run_quick_circuit(self, theta, init_state=None):
        dev = qml.device('default.qubit', wires=self.num_qubits)

        @qml.qnode(dev)
        def general_circuit():
            if init_state is not None:
                qml.StatePrep(init_state, wires=range(self.num_qubits))
            # qml.CNOT(wires=[0, 1])  # big endian
            qml.RX(np.pi/2, 0)
            qml.RX(np.pi/2, 1)
            qml.RZ(theta, 0)
            qml.RZ(theta, 1)
            return qml.state()

        return general_circuit()


num_q = 2
c = PennyCircuit(num_q)

# current_state = GROUND_STATE(num_q)
# _, _, current_state = RX_pulseSPEC(np.pi/2, current_state, "all")
# prints(current_state[-1])

# after all RX(pi/2)

tries = 1
# k = np.linspace(0.04166614976713962, 0.041666966010380106, tries)
k_best = 0.04208416833667334

k_best_negative_pi = 0.04166666094977508
best_vals = []
typ_theta = [-np.pi, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, np.pi]

for t in range(100):

    for i in range(tries):

        theta = random_theta()

        penny_state = c.run_quick_circuit(theta)

        _, _, current_state = RZ_pulseSPEC(theta, Statevector([0.50000000 + 0.00000000j, 0.00000000 - 0.50000000j, 0.00000000 - 0.50000000j, -0.50000000 + 0.00000000j]), "all", k_best_negative_pi)

        sim = statevector_similarity(current_state[-1], penny_state)
        if sim > 0.99:

            print(f"works for {theta}")
            print(sim)
            prints(penny_state)
            prints(current_state[-1])
            print("\n")

        else:
            print(f"############################################################################DID NOT work for {theta}")
            print(sim)
            prints(penny_state)
            prints(current_state[-1])
            print("\n")
            # print(k[i], sim)
            # val_pair = (theta, k[i])
            # best_vals.append(val_pair)

    # print(best_vals)

# [-np.pi, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, np.pi]

# 0:

# 1:
# 2:
# 3:
# 4:
# 5:
# 6:





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
