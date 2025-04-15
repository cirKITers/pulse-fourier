import numpy as np
import pennylane as qml

from pulse.pulse_gates import *
from utils.definitions import *
from utils.helpers import prints


class PennyCircuit:

    def __init__(self, num_qubits):

        self.num_qubits = num_qubits

    def run_quick_circuit(self, wires, init_state=None):
        dev = qml.device('default.qubit', wires=self.num_qubits)

        @qml.qnode(dev)
        def general_circuit():
            if init_state is not None:
                qml.StatePrep(init_state, wires=range(self.num_qubits))

            # qml.Hadamard(1)
            # qml.CZ([0, 1])
            # qml.Hadamard(1)
            qml.CNOT(wires=wires)  # big endian

            return qml.state()

        return general_circuit()


num_q = 2
init = PHI_PLUS_NO_CNOT

# PASSED TEST FOR ALL BELL STATES
BELL_STATE_NO_CNOT = PSI_PLUS_NO_CNOT
wires = [0, 1]

c = PennyCircuit(num_q)
penny_state = c.run_quick_circuit(wires, init.data)
prints(penny_state)


print("-"*20)
# prints(PSI_MINUS)


_, _, states = cnot(init, wires)

final_state = states[-1]

prints(final_state)
sim = statevector_similarity(penny_state, final_state)
fid = statevector_fidelity(penny_state, final_state)
print(f"sim = {sim}, fid = {fid}")


# ps = typical_phases()
#
# for i in range(len(ps)):
#     _, _, states = cnot(PHI_PLUS_NO_CNOT, ps[i])
#
#     prints(states[-1])
#     sim = statevector_similarity(penny_state, states[-1])
#
#     if sim > 0.95:
#         print("yeah")


# num_q = 2
# c = PennyCircuit(num_q)
#
# penny_state = c.run_quick_circuit(PHI_PLUS_NO_CNOT.data)
# prints(penny_state)
#
#
# print("-"*20)
# prints(PHI_PLUS)
#
# print("-"*20)
#
# phases = typical_phases()
#
# found = []
#
# for ryp in phases:
#     for rxp in phases:
#
#         print("y", ryp)
#         print("x", rxp)
#
#         _, _, state = cnot2(PHI_PLUS_NO_CNOT, ryp, rxp)
#         prints(state[-1])
#
#         sim = statevector_similarity(penny_state, state[-1])
#
#         print(sim)
#
#         if sim > 0.95:
#             found.append((ryp, rxp, sim))
#
#         print("-" * 20)
#
#
# print(found)

