import numpy as np
import pennylane as qml

from pulse.pulse_gates import *
from tests.helpers import possible_init_states
from tests.pipeline import generate_wire_pairs
from utils.definitions import *
from utils.helpers import prints

# PASSED ALL BELLSTATE TESTS AS WELL

class PennyCircuitCNOT:

    def __init__(self, num_qubits):

        self.num_qubits = num_qubits

    def run_quick_circuit(self, wire_pairs, init_state=None):
        dev = qml.device('default.qubit', wires=self.num_qubits)

        @qml.qnode(dev)
        def general_circuit():
            if init_state is not None:
                qml.StatePrep(init_state, wires=range(self.num_qubits))

            for wire_pair in wire_pairs:
                if 0 <= wire_pair[0] < self.num_qubits:
                    qml.CNOT(wires=wire_pair)
                else:
                    print(f"Warning: Target qubit {wire_pair[0]} is out of range (0 to {self.num_qubits - 1}).")

            return qml.state()

        print(qml.draw(general_circuit)())
        return general_circuit()


test_cases = generate_wire_pairs(20, 7)

for i, (num_qubits, wire_pairs) in enumerate(test_cases):

    print(f"Test Case {i + 1}:")
    print(f"  Number of qubits (n): {num_qubits}")
    print(f"  Wire pairs: {wire_pairs} \n")

    c = PennyCircuitCNOT(num_qubits)
    for init_function in possible_init_states:
        init = init_function(num_qubits)

        penny_state = c.run_quick_circuit(wire_pairs, init_state=init.data)
        prints(penny_state)

        current_state = [0]
        current_state[-1] = init
        for wire_pair in wire_pairs:
            _, _, current_state = cnot(current_state[-1], wire_pair)

        result_state = current_state[-1]
        prints(result_state)

        sim = statevector_similarity(penny_state, result_state)
        fid = statevector_fidelity(penny_state, result_state)
        print(f"sim = {sim}, fid = {fid}")
        print(20 * "-", "\n")


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

