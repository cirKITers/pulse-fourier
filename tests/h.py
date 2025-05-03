import numpy as np
import pennylane as qml
import random

from pulse.pulse_system import PulseSystem
from tests.pipeline import generate_tests
from utils.definitions import GROUND_STATE, EXCITED_STATE, SUPERPOSITION_STATE_H, RANDOM_STATE, PHASESHIFTED_STATE
from utils.helpers import prints, overlap_components, statevector_fidelity
from helpers import *

class PennyCircuit:

    def __init__(self, num_qubits):

        self.num_qubits = num_qubits

    def run_quick_circuit(self, target_q, init_state=None):
        dev = qml.device('default.qubit', wires=self.num_qubits)

        @qml.qnode(dev)
        def general_circuit():
            if init_state is not None:
                qml.StatePrep(init_state, wires=range(self.num_qubits))

            for qubit in target_q:
                if 0 <= qubit < self.num_qubits:
                    qml.Hadamard(wires=qubit)
                else:
                    print(f"Warning: Target qubit {qubit} is out of range (0 to {self.num_qubits - 1}).")
            return qml.state()

        return general_circuit()


# GLOBAL PHASE ERRORS:
# len(target_qubits) mod 4 = 0: 1 (no error)
# len(target_qubits) mod 4 = 1: j
# len(target_qubits) mod 4 = 2: -1
# len(target_qubits) mod 4 = 3: -j

a = [-0.30722329+0.00000000j, 0.38777628+0.00000000j, -0.25139828+0.00000000j, -0.10535539+0.00000000j, 0.40421610+0.00000000j, -0.29910115+0.00000000j, 0.01741878+0.00000000j, -0.02537806+0.00000000j, -0.25574962+0.00000000j, -0.32134936+0.00000000j, 0.00120608+0.00000000j, -0.17928497+0.00000000j, -0.04647479+0.00000000j, -0.08830558+0.00000000j, 0.26809529+0.00000000j, -0.38012136+0.00000000j]

b = [0.06105852+0.00014391j, 0.03151961+0.00053626j, -0.39439265-0.00435227j, 0.50940695+0.00095206j, -0.05306745-0.00230890j, -0.12141158+0.00028316j, -0.00620323+0.00083608j, 0.09475259+0.00053844j, 0.07350615-0.00008159j, 0.09102138+0.00001935j, -0.28471323-0.00134931j, -0.34005901+0.00169489j, 0.12227507-0.00121028j, -0.11277453+0.00148208j, 0.43902033+0.00079908j, 0.35047266-0.00296480j]

print(statevector_fidelity(a, b))
print(overlap_components(a, b))


# PARALLEL TEST GENERATION, passed with fid ~ 0.99995, sim ~ 0.995
test_cases = generate_tests(20)
sequence_repetitions = 3

for i, (num_qubits, target_qubits) in enumerate(test_cases):

    num_qubits = 1
    target_qubits = [0]

    print(f"Test Case {i + 1}:")
    print(f"  Number of qubits (n): {num_qubits}")
    print(f"  Target qubits: {target_qubits} \n")


    c = PennyCircuit(num_qubits)
    for init_function in possible_init_states:
        init = init_function(num_qubits)

        penny_state = init.data
        for _ in range(sequence_repetitions):
            penny_state = c.run_quick_circuit(target_q=target_qubits, init_state=penny_state)

        prints(penny_state)

        pls = PulseSystem(num_qubits, init)

        for _ in range(sequence_repetitions):
            pls.h(target_qubits)

        result_state = pls.current_state
        prints(result_state)

        # manual_correction = global_phase_correction * current_state[-1].data
        # prints(manual_correction)
        # print("-")

        sim = overlap_components(penny_state, result_state)
        fid = statevector_fidelity(penny_state, result_state)
        print(f"sim = {sim}, fid = {fid}")
        print(20*"-", "\n")


# num_qubits = 2
# target_qubits = [0, 1]
# # global_phase_correction = -1j
#
# init = RANDOM_STATE(num_qubits)
#
# c = PennyCircuit(num_qubits)
# penny_state = c.run_quick_circuit(target_q=target_qubits, init_state=init.data)
# penny_state = c.run_quick_circuit(target_q=target_qubits, init_state=penny_state)
# penny_state = c.run_quick_circuit(target_q=target_qubits, init_state=penny_state)
# penny_state = c.run_quick_circuit(target_q=target_qubits, init_state=penny_state)
# prints(penny_state)
#
# _, _, current_state = H_pulseSPEC(init, target_qubits, -np.pi/2, True)
# _, _, current_state = H_pulseSPEC(current_state[-1], target_qubits, -np.pi/2, True)
# _, _, current_state = H_pulseSPEC(current_state[-1], target_qubits, -np.pi/2, True)
# _, _, current_state = H_pulseSPEC(current_state[-1], target_qubits, -np.pi/2, True)
#
# # _, _, no_correction = H_pulseSPEC(init, target_qubits, -np.pi / 2, False)
# # prints(no_correction[-1])
#
# result_state = current_state[-1]
# prints(result_state)
#
# # manual_correction = global_phase_correction * current_state[-1].data
# # prints(manual_correction)
# # print("-")
#
# sim = statevector_similarity(penny_state, result_state)
# fid = statevector_fidelity(penny_state, result_state)
# print(f"sim = {sim}, fid = {fid}")
# print(20*"-", "\n")
