import numpy as np
import pennylane as qml
import random

from pulse.pulse_gates import *
from utils.definitions import GROUND_STATE, EXCITED_STATE, SUPERPOSITION_STATE_H, RANDOM_STATE, PHASESHIFTED_STATE
from utils.helpers import prints, statevector_similarity, statevector_fidelity
from helpers import *


# Pennylane Big Endian convention: |q0 q1 q2 q3 ... >
class PennyCircuit:

    def __init__(self, num_qubits):

        self.num_qubits = num_qubits

    def run_quick_circuit(self, wires, init_state=None):
        dev = qml.device('default.qubit', wires=self.num_qubits)

        @qml.qnode(dev)
        def general_circuit():
            if init_state is not None:
                qml.StatePrep(init_state, wires=range(self.num_qubits))

            qml.CZ(wires)
            # for qubit in target_q:
            #     if 0 <= qubit < self.num_qubits:
            #         qml.Hadamard(wires=qubit)
            #     else:
            #         print(f"Warning: Target qubit {qubit} is out of range (0 to {self.num_qubits - 1}).")

            return qml.state()

        # print(qml.draw(general_circuit)())
        return general_circuit()



# num_qubits = 3
# init = RANDOM_STATE(num_qubits)
# wires = [0, 1]
#
# prints(init)
#
# c = PennyCircuit(num_qubits)
# penny_state = c.run_quick_circuit(wires, init_state=init.data)
# prints(penny_state)
#
# _, _, current_state = cz(init, wires)
# result_state = current_state[-1]
# prints(result_state)
#
# sim = statevector_similarity(penny_state, result_state)
# fid = statevector_fidelity(penny_state, result_state)
# print(f"sim = {sim}, fid = {fid}")
# print(20 * "-", "\n")





# PARALLEL TESTS GENERATION
def generate_wire_pairs(num_tests, max_qubits=7):
    """
    Generates a list of test cases, where each test case contains the number of qubits
    and a list of unique wire pairs.

    Args:
        num_tests (int): The number of test cases to generate.
        max_qubits (int): The maximum number of qubits to consider (default is 7).

    Returns:
        list[tuple[int, list[list[int]]]]: A list of test cases. Each test case is a
                                           tuple containing the number of qubits and a
                                           list of unique wire pairs.
    """
    test_cases = []
    for _ in range(num_tests):
        num_qubits = np.random.randint(2, max_qubits + 1)  # Need at least 2 qubits for pairs
        max_possible_pairs = num_qubits * (num_qubits - 1) // 2
        num_pairs = np.random.randint(1, max_possible_pairs + 1)

        possible_pairs_indices = np.arange(max_possible_pairs)
        chosen_pair_indices = np.random.choice(possible_pairs_indices, size=num_pairs, replace=False)

        all_pairs = []
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                all_pairs.append([i, j])

        chosen_pairs = [all_pairs[index] for index in chosen_pair_indices]

        test_cases.append((num_qubits, chosen_pairs))
    return test_cases

def generate_tests(num_tests):
    test_cases = []
    for _ in range(num_tests):
        num_qubits = np.random.randint(1, 8)
        num_targets = np.random.randint(1, num_qubits+1)

        possible_targets = np.arange(num_qubits)
        target_qubits = np.random.choice(possible_targets, size=num_targets, replace=False)
        test_cases.append((num_qubits, target_qubits.tolist()))
    return test_cases

def test_gate(gate_name):

    test_cases = generate_tests(20)
    sequence_repetitions = 3

    for i, (num_qubits, target_qubits) in enumerate(test_cases):

        print(f"Test Case {i + 1}:")
        print(f"  Number of qubits (n): {num_qubits}")
        print(f"  Target qubits: {target_qubits} \n")

        c = PennyCircuit(num_qubits)
        for init_function in possible_init_states:
            init = init_function(num_qubits)

            penny_state = c.run_quick_circuit(target_q=target_qubits, init_state=init.data)
            for _ in range(sequence_repetitions):
                penny_state = c.run_quick_circuit(target_q=[0], init_state=penny_state)

            prints(penny_state)

            _, _, current_state = H_pulseSPEC(init, target_qubits, -np.pi / 2, True)

            for _ in range(sequence_repetitions):
                _, _, current_state = H_pulseSPEC(current_state[-1], [0], -np.pi / 2, True)

            # _, _, no_correction = H_pulseSPEC(init, target_qubits, -np.pi / 2, False)
            # prints(no_correction[-1])

            result_state = current_state[-1]
            prints(result_state)

            # manual_correction = global_phase_correction * current_state[-1].data
            # prints(manual_correction)
            # print("-")

            sim = statevector_similarity(penny_state, result_state)
            fid = statevector_fidelity(penny_state, result_state)
            print(f"sim = {sim}, fid = {fid}")
            print(20 * "-", "\n")


# def repeat_circuit(circuit_runner_func, initial_state, target_qubits, repetitions, *args, **kwargs):
#     current_state = initial_state
#     for _ in range(repetitions):
#         if callable(circuit_runner_func):
#             if circuit_runner_func.__name__ == 'run_quick_circuit':
#                 current_state = circuit_runner_func(target_q=target_qubits, init_state=current_state, *args, **kwargs)
#             elif circuit_runner_func.__name__ == 'H_pulseSPEC':
#                 _, _, current_state_tuple = circuit_runner_func(current_state[-1], target_qubits, *args, **kwargs)
#                 current_state = current_state_tuple[-1] # Assuming the last element is the state
#             else:
#                 raise ValueError(f"Unsupported circuit runner function: {circuit_runner_func.__name__}")
#         else:
#             raise TypeError("circuit_runner_func must be a callable function.")
#     return current_state








