import numpy as np
import pennylane as qml
import random

from pulse.pulse_system import *
from utils.definitions import GROUND_STATE, EXCITED_STATE, SUPERPOSITION_STATE_H, RANDOM_STATE, PHASESHIFTED_STATE
from utils.helpers import prints, overlap_components, statevector_fidelity
from helpers import *


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
        num_targets = np.random.randint(1, num_qubits + 1)

        possible_targets = np.arange(num_qubits)
        target_qubits = np.random.choice(possible_targets, size=num_targets, replace=False)
        test_cases.append((num_qubits, target_qubits.tolist()))
    return test_cases


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
