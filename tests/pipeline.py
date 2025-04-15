import numpy as np
import pennylane as qml
import random

from pulse.pulse_gates import H_pulseSPEC, H_pulseSPEC2, H_pulseSPEC3
from utils.definitions import GROUND_STATE, EXCITED_STATE, SUPERPOSITION_STATE_H, RANDOM_STATE, PHASESHIFTED_STATE
from utils.helpers import prints, statevector_similarity, statevector_fidelity
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


# num_q = 6
# target_qubits = [0, 1, 2, 5]
global_phase_correction = -1j

# PARALLEL TESTS GENERATION
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
    num_tests = 20
    test_cases = generate_tests(num_tests)

    print(f"Testing gate: {gate_name}")

    for i, (num_qubits, target_qubits) in enumerate(test_cases):
        print(f"Test Case {i + 1}:")
        print(f"  Number of qubits (n): {num_qubits}")
        print(f"  Target qubits: {target_qubits}")

        c = PennyCircuit(num_qubits)

        for init_function in possible_init_states:
            init = init_function(num_qubits)

            penny_state = c.run_quick_circuit(target_q=target_qubits, init_state=init.data)
            prints(penny_state)

            _, _, current_state = H_pulseSPEC(init, target_qubits, -np.pi / 2, True)

            result_state = current_state[-1]
            prints(result_state)

            sim = statevector_similarity(penny_state, result_state)
            fid = statevector_fidelity(penny_state, result_state)
            print(f"sim = {sim}, fid = {fid}")
            print(20 * "-", "\n")

        print("-" * 20)


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








