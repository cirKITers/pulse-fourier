import numpy as np
import jax.numpy as jnp
from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import expm
from scipy.integrate import quad

from pulse.pulse_gates import RY_pulseSPEC
from utils.helpers import statevector_similarity, statevector_fidelity


# All tests passed!

def ry_gate(theta):
    """Analytical RY gate matrix."""
    return np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                     [np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)


def tensor_product(states):
    """Calculates the tensor product of a list of states."""
    result = states[0]
    for state in states[1:]:
        result = np.kron(result, state)
    return result


def apply_ry_to_qubit(state, theta, qubit_index, num_qubits):
    """Applies RY gate to a specific qubit in a multi-qubit state."""
    ry_matrix = ry_gate(theta)
    identity = np.eye(2, dtype=complex)

    operators = [identity] * num_qubits
    operators[qubit_index] = ry_matrix

    combined_operator = operators[0]
    for op in operators[1:]:
        combined_operator = np.kron(combined_operator, op)

    return np.dot(combined_operator, state)


def generate_ry_test_cases():
    """Generates test cases for the RY gate, including multi-qubit states and target qubit."""

    test_cases = []

    # Single-qubit tests
    test_cases.append((np.array([1, 0], dtype=complex), np.pi / 2, np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex), 0))
    test_cases.append((np.array([0, 1], dtype=complex), np.pi / 2, np.array([-1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex), 0))
    test_cases.append((np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex), np.pi, np.array([-1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex), 0))
    test_cases.append((np.array([1 / np.sqrt(2), -1 / np.sqrt(2)], dtype=complex), np.pi, np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex), 0))
    test_cases.append((np.array([1, 0], dtype=complex), 0, np.array([1, 0], dtype=complex), 0))
    test_cases.append((np.array([0, 1], dtype=complex), 0, np.array([0, 1], dtype=complex), 0))
    test_cases.append((np.array([1, 0], dtype=complex), np.pi, np.array([0, 1], dtype=complex), 0))
    test_cases.append((np.array([0, 1], dtype=complex), np.pi, np.array([-1, 0], dtype=complex), 0))
    test_cases.append((np.array([1, 0], dtype=complex), np.pi / 4, np.array([np.cos(np.pi / 8), np.sin(np.pi / 8)], dtype=complex), 0))  # corrected line
    test_cases.append((np.array([1, 1] / np.sqrt(2), dtype=complex), np.pi / 3, np.dot(ry_gate(np.pi / 3), np.array([1, 1] / np.sqrt(2), dtype=complex)), 0))
    test_cases.append((np.array([1, -1] / np.sqrt(2), dtype=complex), np.pi / 6, np.dot(ry_gate(np.pi / 6), np.array([1, -1] / np.sqrt(2), dtype=complex)), 0))

    # Multi-qubit tests (RY on qubit 0)
    test_cases.append((tensor_product([np.array([1, 0], dtype=complex), np.array([1, 0], dtype=complex)]), np.pi / 2,
                       tensor_product([np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex), np.array([1, 0], dtype=complex)]), 0))
    test_cases.append((tensor_product([np.array([0, 1], dtype=complex), np.array([1, 0], dtype=complex)]), np.pi / 2,
                       tensor_product([np.array([-1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex), np.array([1, 0], dtype=complex)]), 0))
    test_cases.append((tensor_product([np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)]), np.pi / 2,
                       tensor_product([np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex), np.array([0, 1], dtype=complex)]), 0))
    test_cases.append((tensor_product([np.array([0, 1], dtype=complex), np.array([0, 1], dtype=complex)]), np.pi / 2,
                       tensor_product([np.array([-1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex), np.array([0, 1], dtype=complex)]), 0))
    test_cases.append((tensor_product([np.array([1, 0], dtype=complex), np.array([1, 0], dtype=complex)]), np.pi,
                       tensor_product([np.array([0, 1], dtype=complex), np.array([1, 0], dtype=complex)]), 0))
    test_cases.append((tensor_product([np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)]), np.pi,
                       tensor_product([np.array([0, 1], dtype=complex), np.array([0, 1], dtype=complex)]), 0))
    test_cases.append((tensor_product([np.array([0, 1], dtype=complex), np.array([1, 0], dtype=complex)]), np.pi,
                       tensor_product([np.array([-1, 0], dtype=complex), np.array([1, 0], dtype=complex)]), 0))
    test_cases.append((tensor_product([np.array([0, 1], dtype=complex), np.array([0, 1], dtype=complex)]), np.pi,
                       tensor_product([np.array([-1, 0], dtype=complex), np.array([0, 1], dtype=complex)]), 0))
    test_cases.append((tensor_product([np.array([1, 0], dtype=complex), np.array([1, 0], dtype=complex)]), -np.pi,
                       tensor_product([np.array([0, -1], dtype=complex), np.array([1, 0], dtype=complex)]), 0))  # negative theta.


    # Multi-qubit tests (RY on qubit 1)
    test_cases.append((tensor_product([np.array([1, 0], dtype=complex), np.array([1, 0], dtype=complex)]), np.pi / 2,
                       tensor_product([np.array([1, 0], dtype=complex), np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)]), 1))
    test_cases.append((tensor_product([np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)]), np.pi / 2,
                       tensor_product([np.array([1, 0], dtype=complex), np.array([-1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)]), 1))
    test_cases.append((tensor_product([np.array([1, 0], dtype=complex), np.array([1, 0], dtype=complex)]), -np.pi / 2,
                       tensor_product([np.array([1, 0], dtype=complex), np.array([1 / np.sqrt(2), -1 / np.sqrt(2)], dtype=complex)]), 1))

    initial_states = [case[0] for case in test_cases]
    expected_states = [case[2] for case in test_cases]
    thetas = [case[1] for case in test_cases]
    target_qubits = [case[3] for case in test_cases]

    return initial_states, expected_states, thetas, target_qubits


# def fidelity(actual, expected):
#     """Calculates the fidelity between two quantum states."""
#     if actual.ndim == 1 and expected.ndim == 1:
#         return abs(np.vdot(actual, expected)) ** 2
#     elif actual.ndim == 2 and expected.ndim == 2:
#         return np.trace(np.dot(np.sqrt(np.dot(actual, actual.conj().T)), expected)) ** 2 / np.trace(actual) / np.trace(expected)
#     else:
#         raise ValueError("Input states must be either vectors or density matrices.")


def test_ry_pulse_implementation():
    """Tests the RY gate pulse implementation against the expected states."""
    initial_states, expected_states, thetas, target_qubits = generate_ry_test_cases()
    ds = 0.001  # Example value. Adjust as needed.

    for i in range(len(initial_states)):
        initial_state_op = Statevector(initial_states[i])
        _, _, result_state = RY_pulseSPEC(thetas[i], initial_state_op, ds, target_qubits=target_qubits[i])

        sim = statevector_similarity(result_state[-1], expected_states[i])
        f = statevector_fidelity(result_state[-1], expected_states[i])
        print(f"Test case {i + 1}, Similarity: {f}")
        assert f > 0.95, f"RY gate similarity is low for test case {i + 1}: {f}"

    print("RY pulse implementation tests passed!")


test_ry_pulse_implementation()

# old test
# num_q = 2
# c = PennyCircuit(num_q)
#
# penny_state = c.run_quick_circuit([0, 1, 0, 0])
# prints(penny_state)
# ds = np.linspace(0.04278048312942711, 0.04278081292467579, 1)
# # ds = np.linspace(0.03, 0.05)
#
# for i in range(len(ds)):
#     _, _, states = RY_pulseSPEC(-np.pi/4, Statevector([0, 1, 0, 0]), ds[i], 1)
#
#     sim = statevector_similarity(penny_state, states[-1])
#     print(sim, ds[i])
#     if sim > 0.999:
#         prints(states[-1])

