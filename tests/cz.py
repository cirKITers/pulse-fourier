import numpy as np

from pulse.pulse_gates import *
from qft_models.pennylane_models import PennyCircuit
from utils.definitions import *
from utils.helpers import prints, statevector_fidelity


# All tests passed!

def apply_cz_manually(state_vector):
    """Applies the CZ gate to a 2-qubit state vector."""
    if len(state_vector) != 4:
        raise ValueError("State vector must represent 2 qubits.")

    if np.allclose(state_vector, np.array([1, 0, 0, 0])):
        return np.array([1, 0, 0, 0])
    if np.allclose(state_vector, np.array([0, 1, 0, 0])):
        return np.array([0, 1, 0, 0])
    if np.allclose(state_vector, np.array([0, 0, 1, 0])):
        return np.array([0, 0, 1, 0])

    new_state = state_vector.copy()
    new_state[3] *= -1  # Apply phase flip to |11>
    return new_state

def test_cz_gate(cz_function):
    """Tests the CZ gate with various state vectors."""

    test_cases = [
        np.array([1, 0, 0, 0]),  # |00>
        np.array([0, 1, 0, 0]),  # |01>
        np.array([0, 0, 1, 0]),  # |10>
        np.array([0, 0, 0, 1]),  # |11>
        np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]),  # (|00> + |11>)/sqrt(2)
        np.array([1 / np.sqrt(2), 0, 0, -1 / np.sqrt(2)]),  # (|00> - |11>)/sqrt(2)
        np.array([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)]),  # (|01> + |11>)/sqrt(2)
        np.array([0, 1 / np.sqrt(2), 0, -1 / np.sqrt(2)]),  # (|01> - |11>)/sqrt(2)
        np.array([0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)]),  # (|10> + |11>)/sqrt(2)
        np.array([0, 0, 1 / np.sqrt(2), -1 / np.sqrt(2)]),  # (|10> - |11>)/sqrt(2)
        np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]),  # Bell State.
        np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0]),  # Bell State.
    ]

    for initial_state in test_cases:
        expected_state = apply_cz_manually(initial_state)
        _, _, actual_state = cz_function(initial_state)
        similarity = statevector_similarity(expected_state, actual_state[-1])
        fidelity = statevector_fidelity(expected_state, actual_state[-1])
        if not np.isclose(fidelity, 1.0):
            print(f"Test failed for initial state: {initial_state}")
            print(f"Expected: {expected_state}")
            print(f"Actual: {actual_state[-1]}")
            return False
    print("All tests passed!")
    return True


# dsCZ = 0.11884149043553377


test_cz_gate(lambda x: cz(x))

