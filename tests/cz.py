import numpy as np
import pennylane as qml

from pulse.pulse_system import *
from tests.helpers import possible_init_states
from tests.pipeline import *
from utils.definitions import *
from utils.helpers import prints, statevector_fidelity


class PennyCircuitCZ:

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
                    qml.CZ(wires=wire_pair)
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

    c = PennyCircuitCZ(num_qubits)
    for init_function in possible_init_states:
        init = init_function(num_qubits)

        penny_state = c.run_quick_circuit(wire_pairs, init_state=init.data)
        prints(penny_state)

        pls = PulseSystem(num_qubits, init)


        for wire_pair in wire_pairs:
            pls.cz(wire_pair)

        result_state = pls.current_state
        prints(result_state)

        sim = overlap_components(penny_state, result_state)
        fid = statevector_fidelity(penny_state, result_state)
        print(f"sim = {sim}, fid = {fid}")
        print(20 * "-", "\n")


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
        similarity = overlap_components(expected_state, actual_state[-1])
        fidelity = statevector_fidelity(expected_state, actual_state[-1])
        if not np.isclose(fidelity, 1.0):
            print(f"Test failed for initial state: {initial_state}")
            print(f"Expected: {expected_state}")
            print(f"Actual: {actual_state[-1]}")
            return False
    print("All tests passed!")
    return True

# dsCZ = 0.11884149043553377


# test_cz_gate(lambda x: cz(x))
