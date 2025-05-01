import numpy as np
from qiskit.quantum_info import Operator, Statevector, partial_trace, DensityMatrix
import jax.numpy as jnp


# Matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
MINUS = np.array([[0, 0], [1, 0]], dtype=complex)
ZERO = np.array([[0, 0], [0, 0]], dtype=complex)
COND_PI = np.array([[0, 0], [0, np.pi]], dtype=complex)
CNOT_MATRIX = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
CZ = np.diag([1, 1, 1, -1])

# Pauli Operators
SIGMA_X = Operator(X)
SIGMA_Y = Operator(Y)
SIGMA_Z = Operator(Z)
SIGMA_MINUS = Operator(MINUS)


# HAMILTONIAN
# like in: https://qiskit-community.github.io/qiskit-dynamics/tutorials/optimizing_pulse_sequence.html
# derived from:  2 * np.pi * vu * SIGMA_Z / 2
def static_hamiltonian(vu):
    return np.pi * vu * SIGMA_Z

def drive_hamiltonian(drive_strength):
    # original: 2 * np.pi * drive_strength * SIGMA_X / 2
    return np.pi * drive_strength * SIGMA_X

def drive_X_hamiltonian(drive_strength):
    # original: 2 * np.pi * drive_strength * SIGMA_X / 2
    return drive_strength * SIGMA_X

#2 * np.pi * drive_strength * SIGMA_Y / 2
def drive_Y_hamiltonian(drive_strength):
    return drive_strength  * SIGMA_Y


# ONE QUBIT STATES
GROUND_STATE_OneQ = Statevector([1.0, 0.0])  # |0>
EXCITED_STATE_OneQ = Statevector([0.0, 1.0])  # |1>
SUPERPOSITION_STATE_H_OneQ = Statevector([1 / np.sqrt(2), 1 / np.sqrt(2)])
SUPERPOSITION_STATE_RX_OneQ = Statevector([1 / np.sqrt(2), -1j / np.sqrt(2)])
PHASE_SHIFTED_STATE_OneQ = Statevector([1 / np.sqrt(2), 1j / np.sqrt(2)])
RANDOM_STATE_A_OneQ = Statevector([0.8, 0.6])

# MULTI QUBIT STATES

# Bell states / EPR pairs:
PHI_PLUS = Statevector([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])        # ∣Φ+⟩, H(0) CNOT(0,1)
PSI_PLUS = Statevector([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0])        # |Ψ⁺⟩, X(1) H(0) CNOT(0,1)
PHI_MINUS = Statevector([1 / np.sqrt(2), 0, 0, -1 / np.sqrt(2)])      # ∣Φ-⟩, H(0) Z(0) CNOT(0,1)
PSI_MINUS = Statevector([0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0])      # |Ψ-⟩, X(1) H(0) Z(0) CNOT(0,1)


# The above, before CNOT(0, 1)
PHI_PLUS_NO_CNOT = Statevector([1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0])        # H(0)
PSI_PLUS_NO_CNOT = Statevector([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)])        # X(1) H(0)
PHI_MINUS_NO_CNOT = Statevector([1 / np.sqrt(2), 0, -1 / np.sqrt(2), 0])      # H(0) Z(0)
PSI_MINUS_NO_CNOT = Statevector([0, 1 / np.sqrt(2), 0, -1 / np.sqrt(2)])      # X(1) H(0)


GHZ_STATE = Statevector([1 / np.sqrt(2), 0, 0, 0, 0, 0, 0, 1 / np.sqrt(2)])
PLUS_ZERO_STATE = Statevector([1 / np.sqrt(2) + 0.j, 0. + 0.j, 1 / np.sqrt(2) + 0.j, 0. + 0.j])


def GROUND_STATE(num_qubits):
    """Returns the ground state |00...0> for the given number of qubits."""
    return Statevector.from_int(0, 2 ** num_qubits)


def EXCITED_STATE(num_qubits):
    """Returns the fully excited state |11...1> for the given number of qubits."""
    return Statevector.from_int(2 ** num_qubits - 1, 2 ** num_qubits)


def SUPERPOSITION_STATE_H(num_qubits):
    """Returns an equal superposition state (|0> + |1>)/sqrt(2) for one qubit,
    or a tensor product of single-qubit superpositions for multiple qubits."""
    if num_qubits == 1:
        return Statevector([1 / np.sqrt(2), 1 / np.sqrt(2)])
    else:
        return Statevector.from_label('+' * num_qubits)


def PHASESHIFTED_STATE(num_qubits):
    """Returns a phase-shifted state (|0> + i|1>)/sqrt(2) for one qubit,
    or a tensor product of single-qubit phase-shifted states for multiple qubits."""
    if num_qubits == 1:
        return Statevector([1 / np.sqrt(2), 1j / np.sqrt(2)])
    else:
        state = np.array([1 / np.sqrt(2), 1j / np.sqrt(2)])
        result = state
        for _ in range(num_qubits - 1):
            result = np.kron(result, state)
        return Statevector(result)


def RANDOM_STATE(num_qubits):
    """Generates a normalized random statevector for the given number of qubits."""
    num_dims = 2 ** num_qubits
    random_vector = np.random.randn(num_dims) + 1j * np.random.randn(num_dims)
    norm = np.linalg.norm(random_vector)
    if norm == 0:
        return Statevector(np.zeros(num_dims))
    return Statevector(random_vector / norm)
