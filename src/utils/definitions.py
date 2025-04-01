from src.utils.helpers import *


# ENTANGLEMENT

# TODO TEST IT MORE
def quantify_entanglement(state):
    """
    Calculates the entanglement between each pair of qubits in a multi-qubit statevector.

    Returns:
        dict: A dictionary where keys are qubit pairs (tuples) and values are the entanglement measures. 0 min, max 1.
    """

    num_qubits = state.num_qubits
    entanglement_measures = {}

    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):  # Avoid redundant pairs
            # Calculate the reduced density matrix for qubit i
            remaining_qubits_i = [q for q in range(num_qubits) if q != i]
            reduced_density_matrix_i = partial_trace(state, remaining_qubits_i).data

            # Calculate the von Neumann entropy for qubit i
            eigenvalues_i = np.linalg.eigvalsh(reduced_density_matrix_i)
            eigenvalues_i = eigenvalues_i[eigenvalues_i > 1e-15]
            entropy_i = -np.sum(eigenvalues_i * np.log2(eigenvalues_i))

            entanglement_measures[(i, j)] = entropy_i

    all_zero = all(np.isclose(value, 0) for value in entanglement_measures.values())

    if all_zero:
        return "No qubits are entangled"
    else:
        return entanglement_measures


# OPERATOR CONSTRUCTION

def coupling_hamiltonian(total_qubits, qubit_k, qubit_m, g):
    """Coupling term between two qubits: g/2 * (X_k X_m + Y_k Y_m)."""
    op_X_k = operator_on_qubit(SIGMA_X, qubit_k, total_qubits)
    op_X_m = operator_on_qubit(SIGMA_X, qubit_m, total_qubits)
    op_Y_k = operator_on_qubit(SIGMA_Y, qubit_k, total_qubits)
    op_Y_m = operator_on_qubit(SIGMA_Y, qubit_m, total_qubits)
    return (g / 2) * (op_X_k @ op_X_m + op_Y_k @ op_Y_m)

def operator_on_qubit(operator, qubit_index, num_qubits):
    """Places an operator on the specified qubit with identities on others."""
    ops = [I] * num_qubits
    ops[qubit_index] = operator
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

def sum_operator(operator, num_qubits):
    """Creates a sum of the operator applied to each qubit individually."""
    total = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    for k in range(num_qubits):
        total += operator_on_qubit(operator, k, num_qubits)
    return total

def tensor_product_identity(operator, num_qubits):
    """Creates a tensor product of operator with identities for num_qubits."""
    result = operator
    for _ in range(num_qubits - 1):
        result = np.kron(result, np.eye(2))
    return result


def tensor_product_2(operator, num_qubits):
    """Creates a tensor product of operator with identities for num_qubits."""
    result = operator
    for _ in range(num_qubits - 1):
        result = np.kron(result, operator)
    return result


# Matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
CNOT_MATRIX = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

# Pauli Operators
SIGMA_X = Operator(X)
SIGMA_Y = Operator(Y)
SIGMA_Z = Operator(Z)

# Two-qubit Matrices
I0 = kron(I, I)  # Identity on two-qubit space
Z0 = kron(Z, I)  # Z on qubit 0
Z1 = kron(I, Z)  # Z on qubit 1
X0 = kron(X, I)  # X on qubit 0
X1 = kron(I, X)  # X on qubit 1
P1 = 0.5 * (I0 - Z0)

# HAMILTONIAN
# like in: https://qiskit-community.github.io/qiskit-dynamics/tutorials/optimizing_pulse_sequence.html
def static_hamiltonian(omega):
    return 2 * np.pi * omega * SIGMA_Z / 2

def drive_hamiltonian(drive_strength):
    if isinstance(drive_strength, np.ndarray):
        drive_strength = drive_strength[0]
    return 2 * np.pi * drive_strength * SIGMA_X / 2


# ONE QUBIT STATES
GROUND_STATE_OneQ = Statevector([1.0, 0.0])
EXCITED_STATE_OneQ = Statevector([0.0, 1.0])
SUPERPOSITION_STATE_OneQ = Statevector([1 / np.sqrt(2), 1 / np.sqrt(2)])
PHASE_SHIFTED_STATE_OneQ = Statevector([1 / np.sqrt(2), 1j / np.sqrt(2)])
RANDOM_STATE_A_OneQ = Statevector([0.8, 0.6])


# MULTI QUBIT STATES
BELL_STATE = Statevector([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
GHZ_STATE = Statevector([1/np.sqrt(2), 0, 0, 0, 0, 0, 0, 1/np.sqrt(2)])

def GROUND_STATE(num_qubits):
    """Returns the ground state |00...0> for the given number of qubits."""
    return Statevector.from_int(0, 2**num_qubits)

def EXCITED_STATE(num_qubits):
    """Returns the fully excited state |11...1> for the given number of qubits."""
    return Statevector.from_int(2**num_qubits - 1, 2**num_qubits)

def SUPERPOSITION_STATE(num_qubits):
    """Returns an equal superposition state (|0> + |1>)/sqrt(2) for one qubit,
    or a tensor product of single-qubit superpositions for multiple qubits."""
    if num_qubits == 1:
        return Statevector([1/np.sqrt(2), 1/np.sqrt(2)])
    else:
        return Statevector.from_label('+' * num_qubits) / (2**(num_qubits/2))

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
    num_dims = 2**num_qubits
    random_vector = np.random.randn(num_dims) + 1j * np.random.randn(num_dims)
    norm = np.linalg.norm(random_vector)
    if norm == 0:
        return Statevector(np.zeros(num_dims))
    return Statevector(random_vector / norm)



