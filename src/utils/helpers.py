import numpy as np
from matplotlib import pyplot as plt
from qiskit.quantum_info import Operator, Statevector, partial_trace, DensityMatrix
import jax.numpy as jnp
from scipy.linalg import svd

def is_hermitian(matrix):
  """
  Verifies if a given square matrix is Hermitian.

  A matrix A is Hermitian if its conjugate transpose is equal to itself: A^H = A.
  The conjugate transpose (or Hermitian transpose) is obtained by taking the
  transpose of the matrix and then taking the complex conjugate of each entry.

  Args:
    matrix: A square NumPy array representing the matrix to be checked.

  Returns:
    True if the matrix is Hermitian, False otherwise.
  """
  if not isinstance(matrix, np.ndarray):
    raise TypeError("Input must be a NumPy array.")
  if matrix.ndim != 2:
    raise ValueError("Input must be a 2-dimensional array (a matrix).")
  rows, cols = matrix.shape
  if rows != cols:
    raise ValueError("Input matrix must be square.")

  conjugate_transpose = np.conjugate(matrix.T)
  return np.allclose(matrix, conjugate_transpose)


# GENERAL
def binary_c_t(n, c, t):
    if not (0 <= c < n and 0 <= t < n and c != t):
        return []

    indices = []
    for i in range(2 ** n):
        binary_repr = bin(i)[2:].zfill(n)
        # Big Endian
        if binary_repr[c] == '1' and binary_repr[t] == '1':
            indices.append(i)
    return indices


def prints(statevector):
    """Prints the values of a Qiskit Statevector in one line, without brackets or string quotes."""

    def format_complex(z):
        return f"{z.real:.8f}{'+' if z.imag >= 0 else '-'}{abs(z.imag):.8f}j"

    if isinstance(statevector, Statevector):
        formatted_array = [format_complex(x) for x in statevector.data]
        print("[" + ', '.join(formatted_array) + "]")
    elif isinstance(statevector, list) or isinstance(statevector, np.ndarray):
        formatted_array = [format_complex(x) for x in np.array(statevector)]
        print("[" + ', '.join(formatted_array) + "]")
    else:
        print("Input is not a Qiskit Statevector, list, or numpy array.")


def density_matrix(statevector):
    """
    Computes the density matrix of a given statevector.

    Args:
        statevector (numpy.ndarray): The statevector as a numpy array.

    Returns:
        numpy.ndarray: The density matrix.
    """
    statevector = np.array(statevector).reshape(-1, 1)  # Ensure it's a column vector
    density_matrx = np.dot(statevector, statevector.conj().T)
    return density_matrx


def bool_pure_state(statevector):
    dm = density_matrix(statevector)
    trace_squared = np.trace(np.dot(dm, dm))
    return np.isclose(trace_squared, 1.0)


def bool_valid_state(statevector):
    return statevector.is_valid()


def round_statevector(statevector, tolerance=1e-3):
    """
    Rounds the real and imaginary parts of a Statevector's elements.

    Args:
        statevector (Statevector): The input Statevector.
        tolerance (float): Values with absolute magnitude less than this are rounded to 0.

    Returns:
        Statevector: The rounded Statevector.
    """
    rounded_data = []
    for complex_val in statevector.data:
        real_part = complex_val.real
        imag_part = complex_val.imag

        if abs(real_part) < tolerance:
            real_part = 0.0
        if abs(imag_part) < tolerance:
            imag_part = 0.0

        rounded_data.append(complex(real_part, imag_part))

    return Statevector(rounded_data)


def probability_similarity(statevector1, statevector2, tolerance=1e-6):
    """Calculates the similarity of probability distributions."""
    probs1 = prob(statevector1)
    probs2 = prob(statevector2)

    if len(probs1) != len(probs2):
        raise ValueError("state vectors must have the same dimension")

    close_values = np.isclose(probs1, probs2, atol=tolerance)
    return np.sum(close_values) / len(probs1)


def overlap_components(target_state, actual_state, tolerance=1e-6, scaling="linear"):
    """
    Calculates the component-wise similarity between two quantum statevectors,
    sensitive to individual complex component values.
    quadratic: allow small deviations
    exponential: highly sensitive ti minor deviations

    Args:
        target_state (numpy.ndarray): The target statevector.
        actual_state (numpy.ndarray): The actual statevector.
        tolerance (float): The maximum allowed difference for a component to be considered "similar".
        scaling (str): The scaling function to use ("quadratic" or "exponential").

    Returns:
        float: A similarity score between 0 and 1.
    """
    if len(target_state) != len(actual_state):
        raise ValueError("Statevectors must have the same dimension.")

    num_components = len(target_state)
    similarity_sum = 0.0

    for i in range(num_components):
        real_diff = abs(target_state[i].real - actual_state[i].real)
        imag_diff = abs(target_state[i].imag - actual_state[i].imag)
        component_diff = real_diff + imag_diff

        if component_diff <= tolerance:
            similarity_sum += 1.0
        else:
            if scaling == "linear":
                similarity_sum += max(0.0, 1.0 - component_diff)
            elif scaling == "quadratic":
                similarity_sum += max(0.0, 1.0 - component_diff ** 2)
            elif scaling == "exponential":
                similarity_sum += np.exp(-component_diff)
            else:
                raise ValueError("Invalid scaling function. Choose 'quadratic' or 'exponential'.")

    return similarity_sum / num_components


def bool_statevector_closeness(state1, state2, atol=1e-2, rtol=1e-2):
    """
    Compares two quantum states (Statevector or DensityMatrix) with a custom tolerance.

    Args:
        state1: The first quantum state.
        state2: The second quantum state.
        atol: Absolute tolerance.
        rtol: Relative tolerance.

    Returns:
        True if the states are close, False otherwise.
    """
    if state1.dim != state2.dim:
        return False  # Different dimensions

    if isinstance(state1, Statevector):
        return np.allclose(state1.data, state2.data, atol=atol, rtol=rtol)
    elif isinstance(state1, DensityMatrix):
        return np.allclose(state1.data, state2.data, atol=atol, rtol=rtol)
    else:
        raise ValueError("Input states must be Statevector or DensityMatrix.")


# F(|ψ⟩,|ϕ⟩)=|⟨ψ∣ϕ⟩|^2, insensitive to global phase differences. Measures the closeness of overall probability distribution
# fidelity 0, means orthogonal
def statevector_fidelity(target_state, actual_state):
    """
    Calculates the fidelity between two normalized quantum statevectors.

    Args:
        target_state (numpy.ndarray): The expected normalized statevector.
        actual_state (numpy.ndarray): The resulting normalized statevector.

    Returns:
        float: The fidelity between the two statevectors. 0 no fidelity, 1 same vector.
    """
    if len(target_state) != len(actual_state):
        raise ValueError("Statevectors must have the same dimension.")

    target_norm = np.linalg.norm(target_state)
    actual_norm = np.linalg.norm(actual_state)

    if not np.isclose(target_norm, 1.0) or not np.isclose(actual_norm, 1.0):
        raise ValueError("Statevectors must be normalized.")

    inner_product = np.dot(np.conjugate(target_state), actual_state)
    return np.abs(inner_product) ** 2


def kron(A, B):
    return jnp.kron(A, B)


def overlap(state1, state2):
    return np.abs(np.vdot(state1, state2)) ** 2


def diff(v1, v2):
    errors = (v1 - v2)
    return np.mean(errors)


def magnitude_spectrum(c_n):
    magnitude = np.abs(c_n)
    return magnitude

def state_normalized(statevector):
    """Checks if a statevector is normalized. Returns bool"""
    norm = np.sum(np.abs(statevector.data) ** 2)
    return np.isclose(norm, 1.0)


def fourier_series(x, coeffs):
    return sum(c * np.cos(2 * np.pi * k * x) for k, c in enumerate(coeffs))


def prob(statevector):
    return np.abs(statevector) ** 2

# normalized [0, 1]
def normalized_ground_state_prob(statevector):
    if isinstance(statevector, Statevector):
        statevector = statevector.data
    # print(statevector)
    # print(prob(statevector))
    # print(statevector[0])
    # print(prob(statevector[0]))
    return 2*prob(statevector[0])-1


# ENTANGLEMENT
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

def is_three_qubit_entangled(state_vector, epsilon=1e-10):
    """
    Checks for entanglement in a three-qubit pure state using Schmidt decomposition
    across the A-BC bipartition.

    Args:
        state_vector (Statevector or numpy.ndarray): The three-qubit statevector.
        epsilon (float): Tolerance for considering singular values as zero.

    Returns:
        bool: True if entangled (based on A-BC bipartition), False otherwise.
    """
    if isinstance(state_vector, Statevector):
        psi = state_vector.data
    else:
        psi = np.array(state_vector)

    if len(psi) != 8:
        raise ValueError("Input statevector must have dimension 8 for three qubits.")

    # Reshape for A vs BC bipartition (qubit 0 vs qubits 1 and 2)
    # Indices: (q0 q1 q2) -> reshape to (q0, q1 q2)
    matrix_AB_C = psi.reshape((2, 4))

    # Perform Singular Value Decomposition
    U, s, Vh = svd(matrix_AB_C)

    # Schmidt rank is the number of non-zero singular values
    schmidt_rank = np.sum(s > epsilon)

    return schmidt_rank > 1

def is_two_qubit_entangled(state_vector):
    """
    Checks if a two-qubit statevector is entangled.

    Args:
        state_vector (Statevector or numpy.ndarray): The two-qubit statevector.

    Returns:
        bool: True if entangled, False otherwise.
    """
    if isinstance(state_vector, Statevector):
        psi = state_vector.data
    else:
        psi = np.array(state_vector)

    if len(psi) != 4:
        raise ValueError("Input statevector must have dimension 4 for two qubits.")

    # Coefficients of the basis states |00>, |01>, |10>, |11>
    a = psi[0]
    b = psi[1]
    c = psi[2]
    d = psi[3]

    # A two-qubit state is separable if and only if:
    # ac = bd
    return not np.isclose(a * d, b * c)


# PARAMETER GENERATION CIRCUIT 15
def random_parameter_set2(num_samples, ansatze, num_qubits, num_gates):
    parameter_set = []
    for times in range(num_samples):
        parameter_set.append(np.random.uniform(low=-np.pi, high=np.pi, size=(ansatze, num_qubits * num_gates)))
    return parameter_set


# PARAMETER GENERATION

def typical_phases():
    start = -2 * np.pi
    stop = 2 * np.pi
    return np.arange(start, stop + np.pi / 4, np.pi / 4)


def typical_theta(num="one"):
    theta_values = [-np.pi, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, np.pi]
    if num == "one":
        return np.random.choice(theta_values)
    elif num == "all":
        return theta_values
    elif isinstance(num, int):
        return np.linspace(0, np.pi, num)
    else:
        raise ValueError("Invalid input for 'num'. It should be 'one', 'all', or an integer.")


def random_theta():
    return np.random.uniform(-2 * np.pi, 2 * np.pi)


def random_theta_positive():
    return np.random.uniform(0, 2 * np.pi)


# GATE PARAMETER GENERATION
def gradual_parameter_set(num_layer, num_qubits, num_gates, num_samples):
    parameter_set = []
    for theta in typical_theta(num_samples):
        parameter_set.append(fixed_parameter(num_layer, num_qubits, num_gates, theta))
    return parameter_set

def fixed_parameter_set(num_layer, num_qubits, num_gates, num_samples, fixed_param):
    parameter_set = []
    for _ in range(num_samples):
        parameter_set.append(fixed_parameter(num_layer, num_qubits, num_gates, fixed_param))
    return parameter_set


def fixed_parameter(num_layer, num_qubits, num_gates, fixed_param):
    return np.full((num_layer, num_qubits, num_gates), fixed_param)


def random_parameter_set(num_layer, num_qubits, num_gates, num_samples):
    parameter_set = []
    for times in range(num_samples):
        parameter_set.append(random_parameter(num_layer, num_qubits, num_gates))
    return parameter_set


def random_parameter(num_layer, num_qubits, num_gates):
    return np.random.uniform(low=-2 * np.pi, high=2 * np.pi, size=(num_layer, num_qubits, num_gates))
