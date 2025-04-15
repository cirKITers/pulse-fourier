import numpy as np
from matplotlib import pyplot as plt
from qiskit.quantum_info import Operator, Statevector, partial_trace, DensityMatrix
import jax.numpy as jnp
from qiskit_dynamics.signals import Convolution


# GENERAL

def binary_c_t(n, c, t):
    if not (0 <= c < n and 0 <= t < n and c != t):
        return []

    indices = []
    for i in range(2**n):
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
        print("["+', '.join(formatted_array)+"]")
    elif isinstance(statevector, list) or isinstance(statevector, np.ndarray):
        formatted_array = [format_complex(x) for x in np.array(statevector)]
        print("["+', '.join(formatted_array)+"]")
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


def statevector_similarity(target_state, actual_state, tolerance=1e-6, scaling="linear"):
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
                similarity_sum += max(0.0, 1.0 - component_diff**2)
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
    return np.abs(inner_product)**2

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

def prob(statevector):
    return np.abs(statevector) ** 2

def state_normalized(statevector):
    """Checks if a statevector is normalized. Returns bool"""
    norm = np.sum(np.abs(statevector.data) ** 2)
    return np.isclose(norm, 1.0)

def fourier_series(x, coeffs):
    return sum(c * np.cos(2 * np.pi * k * x) for k, c in enumerate(coeffs))


# PARAMETER GENERATION

def typical_phases():
    start = -2*np.pi
    stop = 2*np.pi
    return np.arange(start, stop + np.pi / 4, np.pi / 4)

def typical_theta(num="one"):
    theta_values = [-np.pi, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, np.pi]
    if num == "one":
        return np.random.choice(theta_values)
    elif num == "all":
        return theta_values

def random_theta():
    return np.random.uniform(-2*np.pi, 2*np.pi)

def random_theta_positive():
    return np.random.uniform(0, 2 * np.pi)

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


# TRICKS, get rid off longterm
def swap_amplitudes(trajectory):
    result = [trajectory[0]]
    for state in trajectory[1:]:
        result.append(Statevector([state[1], state[0]]))
    return result

def quadrant_selective_conjugate_transpose(trajectory, num_qubits):
    """
    Applies a quadrant-selective conjugate transpose to each state in a trajectory.

    For each state in the trajectory, it takes the complex conjugate of the
    amplitudes and swaps the real and imaginary parts based on the bitstring
    representation of the basis state.

    Args:
        trajectory (list[Statevector]): A list of Statevector objects.

    Returns:
        list[Statevector]: A list of Statevector objects with the transformed amplitudes.
    """
    result = []
    for state in trajectory:
        new_amps = []
        for i in range(state.dim):
            bitstring = bin(i)[2:].zfill(num_qubits)  # Get bitstring representation
            real_part = state.data[i].real
            imag_part = state.data[i].imag
            if bitstring.count('1') % 2 == 0:  # Even number of 1s
                new_amps.append(complex(imag_part, -real_part))
            else:  # Odd number of 1s
                new_amps.append(complex(-imag_part, real_part))
        result.append(Statevector(new_amps))
    return result


def quadrant_selective_conjugate_transpose_OneQ(trajectory):
    result = [trajectory[0]]
    for state in trajectory[1:]:
        new_amps = [complex(state.data[0].imag, -state.data[0].real), complex(-state.data[1].imag, state.data[1].real)]
        result.append(Statevector(new_amps))
    return result

def swap_probs(probs):
    return np.array([probs[1], probs[0]])

#####
# from Tutorial, what is it for?
def gaussian_conv(t, _dt, sigma):
    return Convolution(2. * _dt / np.sqrt(2. * np.pi * sigma ** 2) * np.exp(-t ** 2 / (2 * sigma ** 2)))

# U_static = expm(-1j * H_static * t_max)


