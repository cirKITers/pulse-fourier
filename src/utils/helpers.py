import numpy as np
from qiskit.quantum_info import Operator, Statevector
import jax.numpy as jnp
from qiskit_dynamics.signals import Convolution


# GENERAL HELPFUL
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

def random_theta():
    return np.random.uniform(-2*np.pi, 2*np.pi)

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


