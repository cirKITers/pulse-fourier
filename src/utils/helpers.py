import numpy as np
from qiskit.quantum_info import Operator, Statevector
import jax.numpy as jnp
from qiskit_dynamics.signals import Convolution
from scipy.linalg import expm

SIGMA_X = Operator(np.array([[0, 1], [1, 0]], dtype=complex))
SIGMA_Y = Operator(np.array([[0, -1j], [1j, 0]], dtype=complex))
SIGMA_Z = Operator(np.array([[1, 0], [0, -1]], dtype=complex))

I = jnp.array([[1, 0], [0, 1]], dtype=complex)
X = jnp.array([[0, 1], [1, 0]], dtype=complex)
Z = jnp.array([[1, 0], [0, -1]], dtype=complex)

CNOT_MATRIX = jnp.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)

# Two-qubit operator helpers
def kron(A, B):
    return jnp.kron(A, B)


I0 = kron(I, I)  # Identity on two-qubit space
Z0 = kron(Z, I)  # Z on qubit 0
Z1 = kron(I, Z)  # Z on qubit 1
X0 = kron(X, I)  # X on qubit 0
X1 = kron(I, X)  # X on qubit 1
P1 = 0.5 * (I0 - Z0)


# U_static = expm(-1j * H_static * t_max)

# ONE QUBIT STATES
GROUND_STATE = Statevector([1.0, 0.0])
EXCITED_STATE = Statevector([0.0, 1.0])
SUPERPOSITION_STATE = Statevector([1/np.sqrt(2), 1/np.sqrt(2)])
PHASE_SHIFTED_STATE = Statevector([1/np.sqrt(2), 1j/np.sqrt(2)])
RANDOM_STATE_A = Statevector([0.8, 0.6])

# TWO QUBIT STATES: |ψ⟩ = α|00⟩ + β|01⟩ + γ|10⟩ + δ|11⟩ s.t. |α|² + |β|² + |γ|² + |δ|² = 1
GROUND_GROUND = Statevector([1.0, 0.0, 0.0, 0.0])
GROUND_EXCITED = Statevector([0.0, 1.0, 0.0, 0.0])
BELL_STATE = Statevector([1/np.sqrt(2), 0.0, 0.0, 1/np.sqrt(2)])
SUPERPOSITION_PRODUCT = Statevector([0.5, 0.5, 0.5, 0.5])

def RANDOM_STATE(num_q=1):
    """
    Generates a random pure quantum statevector for a given number of qubits.

    Args:
        num_q (int): The number of qubits for the statevector.

    Returns:
        Statevector: A random pure quantum statevector.
    """
    if num_q <= 0:
        raise ValueError("Number of qubits must be at least 1.")

    num_amplitudes = 2**num_q
    amplitudes = np.random.randn(num_amplitudes) + 1j * np.random.randn(num_amplitudes)
    norm = np.sqrt(np.sum(np.abs(amplitudes)**2))

    if norm == 0:
        raise ValueError("Generated amplitudes have a norm of zero. This is unlikely but possible with random generation.")

    normalized_amplitudes = amplitudes / norm
    return Statevector(normalized_amplitudes)

def RANDOM_STATE_old():
    alpha_real = np.random.uniform(-1, 1)
    alpha_imag = np.random.uniform(-1, 1)
    beta_real = np.random.uniform(-1, 1)
    beta_imag = np.random.uniform(-1, 1)

    alpha = alpha_real + 1j * alpha_imag
    beta = beta_real + 1j * beta_imag

    norm = np.sqrt(np.abs(alpha) ** 2 + np.abs(beta) ** 2)

    alpha /= norm
    beta /= norm

    return Statevector([alpha, beta])

def random_theta():
    return np.random.uniform(-2*np.pi, 2*np.pi)

# should later return real result not list
def swap_amplitudes(trajectory):
    result = [trajectory[0]]
    for state in trajectory[1:]:
        result.append(Statevector([state[1], state[0]]))
    return result

def quadrant_selective_conjugate_transpose(trajectory):
    result = [trajectory[0]]
    for state in trajectory[1:]:
        new_amps = [complex(state.data[0].imag, -state.data[0].real), complex(-state.data[1].imag, state.data[1].real)]
        result.append(Statevector(new_amps))
    return result

def swap_probs(probs):
    return np.array([probs[1], probs[0]])


def diff(v1, v2):
    errors = (v1 - v2)
    return np.mean(errors)


def state_normalized(statevector):
    norm = np.sum(np.abs(statevector.data) ** 2)
    return np.isclose(norm, 1.0)

def overlap(state1, state2):
    return np.abs(np.vdot(state1, state2)) ** 2

def X_fidelity(u):
    return jnp.abs(jnp.sum(SIGMA_X * u))**2 / 4.

# like in: https://qiskit-community.github.io/qiskit-dynamics/tutorials/optimizing_pulse_sequence.html
def static_hamiltonian(omega):
    return 2 * np.pi * omega * SIGMA_Z / 2

def drive_hamiltonian(drive_strength):
    if isinstance(drive_strength, np.ndarray):
        drive_strength = drive_strength[0]
    return 2 * np.pi * drive_strength * SIGMA_X / 2

def magnitude_spectrum(c_n):
    magnitude = np.abs(c_n)
    return magnitude

def prob(statevector):
    return np.abs(statevector) ** 2

def fourier_series(x, coeffs):
    return sum(c * np.cos(2 * np.pi * k * x) for k, c in enumerate(coeffs))


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


# from Tutorial, what is it for?
def gaussian_conv(t, _dt, sigma):
    return Convolution(2. * _dt / np.sqrt(2. * np.pi * sigma ** 2) * np.exp(-t ** 2 / (2 * sigma ** 2)))
