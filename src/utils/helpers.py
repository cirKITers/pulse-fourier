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
GROUND_STATE = Statevector([1.0, 0.0])
EXCITED_STATE = Statevector([0.0, 1.0])
SUPERPOSITION_STATE = Statevector([1/np.sqrt(2), 1/np.sqrt(2)])
PHASE_SHIFTED_STATE = Statevector([1/np.sqrt(2), 1j/np.sqrt(2)])
RANDOM_STATE_A = Statevector([0.8, 0.6])

def RANDOM_STATE():
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

def random_parameter_set(repetitions, layer, qubits, set_number):
    parameter_set = []
    for times in range(set_number):
        parameter_set.append(random_parameter(repetitions, layer, qubits))
    return parameter_set

def random_parameter(repetitions, layer, qubits):
    """
    Generates a random parameter array with values between -2*pi and 2*pi.
    """
    return np.random.uniform(low=-2 * np.pi, high=2 * np.pi, size=(repetitions, layer, qubits))

def random_parameter2(repetitions, layer, qubits):
    """

    :param repetitions: number of parameter dependent gates
    :param layer: number of layer
    :param qubits: number of qubits
    :return: set of random weights
    """
    np_arr = 2 * np.pi * np.random.random(size=(repetitions, layer, qubits))
    return np_arr.flatten().tolist()

# def random_weights():
#     return 2 * np.pi * np.random.random(size=(2, n_ansatz_layers, n_qubits))

# from Tutorial, what is it for?
def gaussian_conv(t, _dt, sigma):
    return Convolution(2. * _dt / np.sqrt(2. * np.pi * sigma ** 2) * np.exp(-t ** 2 / (2 * sigma ** 2)))
