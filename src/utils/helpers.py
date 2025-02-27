import numpy as np
from qiskit.quantum_info import Operator, Statevector
import jax.numpy as jnp
from qiskit_dynamics.signals import Convolution

SIGMA_X = Operator(np.array([[0, 1], [1, 0]], dtype=complex))
SIGMA_Y = Operator(np.array([[0, -1j], [1j, 0]], dtype=complex))
SIGMA_Z = Operator(np.array([[1, 0], [0, -1]], dtype=complex))

INIT_STATE = Statevector([1.0, 0.0])

def X_fidelity(u):
    return jnp.abs(jnp.sum(SIGMA_X * u))**2 / 4.

# like in: https://qiskit-community.github.io/qiskit-dynamics/tutorials/optimizing_pulse_sequence.html
def static_hamiltonian(omega):
    return 2 * np.pi * omega * SIGMA_Z / 2

def drive_hamiltonian(drive_strength):
    return 2 * np.pi * drive_strength * SIGMA_X / 2

def magnitude_spectrum(c_n):
    magnitude = np.abs(c_n)
    return magnitude

def prob(statevector):
    return np.abs(statevector) ** 2

def fourier_series(x, coeffs):
    return sum(c * np.cos(2 * np.pi * k * x) for k, c in enumerate(coeffs))


def random_parameter(repetitions, layer, qubits):
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
