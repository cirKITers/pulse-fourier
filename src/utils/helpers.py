import numpy as np

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

