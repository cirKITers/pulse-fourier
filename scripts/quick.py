import matplotlib

from utils.definitions import *
from utils.helpers import *
from pulse.envelope import *
from qiskit_dynamics import *

matplotlib.use('TkAgg')  # Force TkAgg backend
import matplotlib.pyplot as plt  # import after setting backend


signal1 = Signal(
    envelope=gaussian_envelope,
    carrier_freq=vu,
    phase=0.0,
)

# signal1 = DiscreteSignal

signal1.draw(t0=-T, tf=T, n=1000, function="signal")


signal = signal_mapping(np.ones(80) * 1e8)
signal.draw(t0=-T, tf=signal.T * signal.dt, n=1000, function='envelope')
plt.show()



























# import pennylane as qml
# import numpy as np
# A = np.array([
#     [1.0 + 0.0j,  0.2 + 0.1j, -0.3 + 0.2j,  0.4 + 0.3j,  0.5 + 0.4j,  0.6 + 0.5j,  0.7 + 0.6j,  0.8 + 0.7j],
#     [0.2 - 0.1j,  2.0 + 0.0j,  0.1 + 0.3j, -0.2 + 0.4j, -0.3 + 0.5j, -0.4 + 0.6j, -0.5 + 0.7j, -0.6 + 0.8j],
#     [-0.3 - 0.2j,  0.1 - 0.3j,  3.0 + 0.0j,  0.3 + 0.5j,  0.2 + 0.6j,  0.1 + 0.7j,  0.0 + 0.8j, -0.1 + 0.9j],
#     [0.4 - 0.3j, -0.2 - 0.4j,  0.3 - 0.5j,  4.0 + 0.0j, -0.1 + 0.7j, -0.2 + 0.8j, -0.3 + 0.9j, -0.4 + 1.0j],
#     [0.5 - 0.4j, -0.3 - 0.5j,  0.2 - 0.6j, -0.1 - 0.7j,  5.0 + 0.0j,  0.0 + 0.9j,  0.1 + 1.0j,  0.2 + 1.1j],
#     [0.6 - 0.5j, -0.4 - 0.6j,  0.1 - 0.7j, -0.2 - 0.8j,  0.0 - 0.9j,  6.0 + 0.0j, -0.1 + 1.1j, -0.2 + 1.2j],
#     [0.7 - 0.6j, -0.5 - 0.7j,  0.0 - 0.8j, -0.3 - 0.9j,  0.1 - 1.0j, -0.1 - 1.1j,  7.0 + 0.0j,  0.0 + 1.2j],
#     [0.8 - 0.7j, -0.6 - 0.8j, -0.1 - 0.9j, -0.4 - 1.0j,  0.2 - 1.1j, -0.2 - 1.2j,  0.0 - 1.2j,  8.0 + 0.0j]
# ])
# H = qml.pauli_decompose(A, pauli=True)
#
# print(H)


# fid = statevector_fidelity(SUPERPOSITION_STATE_H(1), Statevector([0.5 + 0.5j, 0.5 + 0.5j]).data)
#
# print(fid)

