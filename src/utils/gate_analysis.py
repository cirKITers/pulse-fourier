import numpy as np
from matplotlib import pyplot as plt

from gate_coefficients import fourier_coefficients, fourier_series
from basic_q_fourier import BasicQFourier1
from helpers import random_parameter

# Hyper
QuantumModel = BasicQFourier1
simulator = 'statevector_simulator'
num_layer = 3
num_qubits = 1
shots = 32768

interval = [0, 1]
points = 200

# Analysis
num_samples = 100
num_coeff = 5

coeffs_cos = np.zeros((num_samples, num_coeff))
coeffs_sin = np.zeros((num_samples, num_coeff))
# coeffs_cos2 = np.zeros((num_samples, num_coeff))
# coeffs_sin2 = np.zeros((num_samples, num_coeff))
for _ in range(num_samples):

    params = random_parameter(1, num_layer, num_qubits)

    qm = QuantumModel(num_qubits, num_layer, params)

    x, f_x = qm.predict_interval(simulator, shots, interval, points, plot=False)

    f_x = f_x - np.mean(f_x)

    a, b, c = fourier_coefficients(x, f_x, num_coeff=num_coeff)

    f_series = fourier_series(x, f_x, a, b, plot=False)

    coeffs_cos[_, :] = a
    coeffs_sin[_, :] = b
    # test difference
    # coeffs_cos2[_, :] = np.real(a)
    # coeffs_sin2[_, :] = -2 * np.imag(b)
    # print(np.array_equal(coeffs_cos, coeffs_cos2))
    # print(np.array_equal(coeffs_sin, coeffs_sin2))

# make generalized code ncoef != num_coef
n_coeff = len(coeffs_cos[0])

fig, ax = plt.subplots(1, n_coeff, figsize=(15, 4))

for idx, ax_ in enumerate(ax):
    ax_.set_title(r"$c_{:02d}$".format(idx))
    ax_.scatter(
        coeffs_cos[:, idx],
        coeffs_sin[:, idx],
        s=20,
        facecolor="white",
        edgecolor="red",
    )
    ax_.set_aspect("equal")
    ax_.set_ylim(-1, 1)
    ax_.set_xlim(-1, 1)


plt.tight_layout(pad=0.5)
plt.show()




