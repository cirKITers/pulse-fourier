import numpy as np

def prob(statevector):
    return np.abs(statevector) ** 2

def fourier_series(x, coeffs):
    return sum(c * np.cos(2 * np.pi * k * x) for k, c in enumerate(coeffs))


