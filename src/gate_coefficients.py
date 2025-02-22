import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft, ifft, rfft
from scipy.integrate import quad
from sklearn.metrics import mean_squared_error

from src.utils.helpers import random_parameter


# task: Analysis how derivable from gate level

def fourier_coefficients_fft(x, f_x, num_coeff=10, complex_valued_fx=False):
    f_x = f_x - np.mean(f_x)
    N = len(f_x)
    if complex_valued_fx:
        fourier_transform = fft
    else:
        fourier_transform = rfft
    c_n = fourier_transform(f_x) / N  # complex coefficients, divided by N for scaling

    # exponential form
    c_n = c_n[:num_coeff]   # complex coefficients

    # trigonometric form
    a_n = 2 * np.real(c_n)  # describes cosinus part
    b_n = -2 * np.imag(c_n)   # describes sinus part
    a_n[0] = a_n[0] / 2

    return a_n, b_n, c_n

# much higher mse in exponential form
def fourier_series_exp(x, f_x, c_n, plot=False):
    T = (x[-1] - x[0])  # works only if x span is one period, general compute still necessary
    num_coeff = len(c_n)

    f_fourier_series = np.zeros_like(x, dtype=complex)

    for n in range(num_coeff):
        f_fourier_series += c_n[n] * np.exp(1j * 2 * np.pi * n * x / T)

    mse = mean_squared_error(f_x, f_fourier_series.real)  # We compare the real part of the result

    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(x, f_x, '--', label="Input f(x)")
        plt.plot(x, f_fourier_series.real, label="Fourier Reconstruction")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.title("Fourier Series (Exponential Form) vs. Input Function")
        plt.show()

    return f_fourier_series.real, mse

# Better mse values in trigonometric form
def fourier_series_tri(x, f_x, a_n, b_n, plot=False):
    T = (x[-1] - x[0])  # works only if x span is one period, general compute still necessary
    num_coeff = len(a_n)    # num_coeff = 7     # len(f_x) // 2
    f_fourier_series = np.full_like(x, a_n[0] / 2)
    for n in range(1, num_coeff):  # Symmetrische Fourier-Koeffizienten
        f_fourier_series += a_n[n] * np.cos(2 * np.pi * n * x / T) \
                            + b_n[n] * np.sin(2 * np.pi * n * x / T)

    mse = mean_squared_error(f_x, f_fourier_series)

    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(x, f_x, '--', label="Input f(x)")
        plt.plot(x, f_fourier_series, '--', label="Fourier Reconstruction")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.title("Fourier series vs. Input function")
        plt.show()

    return f_fourier_series, mse


def coefficient_distribution(num_samples, num_coeff, quantum_model, num_layer, num_qubits, simulator, shots, interval, points):
    coeffs_cos = np.zeros((num_samples, num_coeff))
    coeffs_sin = np.zeros((num_samples, num_coeff))
    coeffs_all = np.zeros((num_samples, num_coeff), dtype=np.complex128)

    for _ in range(num_samples):

        params = random_parameter(1, num_layer, num_qubits)

        qm = quantum_model(num_qubits, num_layer, params)

        x, f_x = qm.predict_interval(simulator, shots, interval, points, plot=False)

        f_x = f_x - np.mean(f_x)

        a, b, c = fourier_coefficients_fft(x, f_x, num_coeff=num_coeff)

        f_series, mse = fourier_series_tri(x, f_x, a, b, plot=False)
        print(f"Fourier Approximation MSE: {mse:.6f}")

        coeffs_cos[_, :] = a
        coeffs_sin[_, :] = b
        coeffs_all[_, :] = c

    fig, ax = plt.subplots(1, num_coeff, figsize=(15, 4))

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

    return coeffs_cos, coeffs_sin, coeffs_all

# Multiple ways to compute coefficients:
def fourier_coefficients_lstsq(x, f_x, num_coeff=5):
    T = x[-1] - x[0]
    A = np.ones((len(x), 2 * num_coeff))  # Design matrix

    for n in range(1, num_coeff):
        A[:, 2 * n - 1] = np.cos(2 * np.pi * n * x / T)
        A[:, 2 * n] = np.sin(2 * np.pi * n * x / T)

    coeffs, _, _, _ = np.linalg.lstsq(A, f_x, rcond=None)

    a_n = np.zeros(num_coeff)
    b_n = np.zeros(num_coeff)
    a_n[0] = coeffs[0]  # DC component
    for n in range(1, num_coeff):
        a_n[n] = coeffs[2 * n - 1]
        b_n[n] = coeffs[2 * n]
    c = None
    return a_n, b_n, c

# For continuous functions
def fourier_coefficients_integral(x, f_x, num_coeff=5):
    T = x[-1] - x[0]
    a_n = []
    b_n = []

    # Compute a_0 separately
    a_0 = (2 / T) * quad(lambda x: f_x(x), 0, T)[0]
    a_n.append(a_0 / 2)  # Divide by 2 to match Fourier series convention

    # Compute higher order coefficients
    for n in range(1, num_coeff):
        a_n.append((2 / T) * quad(lambda x: f_x(x) * np.cos(2 * np.pi * n * x / T), 0, T)[0])
        b_n.append((2 / T) * quad(lambda x: f_x(x) * np.sin(2 * np.pi * n * x / T), 0, T)[0])

    return np.array(a_n), np.array(b_n)
