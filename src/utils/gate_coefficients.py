import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft, ifft, rfft
import sympy as sp
from sklearn.metrics import mean_squared_error

# task: Analysis how derivable from gate level
def fourier_coefficients(x, f_x, num_coeff=10, complex_valued_fx=False):
    f_x = f_x - np.mean(f_x)
    N = len(f_x)
    if complex_valued_fx:
        fourier_transform = fft
    else:
        fourier_transform = rfft
    c_n = fourier_transform(f_x) / N  # complex coefficients, divided by N for scaling

    c_n = c_n[:num_coeff]

    a_n = 2 * np.real(c_n)  # describes cosinus part
    b_n = -2 * np.imag(c_n)     # describes sinus part
    a_n[0] = a_n[0] / 2

    return a_n, b_n, c_n


def fourier_series(x, f_x, a_n, b_n, plot=True):
    T = (x[-1] - x[0])  # works only if x span is one period, general compute still necessary
    num_coeff = len(a_n)    # num_coeff = 7     # len(f_x) // 2
    f_fourier_series = np.full_like(x, a_n[0] / 2)
    for n in range(1, num_coeff):  # Symmetrische Fourier-Koeffizienten
        f_fourier_series += a_n[n] * np.cos(2 * np.pi * n * x / T) \
                            + b_n[n] * np.sin(2 * np.pi * n * x / T)

    mse = mean_squared_error(f_x, f_fourier_series)
    print(f"Fourier Approximation MSE: {mse:.6f}")

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

def estimate_period_zero_crossing(x, f_x):
    zero_crossings = np.where(np.diff(np.sign(f_x)))[0]
    if len(zero_crossings) > 1:
        T_est = np.mean(np.diff(x[zero_crossings])) * 2  # Assuming half-period per crossing
        return T_est
    else:
        return None

# def estimate_period(x, f_x):
#     peaks, _ = find_peaks(f_x)  # Find local maxima
#     if len(peaks) > 1:
#         T_est = np.mean(np.diff(x[peaks]))  # Average distance between peaks
#         return T_est
#     else:
#         return None


# def both(x, f_x, plot=True):
#     N = len(f_x)
#     complex_valued_f = False
#     if complex_valued_f:
#         fourier_transform = fft
#     else:
#         fourier_transform = rfft
#     f_x_fft = fourier_transform(f_x) / N  # frequency components/coefficients
#
#     # Fourier coefficients
#     a_n = 2 * np.real(f_x_fft)
#     b_n = -2 * np.imag(f_x_fft)
#     a_n[0] = a_n[0] / 2
#
#     print(f_x)
#     print(x)
#     f_fourier_series = np.zeros_like(f_x)
#     for n in range(N//2):  # Symmetrische Fourier-Koeffizienten
#         f_fourier_series += a_n[n] * np.cos(2 * np.pi * n * x / (x[-1] - x[0])) \
#                             + b_n[n] * np.sin(2 * np.pi * n * x / (x[-1] - x[0]))
#
#     mse = mean_squared_error(f_x, f_fourier_series)
#     print(f"Fourier Approximation MSE: {mse:.6f}")
#
#     if plot:
#         plt.figure(figsize=(8, 4))
#         plt.plot(x, f_x, '--', label="Original f(x)")
#         plt.plot(x, f_fourier_series, '--', label="Fourier-Rekonstruktion")
#         plt.xlabel("x")
#         plt.ylabel("f(x)")
#         plt.legend()
#         plt.title("Fourier-Reihe vs. Originalfunktion")
#         plt.show()
#     return a_n, b_n

