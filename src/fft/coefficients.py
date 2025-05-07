import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft, rfft
from scipy.integrate import quad
from sklearn.metrics import mean_squared_error

from utils.data_handler import *

num_coefficients_standard = 8

def magnitude(z):
    return np.abs(z)


def coefficients(f_x, num_coeff=num_coefficients_standard, complex_valued_fx=False):

    f_x = f_x - np.mean(f_x)    # convention
    N = len(f_x)
    if complex_valued_fx:
        fourier_transform = fft
    else:
        fourier_transform = rfft        # in this case returns only N/2+1 many coeffs because of symmetrie
    c_n = fourier_transform(f_x) / N  # complex coefficients, divided by N for scaling

    # print(c_n)
    # exponential form
    # c_n = c_n[:num_coeff+1]  # complex coefficients truncated

    # trigonometric form
    # a_n = 2 * np.real(c_n)  # describes cosinus part
    # b_n = -2 * np.imag(c_n)  # describes sinus part
    # a_n[0] = a_n[0] / 2       # to get rid of the *2 from above, always practically 0
    return c_n  # Return from the second element onwards


def coefficient_set(fx_set, num_coeff=num_coefficients_standard):
    num_samples = len(fx_set)
    # coeffs_cos = np.zeros((num_samples, num_coeff))
    # coeffs_sin = np.zeros((num_samples, num_coeff))
    coeffs_all = np.zeros((num_samples, num_coeff), dtype=np.complex128)

    # very fast for multiple 1000 samples
    for sample in range(num_samples):

        c = coefficients(fx_set[sample], num_coeff=num_coeff)

        # f_series, mse = fourier_series_tri(x, f_x, a, b, plot=False)
        # print(f"Fourier Approximation MSE: {mse:.6f}")
        # coeffs_cos[sample, :] = a
        # coeffs_sin[sample, :] = b

        coeffs_all[sample, :] = c

    return coeffs_all


def order_coefficients_sets(complex_coeffs):
    magnitudes = np.abs(complex_coeffs)

    # Calculate the average magnitude for each of the 30 coefficients across all samples
    average_magnitudes = np.mean(magnitudes, axis=0)

    # Get the indices that would sort the average magnitudes in ascending order
    sorted_indices = np.argsort(average_magnitudes)

    # Order the coefficients based on their average magnitude
    ordered_coefficients = complex_coeffs[:, sorted_indices]

    # You can also get the sorted average magnitudes if you need them
    sorted_average_magnitudes = average_magnitudes[sorted_indices]

    # print("Original shape of complex coefficients:", complex_coeffs.shape)
    # print("Shape of ordered coefficients:", ordered_coefficients.shape)
    print("Sorted indices (indicating the original column order):", sorted_indices)
    # print("Sorted average magnitudes:", sorted_average_magnitudes)

    return ordered_coefficients


# def linearRegression(param_arrays, complex_coeffs, num_params, num_coeffs):
#     # 2. Calculate Magnitudes (as before, but now within the analysis)
#     magnitudes = np.abs(complex_coeffs)  # Shape: (5000, 3)
#
#     # b) Linear Regression (for each coefficient magnitude as a function of parameters)
#     from sklearn.linear_model import LinearRegression
#     import statsmodels.api as sm  # For more detailed statistics
#
#     # --- Linear Regression and Coefficient Extraction ---
#     regression_coefficients = np.zeros((num_params, num_coeffs))
#
#     for j in range(num_coeffs):
#         X = param_arrays  # Shape: (5000, 3)
#         y = magnitudes[:, j]  # Shape: (5000,)
#
#         # Using scikit-learn
#         model = LinearRegression()
#         model.fit(X, y)
#         # print(f"\nLinear Regression for Coefficient Magnitude {j+1} (scikit-learn):")
#         # print(f"  Coefficients: {model.coef_}")
#         # print(f"  Intercept: {model.intercept_}")
#         # print(f"  R-squared: {model.score(X, y)}")
#
#         # Using statsmodels for more detailed output
#         X_with_intercept = sm.add_constant(X)  # Add a constant term for the intercept
#         model_sm = sm.OLS(y, X_with_intercept).fit()
#         # print(f"\nLinear Regression for Coefficient Magnitude {j+1} (statsmodels):")
#         # print(model_sm.summary())
#         regression_coefficients[:, j] = model.coef_
#
#     # --- Plotting the Heatmap of Regression Coefficients ---
#     plt.figure(figsize=(10, 8))
#     im = plt.imshow(regression_coefficients, cmap="coolwarm", aspect="auto")
#
#     # Add colorbar
#     cbar = plt.colorbar(im)
#     cbar.set_label("Regression Coefficient")
#
#     # Add labels
#     plt.xticks(range(num_coeffs), [f"Coeff {i + 1}" for i in range(num_coeffs)])
#     plt.yticks(range(num_params), [f"Param {i + 1}" for i in range(num_params)])
#
#     plt.title("Heatmap of Linear Regression Coefficients")
#     plt.tight_layout()
#     plt.show()



def check_2pi_periodicity(x, y, tolerance=1e-6):
    """
    Checks if a function, given by its values at discrete points,
    exhibits a period of approximately 2*pi.

    Args:
        x (np.ndarray): Array of x-values.
        y (np.ndarray): Array of corresponding y-values (function values).
        tolerance (float, optional): Tolerance for comparing floating-point numbers.
                                     Defaults to 1e-6.

    Returns:
        bool: True if the function appears to have a period of 2*pi, False otherwise.
    """
    period = 2 * np.pi
    n_points = len(x)

    # We need to compare y(t) with y(t + 2pi) for various t.
    # Since we have discrete points, we need to find indices that correspond
    # to x_i and x_i + 2pi.

    for i in range(n_points):
        x_val = x[i]
        x_shifted = x_val + period

        # Find the index of the point closest to x_shifted
        idx_shifted = np.argmin(np.abs(x - x_shifted))

        print(x_val)
        print(x_shifted)
        print(idx_shifted)

        # Ensure the shifted point is within the bounds of our x array
        if np.abs(x[idx_shifted] - x_shifted) < (x[1] - x[0]) / 2:  # Check if close enough


            if not np.isclose(y[i], y[idx_shifted], atol=tolerance):
                print(i)
                return False  # Periodicity check failed for this point

    return True




#----------------------
# OLD: Computes for multiple fourier series (a set) samples the coefficients with FFT and returns a visual distribution
def coefficient_distribution_fft(num_samples, num_coeff, x, fx_set, plot=True):
    coeffs_cos = np.zeros((num_samples, num_coeff))
    coeffs_sin = np.zeros((num_samples, num_coeff))
    coeffs_all = np.zeros((num_samples, num_coeff), dtype=np.complex128)
    if isinstance(x, list):
        x = np.array(x)

    if isinstance(fx_set, list):
        fx_set = np.array(fx_set)

    for _ in range(num_samples):
        a, b, c = coefficients(fx_set[_], num_coeff=num_coeff)
        # f_series, mse = fourier_series_tri(x, f_x, a, b, plot=False)
        # print(f"Fourier Approximation MSE: {mse:.6f}")
        coeffs_cos[_, :] = a
        coeffs_sin[_, :] = b
        coeffs_all[_, :] = c

    if plot:
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


# ----------------------------------------
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
    num_coeff = len(a_n)  # num_coeff = 7     # len(f_x) // 2
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


# def coeffs_fft(f_x, num_coeff=10, complex_valued_fx=False):
#     f_x = f_x - np.mean(f_x)
#     N = len(f_x)
#     if complex_valued_fx:
#         fast_fourier_transform = fft
#     else:
#         fast_fourier_transform = rfft
#     c_n = fast_fourier_transform(f_x) / N  # complex coefficients, divided by N for scaling
#
#     # exponential form
#     c_n = c_n[:num_coeff]  # complex coefficients
#
#     return c_n
#
# def reconstruct_fx(cn, x, T):
#     """
#       Reconstructs the function values from the complex Fourier series coefficients.
#
#       Args:
#         cn (dict): A dictionary where keys are the integer indices (n)
#                    and values are the complex coefficients.
#         x (np.ndarray): The array of x-values at which to reconstruct the function.
#         T (float): The period of the original function.
#
#       Returns:
#         np.ndarray: The reconstructed function values.
#       """
#     fx_reconstructed = np.zeros_like(x, dtype=complex)
#     omega0 = 2 * np.pi / T
#     for n, c in cn.items():
#         fx_reconstructed += c * np.exp(1j * n * omega0 * x)
#     return np.real(fx_reconstructed)