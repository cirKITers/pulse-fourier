import numpy as np
from matplotlib import pyplot as plt


def subplot(coeffs_cos, coeffs_sin):
    num_samples = coeffs_cos.shape[0]
    num_coeff = coeffs_cos.shape[1]

    num_fx = num_samples  # Assuming num_fx is the same as num_samples for consistency
    cmap = plt.cm.get_cmap('viridis', num_fx)

    # Assuming coeffs_cos and coeffs_sin are already calculated

    fig, ax = plt.subplots(1, num_coeff, figsize=(20, 10))

    for idx, ax_ in enumerate(ax):
        ax_.set_title(r"$c_{:02d}$".format(idx))
        for i in range(num_samples):
            color = cmap(i / num_fx)
            size = 15 + (25 - 15) * (i / (num_samples - 1)) if num_samples > 1 else 20
            ax_.scatter(
                coeffs_cos[i, idx],
                coeffs_sin[i, idx],
                s=size,
                facecolor="white",
                edgecolor=color,
            )
        ax_.set_aspect("equal")
        ax_.set_ylim(-1, 1)
        ax_.set_xlim(-1, 1)

    # Add a legend to the first subplot
    # ax[0].legend(fontsize='small')
    plt.tight_layout(pad=0.5)
    plt.show()









def plot_multiple_fourier_series(coefficients_list):
    """
    Plots the real and imaginary parts of multiple Fourier series, each with a different color.

    Args:
        coefficients_list: A list of lists or numpy arrays. Each inner list/array
                           contains 10 complex Fourier coefficients for one series.
    """
    num_fx = len(coefficients_list)
    if num_fx == 0:
        print("No Fourier series to plot.")
        return

    cmap = plt.cm.get_cmap('viridis', num_fx)  # Get a colormap

    plt.figure(figsize=(16, 8))  # Make the figure bigger

    for i, coefficients in enumerate(coefficients_list):
        if len(coefficients) != 10:
            print(f"Warning: Fourier series {i+1} has {len(coefficients)} coefficients, expected 10. Skipping.")
            continue

        t = np.linspace(0, 1, 500)
        f_t = np.zeros_like(t, dtype=complex)

        f_t += coefficients[0]

        for n in range(1, 5):
            f_t += coefficients[n] * np.exp(2j * np.pi * n * t)
            f_t += np.conj(coefficients[n]) * np.exp(-2j * np.pi * n * t)

        f_t += coefficients[5] * np.exp(2j * np.pi * 5 * t)
        f_t += np.conj(coefficients[5]) * np.exp(-2j * np.pi * 5 * t)

        color = cmap(i / num_fx)  # Get color for this series

        plt.subplot(1, 2, 1)  # Real part subplot
        plt.plot(t, np.real(f_t), color=color, label=f'Series {i+1}')
        plt.title("Real Part of Fourier Series")
        plt.xlabel("t")
        plt.ylabel("Re(f(t))")
        plt.grid(True)

        plt.subplot(1, 2, 2)  # Imaginary part subplot
        plt.plot(t, np.imag(f_t), color=color, label=f'Series {i+1}')
        plt.title("Imaginary Part of Fourier Series")
        plt.xlabel("t")
        plt.ylabel("Im(f(t))")
        plt.grid(True)

    plt.subplot(1, 2, 1)
    plt.legend()  # Show legend in the real part plot

    plt.subplot(1, 2, 2)
    plt.legend()  # Show legend in the imaginary part plot

    plt.tight_layout()
    plt.show()


