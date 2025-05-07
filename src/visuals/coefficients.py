import random

import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

def visualize_complex_colors(fx_set, complex_colors):
    """
    Visualizes the complex color data as a heatmap, with samples on the x-axis
    and 10 dependent color values on the y-axis.  No labels or axes are shown.

    Args:
        fx_set: A list of samples.  Each sample is assumed to have 10
            dependent values, which will be visualized as colors.
        complex_colors: A list of color names (strings) corresponding to the
            dependent values for each sample.  It is assumed that
            len(complex_colors) == len(fx_set) * 10.
    """
    # Ensure that fx_set and complex_colors are not None and are not empty
    if not fx_set:
        raise ValueError("fx_set cannot be None or empty")
    if not complex_colors:
        raise ValueError("complex_colors cannot be None or empty")

    num_samples = len(fx_set)
    num_values_per_sample = 10  # Assuming 10 dependent values per sample

    # Ensure the length of complex_colors matches the expected total number of values
    if len(complex_colors) != num_samples * num_values_per_sample:
        raise ValueError(f"Expected {num_samples * num_values_per_sample} colors, but got {len(complex_colors)}")
    # Reshape the complex_colors list into a 2D array
    color_matrix = np.array(complex_colors).reshape(num_samples, num_values_per_sample)

    # Define the color mapping
    color_map = {
        "red": (1, 0, 0),
        "blue": (0, 0, 1),
        "green": (0, 1, 0),
        "purple": (0.5, 0, 0.5)
    }

    # Convert color names to RGB values
    rgb_matrix = np.zeros((num_samples, num_values_per_sample, 3))
    for i in range(num_samples):
        for j in range(num_values_per_sample):
            color_name = color_matrix[i, j]
            rgb_matrix[i, j, :] = color_map[color_name]

    # Create the heatmap plot
    plt.figure(figsize=(num_samples, num_values_per_sample))  # Adjust figure size as needed
    plt.imshow(rgb_matrix, aspect='auto')  # Use 'auto' to adjust aspect ratio
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks
    plt.axis('off')  # Turn off axes
    plt.tight_layout() # tight layout
    plt.show()


def subplot(coeffs_cos, coeffs_sin, coloring="gradual"):
    num_samples = coeffs_cos.shape[0]
    num_coeff = coeffs_cos.shape[1]

    num_fx = num_samples  # Assuming num_fx is the same as num_samples for consistency
    cmap = plt.cm.get_cmap('viridis', num_fx)
    colors = [(random.random(), random.random(), random.random()) for _ in range(num_fx)]

    # Assuming coeffs_cos and coeffs_sin are already calculated

    fig, ax = plt.subplots(1, num_coeff, figsize=(20, 10))

    print(f"starting subplots... (0-{num_coeff-1})")

    for idx, ax_ in enumerate(ax):

        print(f"subplot number: {idx}")

        ax_.set_title(r"$c_{:02d}$".format(idx))
        for i in range(num_samples):
            if coloring == "gradual":
                color = cmap(i / num_fx)

            elif coloring == "random":
                color = colors[i]

            else:
                color = complex_coloring(coeffs_cos[i, idx] + 1j * coeffs_sin[i, idx])

            # size = 15 + (25 - 15) * (i / (num_samples - 1)) if num_samples > 1 else 20
            size = 20

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


def complex_coloring(Z):
    """
    Returns a NumPy array of colors (blue, red, green, or purple) based on which component
    (positive real, negative real, positive imaginary, negative imaginary)
    has the largest magnitude for each complex number in the input NumPy array.

    Args:
        Z (np.ndarray): A NumPy array of complex numbers with shape (num_samples, num_coeffs).

    Returns:
        np.ndarray: A NumPy array of strings (colors) with the same shape as Z.
    """
    real_part = np.real(Z)
    imag_part = np.imag(Z)

    abs_real_pos = np.where(real_part > 0, np.abs(real_part), -1)
    abs_real_neg = np.where(real_part < 0, np.abs(real_part), -1)
    abs_imag_pos = np.where(imag_part > 0, np.abs(imag_part), -1)
    abs_imag_neg = np.where(imag_part < 0, np.abs(imag_part), -1)

    magnitudes = np.stack([abs_real_pos, abs_real_neg, abs_imag_pos, abs_imag_neg], axis=-1)
    dominant_indices = np.argmax(magnitudes, axis=-1)

    colors = np.array(["red", "blue", "green", "purple"])
    scientific_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

    # 'Set1', 'Paired', or 'Dark2'.

    # cmap = cm.get_cmap('Set1', 4)  # 'Set1' with 4 distinct colors
    # scientific_colors = cmap.colors

    cmap = ListedColormap(scientific_colors)
    bounds = np.arange(len(scientific_colors) + 1)  # [0, 1, 2, 3, 4]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
    im = ax.imshow(dominant_indices, cmap=cmap, norm=norm, aspect='auto')  # 'auto' for aspect ratio

    # Remove x-axis tick labels (for large num_samples)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("Samples")
    ax.set_xlabel("Coefficients")  # General label, no ticks

    # Add legend
    labels_with_meaning = [
        mpatches.Patch(color="red", label="Positive Real"),
        mpatches.Patch(color="blue", label="Negative Real"),
        mpatches.Patch(color="green", label="Positive Imaginary"),
        mpatches.Patch(color="purple", label="Negative Imaginary")
    ]
    plt.legend(handles=labels_with_meaning, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # patches = [mpatches.Patch(color=colors[i], label=colors[i].capitalize()) for i in range(len(colors))]
    # plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.title("Samples to coefficients coloring")
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.show()

    z_color = colors[dominant_indices]
    return z_color

#
#
# def plot_multiple_fourier_series(coefficients_list):
#     """
#     Plots the real and imaginary parts of multiple Fourier series, each with a different color.
#
#     Args:
#         coefficients_list: A list of lists or numpy arrays. Each inner list/array
#                            contains 10 complex Fourier coefficients for one series.
#     """
#     num_fx = len(coefficients_list)
#     if num_fx == 0:
#         print("No Fourier series to plot.")
#         return
#
#     cmap = plt.cm.get_cmap('viridis', num_fx)  # Get a colormap
#
#     plt.figure(figsize=(16, 8))  # Make the figure bigger
#
#     for i, coefficients in enumerate(coefficients_list):
#         if len(coefficients) != 10:
#             print(f"Warning: Fourier series {i+1} has {len(coefficients)} coefficients, expected 10. Skipping.")
#             continue
#
#         t = np.linspace(0, 1, 500)
#         f_t = np.zeros_like(t, dtype=complex)
#
#         f_t += coefficients[0]
#
#         for n in range(1, 5):
#             f_t += coefficients[n] * np.exp(2j * np.pi * n * t)
#             f_t += np.conj(coefficients[n]) * np.exp(-2j * np.pi * n * t)
#
#         f_t += coefficients[5] * np.exp(2j * np.pi * 5 * t)
#         f_t += np.conj(coefficients[5]) * np.exp(-2j * np.pi * 5 * t)
#
#         color = cmap(i / num_fx)  # Get color for this series
#
#         plt.subplot(1, 2, 1)  # Real part subplot
#         plt.plot(t, np.real(f_t), color=color, label=f'Series {i+1}')
#         plt.title("Real Part of Fourier Series")
#         plt.xlabel("t")
#         plt.ylabel("Re(f(t))")
#         plt.grid(True)
#
#         plt.subplot(1, 2, 2)  # Imaginary part subplot
#         plt.plot(t, np.imag(f_t), color=color, label=f'Series {i+1}')
#         plt.title("Imaginary Part of Fourier Series")
#         plt.xlabel("t")
#         plt.ylabel("Im(f(t))")
#         plt.grid(True)
#
#     plt.subplot(1, 2, 1)
#     plt.legend()  # Show legend in the real part plot
#
#     plt.subplot(1, 2, 2)
#     plt.legend()  # Show legend in the imaginary part plot
#
#     plt.tight_layout()
#     plt.show()
