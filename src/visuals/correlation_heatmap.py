import numpy as np
import matplotlib.pyplot as plt
import cmath

def complex_correlation(data: np.ndarray) -> np.ndarray:
    """
    Calculates the correlation matrix for complex-valued data.

    Args:
        data: A 2D NumPy array where rows represent coefficients
              (complex numbers) and columns represent samples.

    Returns:
        A 2D NumPy array representing the correlation matrix.
    """
    num_coeffs, num_samples = data.shape

    # Center the data for each coefficient
    # centered_data = data - np.mean(data, axis=1, keepdims=True)

    # Calculate the covariance matrix
    covariance_matrix = (data @ data.conj().T) / (num_samples - 1)

    # Calculate the standard deviations
    std_devs = np.sqrt(np.diag(covariance_matrix).real)
    outer_std = np.outer(std_devs, std_devs)

    # Avoid division by zero
    if np.any(outer_std == 0):
        return np.zeros_like(covariance_matrix)  # Or handle this case differently

    # Calculate the correlation matrix
    correlation_matrix = (covariance_matrix / outer_std).real  # Take the real part for heatmap

    plt.figure(figsize=(8, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Correlation')
    plt.title('Correlation Matrix of Complex Coefficients (across Samples)')
    plt.xlabel('Coefficient Index')
    plt.ylabel('Coefficient Index')
    plt.tight_layout()
    plt.show()

    return correlation_matrix


