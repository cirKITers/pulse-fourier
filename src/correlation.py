import colorsys
import plotly.graph_objects as go
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle
import os
from plotly.subplots import make_subplots

from constants import *


def plot_interactive_3d_heatmaps(parameters, coeffs_all, output_dir=correlation_dir):
    """
    Generate interactive 3D scatter plots for each coefficient, with points at (param1, param2, param3)
    colored by complex coefficient values (phase as hue, magnitude as brightness), including a legend.

    Parameters:
    - parameters (np.ndarray): Array of shape (n_samples, 1, 3, 1) containing parameter values.
    - coeffs_all (np.ndarray): Array of shape (n_samples, 5, 2) containing real and imag parts of coefficients.
    - output_dir (str): Directory to save the output HTML files.

    Returns:
    - None: Saves five interactive HTML files in output_dir, one per coefficient.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Reshape parameters from (n_samples, 1, 3, 1) to (n_samples, 3)
    n_samples = parameters.shape[0]
    parameters = parameters.reshape(n_samples, 3)  # Shape: (n_samples, 3)

    # Convert coeffs_all from (n_samples, 5, 2) to (n_samples, 5) complex numbers
    coeffs_complex = np.complex128(coeffs_all[..., 0] + 1j * coeffs_all[..., 1])  # Shape: (n_samples, 5)

    # Number of coefficients (should be 5 based on input shape)
    num_coeffs = coeffs_complex.shape[1]

    # Extract parameters for plotting
    param1 = parameters[:, 0]
    param2 = parameters[:, 1]
    param3 = parameters[:, 2]

    # Generate a plot for each coefficient
    for coeff_idx in range(num_coeffs):
        # Get the complex coefficients for this index
        coeffs = coeffs_complex[:, coeff_idx]

        # Compute phase (angle) and magnitude
        phases = np.angle(coeffs)  # In radians, [-pi, pi]
        magnitudes = np.abs(coeffs)

        # Normalize phase to [0, 1] for hue (0 to 2pi range mapped to 0 to 1)
        normalized_phases = (phases + np.pi) % (2 * np.pi) / (2 * np.pi)

        # Normalize magnitude to [0, 1] based on the maximum for this coefficient
        max_magnitude = np.max(magnitudes)
        if max_magnitude == 0:  # Avoid division by zero
            max_magnitude = 1.0
        normalized_magnitudes = magnitudes / max_magnitude

        # Generate colors: phase to hue, magnitude to brightness
        colors = []
        for phase, mag in zip(normalized_phases, normalized_magnitudes):
            rgb = colorsys.hsv_to_rgb(phase, 1.0, mag)
            rgb_scaled = tuple(int(val * 255) for val in rgb)
            colors.append(f'rgb{rgb_scaled}')

        # Create hover text for each point
        hover_text = [f'Param1: {p1:.2f}<br>Param2: {p2:.2f}<br>Param3: {p3:.2f}<br>Coeff: {c.real:.2f} + {c.imag:.2f}i'
                      for p1, p2, p3, c in zip(param1, param2, param3, coeffs)]

        # Define representative complex numbers for the legend
        legend_complex = [
            1.0 + 0j,  # Magnitude 1, phase 0 (red)
            0.5 * np.exp(1j * np.pi / 2),  # Magnitude 0.5, phase π/2 (greenish)
            1.0 * np.exp(1j * np.pi),  # Magnitude 1, phase π (cyan)
            0.5 * np.exp(1j * 3 * np.pi / 2),  # Magnitude 0.5, phase 3π/2 (blueish)
            0.0 + 0j  # Magnitude 0 (black)
        ]

        # Compute colors for the legend using the same max_magnitude
        legend_phases = np.angle(legend_complex)
        legend_magnitudes = np.abs(legend_complex)
        legend_normalized_phases = (legend_phases + np.pi) % (2 * np.pi) / (2 * np.pi)
        legend_normalized_magnitudes = legend_magnitudes / max_magnitude
        legend_colors = [f'rgb{tuple(int(val * 255) for val in colorsys.hsv_to_rgb(p, 1.0, m))}'
                         for p, m in zip(legend_normalized_phases, legend_normalized_magnitudes)]

        # Create legend labels
        legend_labels = [f'{c.real:.1f} + {c.imag:.1f}i' for c in legend_complex]

        # Create a subplot layout with 3D plot and 2D legend side by side
        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}]],
                            column_widths=[0.7, 0.3])

        # Add the 3D scatter plot
        fig.add_trace(
            go.Scatter3d(
                x=param1,
                y=param2,
                z=param3,
                mode='markers',
                marker=dict(size=5, color=colors, opacity=0.8),
                hoverinfo='text',
                text=hover_text
            ),
            row=1, col=1
        )

        # Add the 2D scatter legend
        fig.add_trace(
            go.Scatter(
                x=[0] * len(legend_complex),  # Dummy x values
                y=list(range(len(legend_complex))),  # Y positions for legend items
                mode='markers+text',
                marker=dict(size=15, color=legend_colors),
                text=legend_labels,
                textposition='middle right',
                hoverinfo='none'
            ),
            row=1, col=2
        )

        # Update layout for both subplots
        fig.update_layout(
            title=f'3D Heatmap for Coefficient {coeff_idx + 1}',
            scene=dict(
                xaxis_title='Parameter 1',
                yaxis_title='Parameter 2',
                zaxis_title='Parameter 3',
                xaxis=dict(backgroundcolor="white", gridcolor="lightgrey"),
                yaxis=dict(backgroundcolor="white", gridcolor="lightgrey"),
                zaxis=dict(backgroundcolor="white", gridcolor="lightgrey")
            ),
            width=1000,  # Wider to accommodate legend
            height=600,
            margin=dict(l=0, r=0, b=0, t=40),
            showlegend=False,  # Disable default legend
            # Customize the 2D legend subplot
            yaxis2=dict(
                showticklabels=False,
                range=[-1, len(legend_complex)],
                title='Complex Values'
            ),
            xaxis2=dict(
                showticklabels=False,
                range=[-0.5, 0.5]
            )
        )

        # Save as HTML file
        output_file = os.path.join(output_dir, f'3d_heatmap_coeff{coeff_idx + 1}.html')
        fig.write_html(output_file)
        print(f'Saved interactive plot for Coefficient {coeff_idx + 1} to {output_file}')


# -#-#-#-#-#- OLD:

def plot_interactive_3d_heatmaps2(parameters, coeffs_all, output_dir=correlation_dir):
    """
    Generate interactive 3D scatter plots for each coefficient, with points at (param1, param2, param3)
    colored by complex coefficient values (phase as hue, magnitude as brightness).

    Parameters:
    - parameters (np.ndarray): Array of shape (n_samples, 1, 3, 1) containing parameter values.
    - coeffs_all (np.ndarray): Array of shape (n_samples, 5, 2) containing real and imag parts of coefficients.
    - output_dir (str): Directory to save the output HTML files.

    Returns:
    - None: Saves five interactive HTML files in output_dir, one per coefficient.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Reshape parameters from (n_samples, 1, 3, 1) to (n_samples, 3)
    n_samples = parameters.shape[0]
    parameters = parameters.reshape(n_samples, 3)  # Shape: (n_samples, 3)

    # Convert coeffs_all from (n_samples, 5, 2) to (n_samples, 5) complex numbers
    coeffs_complex = np.complex128(coeffs_all[..., 0] + 1j * coeffs_all[..., 1])  # Shape: (n_samples, 5)

    # Number of coefficients (should be 5 based on input shape)
    num_coeffs = coeffs_complex.shape[1]

    # Extract parameters for plotting
    param1 = parameters[:, 0]
    param2 = parameters[:, 1]
    param3 = parameters[:, 2]

    # Generate a plot for each coefficient
    for coeff_idx in range(num_coeffs):
        # Get the complex coefficients for this index
        coeffs = coeffs_complex[:, coeff_idx]

        # Compute phase (angle) and magnitude
        phases = np.angle(coeffs)  # In radians, [-pi, pi]
        magnitudes = np.abs(coeffs)

        # Normalize phase to [0, 1] for hue (0 to 2pi range mapped to 0 to 1)
        normalized_phases = (phases + np.pi) % (2 * np.pi) / (2 * np.pi)

        # Normalize magnitude to [0, 1] based on the maximum for this coefficient
        max_magnitude = np.max(magnitudes)
        if max_magnitude == 0:  # Avoid division by zero
            max_magnitude = 1.0
        normalized_magnitudes = magnitudes / max_magnitude

        # Generate colors: phase to hue, magnitude to brightness
        colors = []
        for phase, mag in zip(normalized_phases, normalized_magnitudes):
            # Convert HSV (hue, saturation=1, value=magnitude) to RGB
            rgb = colorsys.hsv_to_rgb(phase, 1.0, mag)
            # Convert RGB from [0, 1] to [0, 255] and format as string
            rgb_scaled = tuple(int(val * 255) for val in rgb)
            colors.append(f'rgb{rgb_scaled}')

        # Create hover text for each point
        hover_text = [f'Param1: {p1:.2f}<br>Param2: {p2:.2f}<br>Param3: {p3:.2f}<br>Coeff: {c.real:.2f} + {c.imag:.2f}i'
                      for p1, p2, p3, c in zip(param1, param2, param3, coeffs)]

        # Define representative complex numbers for the legend (e.g., different phases and magnitudes)
        legend_complex = [
            1.0 + 0j,  # Magnitude 1, phase 0 (red)
            0.5 * np.exp(1j * np.pi / 2),  # Magnitude 0.5, phase π/2 (greenish)
            1.0 * np.exp(1j * np.pi),  # Magnitude 1, phase π (cyan)
            0.5 * np.exp(1j * 3 * np.pi / 2),  # Magnitude 0.5, phase 3π/2 (blueish)
            0.0 + 0j  # Magnitude 0 (black)
        ]

        # Compute colors for the legend
        legend_phases = np.angle(legend_complex)
        legend_magnitudes = np.abs(legend_complex)
        max_magnitude = np.max(magnitudes) if np.max(magnitudes) > 0 else 1.0  # Use same max as main plot
        legend_normalized_phases = (legend_phases + np.pi) % (2 * np.pi) / (2 * np.pi)
        legend_normalized_magnitudes = legend_magnitudes / max_magnitude
        legend_colors = [f'rgb{tuple(int(val * 255) for val in colorsys.hsv_to_rgb(p, 1.0, m))}'
                         for p, m in zip(legend_normalized_phases, legend_normalized_magnitudes)]

        # Create legend labels
        legend_labels = [f'{c.real:.1f} + {c.imag:.1f}i' for c in legend_complex]

        # Create a subplot layout with 3D plot and 2D legend side by side

        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}]],
                            column_widths=[0.7, 0.3])

        # Create the interactive 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=param1,
            y=param2,
            z=param3,
            mode='markers',
            marker=dict(
                size=5,  # Adjust size as needed
                color=colors,  # RGB strings for each point
                opacity=0.8
            ),
            hoverinfo='text',
            text=hover_text
        )])

        fig.add_trace(
            go.Scatter3d(
                x=param1,
                y=param2,
                z=param3,
                mode='markers',
                marker=dict(size=5, color=colors, opacity=0.8),
                hoverinfo='text',
                text=hover_text
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=[0] * len(legend_complex),  # Dummy x values
                y=list(range(len(legend_complex))),  # Y positions for legend items
                mode='markers+text',
                marker=dict(size=15, color=legend_colors),
                text=legend_labels,
                textposition='middle right',
                hoverinfo='none'
            ),
            row=1, col=2
        )

        # Update layout with titles and labels
        fig.update_layout(
            title=f'3D Heatmap for Coefficient {coeff_idx + 1}',
            scene=dict(
                xaxis_title='Parameter 1',
                yaxis_title='Parameter 2',
                zaxis_title='Parameter 3',
                xaxis=dict(backgroundcolor="white", gridcolor="lightgrey"),
                yaxis=dict(backgroundcolor="white", gridcolor="lightgrey"),
                zaxis=dict(backgroundcolor="white", gridcolor="lightgrey")
            ),
            width=1000,  # Wider to accommodate legend
            height=600,
            margin=dict(l=0, r=0, b=0, t=40),
            showlegend=False,  # Disable default legend
            # Customize the 2D legend subplot
            yaxis2=dict(
                showticklabels=False,
                range=[-1, len(legend_complex)],
                title='Complex Values'
            ),
            xaxis2=dict(
                showticklabels=False,
                range=[-0.5, 0.5]
            )
        )

        # Save as HTML file
        output_file = os.path.join(output_dir, f'3d_heatmap_coeff{coeff_idx + 1}.html')
        fig.write_html(output_file)
        print(f'Saved interactive plot for Coefficient {coeff_idx + 1} to {output_file}')

def plot_3d_heatmap(parameters, coeffs_all, output_dir=correlation_dir):
    """
    Generate a single 3D scatter plot (heatmap) with parameter, real part, and imaginary part axes.

    Parameters:
    - parameters (np.ndarray): Array of shape (n_samples, 1, 3, 1) containing parameter values.
    - coeffs_all (np.ndarray): Array of shape (n_samples, 5, 2) containing real and imag parts of coefficients.
    - output_dir (str): Directory to save the output plot file.

    Returns:
    - None: Saves the plot to a file in output_dir.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Reshape and extract data
    n_samples = parameters.shape[0]
    parameters = parameters.reshape(n_samples, 3)  # Shape: (75, 3)
    coeffs_complex = np.complex128(coeffs_all[..., 0] + 1j * coeffs_all[..., 1])  # Shape: (75, 5)

    # Select first parameter and first coefficient for simplicity
    param_values = parameters[:, 0]  # First parameter
    real_parts = coeffs_complex[:, 0].real  # Real part of first coefficient
    imag_parts = coeffs_complex[:, 0].imag  # Imaginary part of first coefficient

    # Create the 3D scatter plot
    sns.set_style('whitegrid')  # Professional style
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with color based on parameter values
    sc = ax.scatter(param_values, real_parts, imag_parts, c=param_values, cmap='viridis', alpha=0.8, s=50)

    # Customize axes and labels
    ax.set_xlabel('Parameter 1', fontsize=12, labelpad=10)
    ax.set_ylabel('Real Part', fontsize=12, labelpad=10)
    ax.set_zlabel('Imaginary Part', fontsize=12, labelpad=10)
    ax.set_title('3D Heatmap: Parameter 1 vs. Real and Imaginary Parts of Coefficient 1', fontsize=16, pad=20)

    # Add color bar
    cbar = fig.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label('Parameter 1 Value', fontsize=12)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3d_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
def correlation(parameters, coeffs_all, output_dir=correlation_dir_detailed):
    """
    Analyze the dependency of complex coefficients on parameters from variational quantum circuits.

    Parameters:
    - jsonl_file_path (str): Path to the JSONL file containing parameter and coefficient data.
    - output_dir (str): Directory to save output files (plots and results).

    Returns:
    - None: Saves results to files in output_dir.
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # # Step 1: Parse JSONL data
    # parameters = []
    # coeffs_all = []
    # with open(jsonl_file_path, 'r') as f:
    #     for line in f:
    #         data = json.loads(line)
    #         # Flatten the nested parameter structure: [[[-1.62]], [[1.32]], [[-1.22]]] -> [-1.62, 1.32, -1.22]
    #         param = [p[0][0] for p in data["parameter"]]
    #         parameters.append(param)
    #         # Extract coefficients as complex numbers
    #         coeffs = [complex(real, imag) for real, imag in data["coeffs_all"]]
    #         coeffs_all.append(coeffs)

    # Convert to numpy arrays for easier manipulation
    n_samples = parameters.shape[0]
    parameters = parameters.reshape(n_samples, 3)  # Remove singleton dimensions

    # Coeffs_all: from (75, 5, 2) to (75, 5) as complex numbers
    coeffs_complex = np.complex128(coeffs_all[..., 0] + 1j * coeffs_all[..., 1])  # Shape: (75, 5)

    # Assign to variables for consistency with original code
    parameters = np.array(parameters)  # Shape: (n_samples, 3)
    coeffs_all = np.array(coeffs_complex)  # Shape: (n_samples, 5)
    n_params = parameters.shape[1]  # Should be 3
    n_coeffs = coeffs_all.shape[1]  # Should be 5

    # Step 2: Prepare DataFrame for analysis
    df = pd.DataFrame(parameters, columns=[f'param{i + 1}' for i in range(n_params)])
    for i in range(n_coeffs):
        df[f'real_{i + 1}'] = coeffs_all[:, i].real
        df[f'imag_{i + 1}'] = coeffs_all[:, i].imag
        df[f'magnitude_{i + 1}'] = np.abs(coeffs_all[:, i])
        df[f'phase_{i + 1}'] = np.angle(coeffs_all[:, i])

    # Standardize parameters for consistent regression
    scaler = StandardScaler()
    param_scaled = scaler.fit_transform(parameters)
    df[[f'param{i + 1}' for i in range(n_params)]] = param_scaled

    # Step 3: Correlation Analysis
    corr_matrix = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', center=0)
    plt.title('Correlation Matrix: Parameters vs. Coefficients')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()

    # Step 4: Visualization - Marginal Plots
    for param_idx in range(n_params):
        param_name = f'param{param_idx + 1}'
        for coeff_idx in range(n_coeffs):
            coeff_name = f'coeff{coeff_idx + 1}'
            # Real part
            plt.figure(figsize=(6, 4))
            plt.scatter(df[param_name], df[f'real_{coeff_idx + 1}'], alpha=0.5)
            plt.xlabel(param_name)
            plt.ylabel(f'Real Part of {coeff_name}')
            plt.title(f'Real Part of {coeff_name} vs {param_name}')
            plt.savefig(os.path.join(output_dir, f'real_{coeff_name}_vs_{param_name}.png'))
            plt.close()

            # Imaginary part
            plt.figure(figsize=(6, 4))
            plt.scatter(df[param_name], df[f'imag_{coeff_idx + 1}'], alpha=0.5)
            plt.xlabel(param_name)
            plt.ylabel(f'Imaginary Part of {coeff_name}')
            plt.title(f'Imaginary Part of {coeff_name} vs {param_name}')
            plt.savefig(os.path.join(output_dir, f'imag_{coeff_name}_vs_{param_name}.png'))
            plt.close()

            # Magnitude
            plt.figure(figsize=(6, 4))
            plt.scatter(df[param_name], df[f'magnitude_{coeff_idx + 1}'], alpha=0.5)
            plt.xlabel(param_name)
            plt.ylabel(f'Magnitude of {coeff_name}')
            plt.title(f'Magnitude of {coeff_name} vs {param_name}')
            plt.savefig(os.path.join(output_dir, f'magnitude_{coeff_name}_vs_{param_name}.png'))
            plt.close()

            # Phase
            plt.figure(figsize=(6, 4))
            plt.scatter(df[param_name], df[f'phase_{coeff_idx + 1}'], alpha=0.5)
            plt.xlabel(param_name)
            plt.ylabel(f'Phase of {coeff_name}')
            plt.title(f'Phase of {coeff_name} vs {param_name}')
            plt.savefig(os.path.join(output_dir, f'phase_{coeff_name}_vs_{param_name}.png'))
            plt.close()

    # Step 5: Linear Regression Analysis
    regression_results = {}
    model = LinearRegression()
    X = df[[f'param{i + 1}' for i in range(n_params)]].values

    for coeff_idx in range(n_coeffs):
        coeff_name = f'coeff{coeff_idx + 1}'
        for part in ['real', 'imag', 'magnitude', 'phase']:
            y = df[f'{part}_{coeff_idx + 1}'].values
            model.fit(X, y)
            regression_results[f'{coeff_name}_{part}'] = {
                'coefficients': model.coef_.tolist(),
                'intercept': model.intercept_,
                'r_squared': model.score(X, y)
            }

    # Save regression results
    with open(os.path.join(output_dir, 'analysis_results.pickle'), 'wb') as f:
        pickle.dump(regression_results, f)

    # Print summary of regression results
    print("Linear Regression Summary:")
    for key, result in regression_results.items():
        print(f"{key}:")
        print(f"  Coefficients: {result['coefficients']}")
        print(f"  Intercept: {result['intercept']}")
        print(f"  R-squared: {result['r_squared']:.4f}\n")

