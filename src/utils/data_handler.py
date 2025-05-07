import json
import numpy as np

def load(file_path):
    """Loads data from a JSON file and converts fx_set values back to NumPy arrays."""
    try:
        with open(file_path, 'r') as f:
            loaded_data = json.load(f)

        x = loaded_data.get('x')
        fx_set_list = loaded_data.get('fx_set')

        # Convert fx_set lists back to NumPy arrays
        fx_set_arrays = [np.array(fx_values) for fx_values in fx_set_list] if fx_set_list else []

        return x, fx_set_arrays

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None, None

def save(model_name, num_qubits, num_layers, num_samples, start, stop, points, x, fx_set, direc):
    data_to_save = {
        "title": f"{model_name} - {num_qubits} qubits - {num_layers} layers - {num_samples} samples - {start} start - {stop} stop - {points} points ",
        "x": x.tolist() if isinstance(x, np.ndarray) else x,
        "fx_set": [fx_values.tolist() if isinstance(fx_values, np.ndarray) else fx_values for fx_values in fx_set]
    }

    filename = f"{model_name}_{num_qubits}qubits_{num_layers}layers_{num_samples}samples_{start}start_{stop}stop_{points}points.json"
    file_path = "/pfs/data6/home/ka/ka_scc/ka_tc6850/pulse-fourier/results/" + direc + filename       # ../results/ locally

    with open(file_path, 'w') as f:
        json.dump(data_to_save, f, indent=1)  # indent is for better readability

    print(f"Data saved to {file_path}")

