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

def load_and_combine_fx_sets(directory, model_name, num_qubits, num_layers, num_samples, start, stop, points, seed_start, seed_stop):
    """
    Loads fx_set data from multiple JSON files and combines them into a single NumPy array.

    Args:
        directory (str): The directory containing the JSON files.
        model_name
        num_qubits (int): The number of qubits in the circuit.
        num_layers (int): The number of layers in the circuit.
        num_samples (int): The number of samples used.
        start (float): The start value of the parameter range.
        stop (float): The stop value of the parameter range.
        points (int): The number of points in the parameter range.
        seed_start
        seed_stop

    Returns:
        numpy.ndarray: A single NumPy array containing all the combined fx_set data,
                       or None if no files are found or an error occurs.
    """
    all_fx_sets = []
    for seed in range(seed_start, seed_stop):
        file_name = f"{model_name}_{num_qubits}qubits_{num_layers}layers_{num_samples}samples_{start}start_{stop}stop_{points}points_{seed}seed.json"
        file_path = directory + file_name
        # print(file_path)

        try:
            with open(file_path, 'r') as f:
                loaded_data = json.load(f)
                fx_set_list = loaded_data.get('fx_set')
                if fx_set_list:
                    # Extend the list with the NumPy arrays from the current file
                    all_fx_sets.extend([np.array(fx_values) for fx_values in fx_set_list])
        except FileNotFoundError:
            print(f"Warning: File not found at {file_path}")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}")
            return None

    if all_fx_sets:
        # Stack all the NumPy arrays vertically to create a single array
        combined_fx_set = np.vstack(all_fx_sets)
        return combined_fx_set
    else:
        print("No fx_set data found in the specified files.")
        return None

def save(model_name, num_qubits, num_layers, num_samples, start, stop, points, seed, x, fx_set, direc, cluster):
    data_to_save = {
        "title": f"{model_name} - {num_qubits} qubits - {num_layers} layers - {num_samples} samples - {start} start - {stop} stop - {points} points - {seed} seed",
        "x": x.tolist() if isinstance(x, np.ndarray) else x,
        "fx_set": [fx_values.tolist() if isinstance(fx_values, np.ndarray) else fx_values for fx_values in fx_set]
    }

    filename = f"{model_name}_{num_qubits}qubits_{num_layers}layers_{num_samples}samples_{start}start_{stop}stop_{points}points_{seed}seed.json"

    if cluster:
        highdir = "/pfs/data6/home/ka/ka_scc/ka_tc6850/pulse-fourier/results/"
    else:
        highdir = "../results/"
    file_path = highdir + direc + filename

    with open(file_path, 'w') as f:
        json.dump(data_to_save, f, indent=1)  # indent is for better readability

    print(f"Data saved to {file_path}")

