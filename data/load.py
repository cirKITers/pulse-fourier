import json
import numpy as np

def load_data_from_jsonl(filename, index):
    try:
        with open(filename, "r") as f:
            for i, line in enumerate(f):
                if i == index:
                    data = json.loads(line)
                    x = np.array(data["x"])
                    fx = [np.array(val) if isinstance(val, list) else val for val in data["fx"]]
                    parameters = np.array(data["parameters"])
                    return x, fx, parameters, parameters.shape
                elif i > index:
                    return None
            return None
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return None
    except KeyError as e:
        print(f"KeyError: Missing key in JSON data: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def get_row_by_index_lazy(filename, index):
    """Reads a specific row lazily from a JSON Lines file."""
    try:
        with open(filename, "r") as f:
            for i, line in enumerate(f):
                if i == index:
                    return json.loads(line)
                elif i > index:
                    return None
            return None
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return None


