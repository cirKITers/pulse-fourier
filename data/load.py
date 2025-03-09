import json

def load_rows_between(filename, start_index, end_index):
    """
    Loads rows from a JSONL file between start_index and end_index (inclusive), storing values in lists.

    Args:
        filename (str): The path to the JSONL file.
        start_index (int): The starting row index.
        end_index (int): The ending row index.

    Returns:
        dict: A dictionary where keys are attribute names and values are lists containing
              the corresponding values from the selected rows. Returns None if an error occurs.
    """
    try:
        results = {}
        row_count = 0
        attribute_names = None

        with open(filename, "r") as f:
            for i, line in enumerate(f):
                if start_index <= i <= end_index:
                    data = json.loads(line)
                    row_count += 1

                    if attribute_names is None:
                        attribute_names = set(data.keys())
                        for key in attribute_names:
                            results[key] = []  # Initialize lists instead of sets
                    elif set(data.keys()) != attribute_names:
                        print(f"Error: Row {i} has inconsistent attribute names.")
                        return None

                    for key, value in data.items():
                        results[key].append(value)  # Append values to lists

                elif i > end_index:
                    break

        if row_count != (end_index - start_index + 1):
            print(f"Warning: Some rows between {start_index} and {end_index} were not found.")

        return results

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
