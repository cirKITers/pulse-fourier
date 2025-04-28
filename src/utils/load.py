import csv
import json
import re

import numpy as np


#

def load_set(filename, index):
    """
    Loads 'x' and a list of 'fx' arrays for a specific data set index
    from a plain text file.

    Args:
        filename (str): The path to the text file.
        index (int): The 0-based index of the data set to load.

    Returns:
        tuple: A tuple containing:
               - x (list or None): The 'x' array for the specified data set,
                                    or None if the index is out of bounds
                                    or the format is unexpected.
               - fx_list (list or None): A list of 'fx' arrays for the
                                          specified data set, or None if the
                                          index is out of bounds or the format
                                          is unexpected.
    """
    x_data = None
    fx_list = []
    current_dataset_index = -1
    reading_dataset = False

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if re.match(r".* - \d+ qubits - \d+ layers - \d+ samples", line):
                current_dataset_index += 1
                if current_dataset_index == index:
                    reading_dataset = True
                    x_data = None
                    fx_list = []
                else:
                    reading_dataset = False
            elif reading_dataset and line.startswith('['):
                try:
                    data = eval(line)
                    if x_data is None:
                        x_data = data
                    else:
                        fx_list.append(data)
                except (SyntaxError, NameError):
                    print(f"Error parsing data on line: {line}")
                    return None, None
            elif reading_dataset and line.startswith('#'):
                return x_data, fx_list
            elif current_dataset_index > index:
                # Optimization: If we've passed the desired index, we can stop
                break

    if current_dataset_index < index:
        print(f"Warning: Data set at index {index} not found in the file.")
        return None, None

    # Handle the case where the last dataset matches the index and ends with EOF
    if current_dataset_index == index and x_data is not None:
        return x_data, fx_list

    return None, None

# def load_rows_between(filename, start_index, end_index):
#     """
#     Loads rows from a JSONL file between start_index and end_index (inclusive), storing values in lists.
#
#     Args:
#         filename (str): The path to the JSONL file.
#         start_index (int): The starting row index.
#         end_index (int): The ending row index.
#
#     Returns:
#         dict: A dictionary where keys are attribute names and values are lists containing
#               the corresponding values from the selected rows. Returns None if an error occurs.
#     """
#     try:
#         results = {}
#         row_count = 0
#         attribute_names = None
#
#         with open(filename, "r") as f:
#             for i, line in enumerate(f):
#                 if start_index <= i <= end_index:
#                     data = json.loads(line)
#                     row_count += 1
#
#                     if attribute_names is None:
#                         attribute_names = set(data.keys())
#                         for key in attribute_names:
#                             results[key] = []  # Initialize lists instead of sets
#                     elif set(data.keys()) != attribute_names:
#                         print(f"Error: Row {i} has inconsistent attribute names.")
#                         return None
#
#                     for key, value in data.items():
#                         results[key].append(value)  # Append values to lists
#
#                 elif i > end_index:
#                     break
#
#         if row_count != (end_index - start_index + 1):
#             print(f"Warning: Some rows between {start_index} and {end_index} were not found.")
#
#         return results
#
#     except FileNotFoundError:
#         print(f"Error: File '{filename}' not found.")
#         return None
#     except json.JSONDecodeError as e:
#         print(f"JSON decoding error: {e}")
#         return None
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#         return None


# def load_set_csv(file_name, entry_index):
#     """
#     Loads a specific entry (x and fx set) from a CSV file.
#
#     Args:
#         file_name (str): The name of the CSV file to load from.
#         entry_index (int): The index of the dataset to retrieve (0-based).
#
#     Returns:
#         tuple: A tuple containing:
#                - x (numpy.ndarray or None): The x values, or None if not found.
#                - fx_bundle (list or None): A list of fx value lists for the specified entry,
#                                             or None if not found.
#     """
#     x = None
#     fx_bundle = []
#
#     if entry_index == 0:
#         try:
#             with open(file_name, 'r', newline='') as csvfile:
#                 reader = csv.reader(csvfile)
#                 for i, row in enumerate(reader):
#                     if i == 1 and row and row[0] == 'x':
#                         try:
#                             x = np.array([float(val) for val in row[1:]])
#                         except ValueError:
#                             print(f"Warning: Could not parse x values in dataset {entry_index}.")
#                             return None, None
#                     elif i > 1 and row and row[0].startswith('fx_'):
#                         try:
#                             fx = [float(val) for val in row[1:]]
#                             fx_bundle.append(fx)
#                         except ValueError:
#                             print(f"Warning: Could not parse fx values in dataset {entry_index}.")
#                             return None, None
#                     elif row and row[0] == '#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-':
#                         break  # Stop reading after the first separator for index 0
#             if x is not None:
#                 return x, fx_bundle
#             else:
#                 print(f"Error: Dataset at index {entry_index} not found or incomplete.")
#                 return None, None
#         except FileNotFoundError:
#             print(f"Error: File not found: {file_name}")
#             return None, None
#     else:
#         # Logic for subsequent entries (index > 0)
#         dataset_count = -1
#         found_entry = False
#         try:
#             with open(file_name, 'r', newline='') as csvfile:
#                 reader = csv.reader(csvfile)
#                 reading_data = False
#                 row_index_in_dataset = 0
#                 for row in reader:
#                     if row and row[0] == '#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-':
#                         dataset_count += 1
#                         reading_data = False
#                         row_index_in_dataset = 0
#                         x = None
#                         fx_bundle = []
#                         if dataset_count == entry_index:
#                             found_entry = True
#                             reading_data = True
#                         continue
#
#                     if dataset_count == entry_index and reading_data:
#                         row_index_in_dataset += 1
#                         if row_index_in_dataset == 2 and row and row[0] == 'x':
#                             try:
#                                 x = np.array([float(val) for val in row[1:]])
#                             except ValueError:
#                                 print(f"Warning: Could not parse x values in dataset {entry_index}.")
#                                 return None, None
#                         elif row_index_in_dataset > 2 and row and row[0].startswith('fx_'):
#                             try:
#                                 fx = [float(val) for val in row[1:]]
#                                 fx_bundle.append(fx)
#                             except ValueError:
#                                 print(f"Warning: Could not parse fx values in dataset {entry_index}.")
#                                 return None, None
#                 if found_entry and x is not None:
#                     return x, fx_bundle
#                 else:
#                     print(f"Error: Dataset at index {entry_index} not found or incomplete.")
#                     return None, None
#         except FileNotFoundError:
#             print(f"Error: File not found: {file_name}")
#             return None, None
#
#
# def extract_data_set(file_path, i):
#     """
#     Extracts the ith data set (x array and all fx arrays) from a CSV file.
#
#     Args:
#         file_path (str): The path to the CSV file.
#         i (int): The index of the data set to extract (0-based).
#
#     Returns:
#         tuple: A tuple containing:
#             - x (numpy.ndarray or None): The x array, or None if not found.
#             - fx_set (list or None): A list of numpy.ndarrays representing the fx arrays,
#                                      or None if not found.
#     """
#     x = None
#     fx_set = []
#     start_index = i * 2  # Each set starts after the title and x row
#     end_separator = '#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-'
#     current_set = 0
#     reading_set = False
#
#     with open(file_path, 'r') as csvfile:
#         reader = csv.reader(csvfile)
#         row_index = 0
#         for row in reader:
#             if row and row[0] == end_separator:
#                 reading_set = False
#                 if current_set == i:
#                     break
#                 current_set += 1
#                 x = None
#                 fx_set = []
#             elif current_set == i:
#                 if row_index == start_index + 1:  # This should be the x array row
#                     if len(row) > 1 and row[0] == 'x':
#                         try:
#                             x = np.array([float(val) for val in row[1:]])
#                         except ValueError:
#                             print(f"Warning: Could not convert x values to float array: {row[1:]}")
#                             x = None
#                     else:
#                         print(f"Warning: Expected 'x' label in x array row: {row}")
#                         x = None
#                 elif row_index > start_index + 1 and row and row[0] != end_separator:
#                     if len(row) > 1 and row[0].startswith('fx_'):
#                         try:
#                             fx_set.append(np.array([float(val) for val in row[1:]]))
#                         except ValueError:
#                             print(f"Warning: Could not convert fx values to float array: {row[1:]}")
#                     elif row:
#                         try:
#                             fx_set.append(np.array([float(val) for val in row]))
#                         except ValueError:
#                             print(f"Warning: Could not convert fx values to float array: {row}")
#             elif row and row[0] != end_separator and current_set < i:
#                 # Skip rows until the start of the desired set
#                 pass
#
#             row_index += 1
#
#     return x, fx_set

