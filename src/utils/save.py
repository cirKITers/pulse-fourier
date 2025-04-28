import csv
import json
import os


def save_set(model_name, num_qubits, num_layers, num_samples, x, fx_set, file_name):
    title = f"{model_name} - {num_qubits} qubits - {num_layers} layers - {num_samples} samples"
    with open(file_name, "a") as f:
        f.write("\n")
        f.write(title)
        f.write("\n")
        # Save x values in the first line
        f.write(json.dumps(list(x)))
        f.write("\n")
        # Save fx values
        for fx in fx_set:
            f.write(json.dumps(list(fx)))
            f.write("\n")
        f.write("#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-")
        f.write("\n")
        print(f"Saved in {file_name}")


# def save_set_csv(model_name, num_qubits, num_layers, num_samples, x, fx_set, file_name):
#     title = f"{model_name} - {num_qubits} qubits - {num_layers} layers - {num_samples} samples"
#     with open(file_name, "a", newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow([title])
#         writer.writerow(list(x))
#         for i, fx in enumerate(fx_set):
#             writer.writerow(list(fx))
#         writer.writerow(["#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-"])
#         print(f"Saved in {file_name}")


# def save_to(data, file):
#     with open(file, "a") as f:
#         json.dump(data, f)
#         f.write("\n")
