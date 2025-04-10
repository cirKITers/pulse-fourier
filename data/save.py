import json


# SAVE DATA TEMPLATE
# data_to_save = {
#     "model_name": qm.model_name+"("+str(parameter.shape).replace("(", "").replace(")", "").replace(" ", "")+")",
#     "x": x_.tolist(),  #
#     "fx": [val.tolist() if isinstance(val, np.ndarray) else val for val in fx],
#     "parameters": parameter.tolist(),
# }
#
# save_to(data_to_save, pulse_file)

def new_set(title, file):
    with open(file, "a") as f:
        f.write("\n")
        f.write(title)
        f.write("\n")

def save_to(data, file):
    with open(file, "a") as f:
        json.dump(data, f)
        f.write("\n")


def set_done(file):
    with open(file, "a") as f:
        f.write("#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-")
        f.write("\n")
    print("Saved in "+file)
