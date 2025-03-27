import json

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
