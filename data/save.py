import json

def save(data, file):
    with open(file, "a") as f:
        json.dump(data, f)
        f.write("\n")
    print("Saved in "+file)

