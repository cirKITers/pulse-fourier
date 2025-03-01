import matplotlib.pyplot as plt

def plot_fx(x_data, y_data, title):
    plt.figure(figsize=(8, 6))
    plt.plot(x_data, y_data, label=title, color='b')
    plt.title(title)
    plt.xlabel("Input x")
    plt.ylabel("Predicted f(x)")
    plt.grid(True)
    plt.legend()
    plt.show()


