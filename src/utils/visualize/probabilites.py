from matplotlib import pyplot as plt


def plot_probabilities(t_span, state_probs):
    plt.figure(figsize=(10, 6))
    plt.plot(t_span, state_probs[:, 0], label="P(|0⟩)")
    plt.plot(t_span, state_probs[:, 1], label="P(|1⟩)")
    plt.xlabel("Time (ns)")
    plt.ylabel("Probability")
    plt.title("Qubit State Evolution Under Gaussian Pulse")
    plt.legend()
    plt.grid()
    plt.show()

