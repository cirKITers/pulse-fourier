from matplotlib import pyplot as plt

# https://qiskit-community.github.io/qiskit-dynamics/tutorials/qiskit_pulse.html
def plot_populations(sol, T):
    pop0 = [psi.probabilities()[0] for psi in sol.y]
    pop1 = [psi.probabilities()[1] for psi in sol.y]

    fig = plt.figure(figsize=(8, 5))
    plt.plot(sol.t, pop0, lw=3, label="Population in |0>")
    plt.plot(sol.t, pop1, lw=3, label="Population in |1>")
    plt.xlabel("Time (ns)")
    plt.ylabel("Population")
    plt.legend(frameon=False)
    plt.ylim([0, 1.05])
    plt.xlim([0, 2*T])
    plt.vlines(T, 0, 1.05, "k", linestyle="dashed")
    plt.show()


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

