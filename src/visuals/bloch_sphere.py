import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector, Pauli

from utils.definitions import *
# from matplotlib import pyplot as plt
from utils.helpers import *
matplotlib.use('Agg')             # 'TkAgg' for plotting locally, 'Agg' for cluster (no plotting)

def bloch_sphere_multiqubit_trajectory(states, qubits_to_plot=None, laboratory_frame=False):
    """
    Plots the Bloch sphere trajectories for specified qubits in a multi-qubit system.

    Args:
        states (list[Statevector]): List of Statevectors representing the evolution of the multi-qubit system.
        qubits_to_plot (list[int], optional): List of qubit indices to plot. If None, plots all qubits.
        laboratory_frame (bool, optional): If True, applies a static unitary to the states before plotting.
    """

    if not all(isinstance(state, Statevector) for state in states):
        raise TypeError("All elements in 'states' must be Statevector objects.")

    num_qubits = states[0].num_qubits

    if qubits_to_plot is None:
        qubits_to_plot = list(range(num_qubits))

    if not all(isinstance(q, int) and 0 <= q < num_qubits for q in qubits_to_plot):
        raise ValueError("Invalid qubit indices in 'qubits_to_plot'.")

    num_plots = len(qubits_to_plot)

    fig = plt.figure(figsize=(8 * num_plots, 6))
    fig.suptitle("Bloch Sphere Trajectories for Selected Qubits", fontsize=16)

    for i, qubit_index in enumerate(qubits_to_plot):
        bloch_x = []
        bloch_y = []
        bloch_z = []

        for state in states:
            # Construct Pauli operators for the selected qubit
            pauli_x = Pauli('I' * qubit_index + 'X' + 'I' * (num_qubits - qubit_index - 1))
            pauli_y = Pauli('I' * qubit_index + 'Y' + 'I' * (num_qubits - qubit_index - 1))
            pauli_z = Pauli('I' * qubit_index + 'Z' + 'I' * (num_qubits - qubit_index - 1))

            bloch_x.append(np.real(state.expectation_value(pauli_x)))
            bloch_y.append(np.real(state.expectation_value(pauli_y)))
            bloch_z.append(np.real(state.expectation_value(pauli_z)))

        ax = fig.add_subplot(1, num_plots, i + 1, projection='3d')

        # Wireframe sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(x, y, z, color='gray', alpha=0.2)

        ax.plot(bloch_x, bloch_y, bloch_z, 'b-', label='Trajectory')
        ax.scatter([bloch_x[0]], [bloch_y[0]], [bloch_z[0]], color='g', label='Initial')
        ax.scatter([bloch_x[-1]], [bloch_y[-1]], [bloch_z[-1]], color='r', label='Final')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Qubit {qubit_index}')
        ax.legend()
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.show()


# this type does not work: <class 'qiskit_aer.backends.compatibility.Statevector'>
# this type the function .expectation value works: <class 'qiskit.quantum_info.states.statevector.Statevector'>
def bloch_sphere_trajectory(states, laboratory_frame=False):
    bloch_x = []
    bloch_y = []
    bloch_z = []

    # test with Hadamard
    # if laboratory_frame:
    #     trajectory_lab = []
    #     for state in states:
    #         trajectory_lab.append(Statevector(U_static @ state.data))

    for state in states:

        # check correct type
        if not isinstance(state, Statevector):
            raise TypeError(f"Invalid state type: {type(state)}. Expected: {Statevector}.")

        # check if normalized
        norm = np.sum(np.abs(state.data) ** 2)
        if not np.isclose(norm, 1.0):
            raise ValueError(f"Invalid quantum state: Norm = {norm}. Expected: 1.")

        # Bloch coordinates: <sigma_x>, <sigma_y>, <sigma_z>
        bloch_x.append(np.real(state.expectation_value(SIGMA_X)))
        bloch_y.append(np.real(state.expectation_value(SIGMA_Y)))
        bloch_z.append(np.real(state.expectation_value(SIGMA_Z)))

    # final_state = states[-1]
    # print("Final state:", final_state)

    # Plot Bloch sphere trajectory
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle("Keep in mind: The trajectories are usually curves", fontsize=12, color="red")

    ax = fig.add_subplot(111, projection='3d')

    # wireframe
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.2)

    ax.plot(bloch_x, bloch_y, bloch_z, 'b-', label='Pulse trajectory')
    ax.scatter([bloch_x[0]], [bloch_y[0]], [bloch_z[0]], color='g', label=f'Initial State ({states[0].data})')
    ax.scatter([bloch_x[-1]], [bloch_y[-1]], [bloch_z[-1]], color='r', label=f'Final State ({states[-1].data})')

    # axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Bloch Sphere Trajectory')
    ax.legend()

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    plt.show()

    ax.view_init(elev=20, azim=45)

    return bloch_x, bloch_y, bloch_z

