import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt
from src.utils.helpers import *
matplotlib.use('TkAgg')

# this type does not work: <class 'qiskit_aer.backends.compatibility.Statevector'>
# this type the function .expectation value works: <class 'qiskit.quantum_info.states.statevector.Statevector'>
def plot_bloch_sphere(states):
    bloch_x = []
    bloch_y = []
    bloch_z = []

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

