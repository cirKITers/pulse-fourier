import jax
import jax.numpy as jnp
from qiskit_dynamics import Solver, Signal
import matplotlib.pyplot as plt
import matplotlib
from qiskit_dynamics import DiscreteSignal
from qiskit_dynamics.signals import Convolution

from utils.helpers import *
from utils.visualize.bloch_sphere import *
matplotlib.use('TkAgg')  # Force TkAgg backend
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

def h_pulse(drive_strength, sigma, plot, bool_blochsphere=False):

    final_probs = np.zeros(2)

    omega = 5.0                         # v
    drive_strength = drive_strength     # r
    sigma = sigma                       # sigma

    H_static = static_hamiltonian(omega=omega)
    H_drive = drive_hamiltonian(drive_strength=drive_strength)

    ham_solver = Solver(
        static_hamiltonian=H_static,
        hamiltonian_operators=[H_drive],
        rotating_frame=H_static
    )

    duration = 120  # Number of time steps (samples)
    amp = 1.0      # Amplitude, height gaussian bell at peak, default Max
    _dt = 0.1       # Time step in ns
    t_span = np.linspace(0, duration * _dt, duration + 1)   # Tells solver when to check the qubits state

    def gaussian_envelope(t):
        center = duration * _dt / 2
        return amp * jnp.exp(-((t - center) ** 2) / (2 * sigma ** 2))

    # from Tutorial, what is it for?
    def gaussian_conv(t):
        return Convolution(2. * _dt / np.sqrt(2. * np.pi * sigma ** 2) * np.exp(-t ** 2 / (2 * sigma ** 2)))

    gaussian_signal = Signal(
        envelope=gaussian_envelope,
        carrier_freq=omega,
        phase=np.pi/2
    )

    result = ham_solver.solve(
        t_span=t_span,
        y0=INIT_STATE,
        method='jax_odeint',
        signals=[gaussian_signal]
    )

    if plot:
        state_probs = np.abs(result.y) ** 2
        plt.figure(figsize=(10, 6))
        plt.plot(t_span, state_probs[:, 0], label="P(|0⟩)")
        plt.plot(t_span, state_probs[:, 1], label="P(|1⟩)")
        plt.xlabel("Time (ns)")
        plt.ylabel("Probability")
        plt.title("Qubit State Evolution Under Gaussian Pulse")
        plt.legend()
        plt.grid()
        plt.show()

    if bool_blochsphere:
        plot_bloch_sphere(result.y)

    expected_state = np.array([1, 1]) / np.sqrt(2)
    final_state = result.y[-1]
    overlap = np.abs(np.vdot(expected_state, final_state)) ** 2

    state_probs = prob(result.y)
    final_probs[0] = state_probs[-1, 0]
    final_probs[1] = state_probs[-1, 1]

    return drive_strength, sigma, final_probs, overlap, result.y

