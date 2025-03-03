import jax
import jax.numpy as jnp
from qiskit_dynamics import Solver, Signal
import matplotlib.pyplot as plt
import matplotlib
from qiskit_dynamics import DiscreteSignal
from qiskit_dynamics.signals import Convolution

from utils.visualize.bloch_sphere import plot_bloch_sphere
from utils.visualize.probabilites import plot_probabilities

from utils.helpers import *

matplotlib.use('TkAgg')  # Force TkAgg backend
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')


# Expected state always for when |0> initially
# Duration always assuming _dt=0.1, Number of time steps (samples)

# drive_strength  r

# rotation_speed = drive_strength / (2 * np.pi)  # Example value
# duration = int(np.round(theta / (2 * np.pi * rotation_speed)))


def H_pulse(drive_strength, sigma, current_state, plot_prob=False, plot_blochsphere=False):
    omega = 5.0
    duration = 120
    phase = np.pi / 2
    expected_state = np.array([1, 1]) / np.sqrt(2)
    drive_strength, sigma, probs, ol, trajectory = pulse(drive_strength=drive_strength, sigma=sigma, duration=duration, omega=omega, phase=phase,
                                                         expected_state=expected_state, current_state=current_state,
                                                         plot=plot_prob, bool_blochsphere=plot_blochsphere)
    result = swap_dims(trajectory)
    final_probs = swap_probs(probs)
    return drive_strength, sigma, final_probs, ol, result


def RZ_pulse(theta, drive_strength, sigma, current_state, plot_prob=False, plot_blochsphere=False):
    omega = 5.0
    duration = int(theta / (omega * 0.1))
    if duration < 1:
        duration = 1
    phase = np.pi
    expected_state = np.array([np.exp(-1j * theta / 2), 0])
    return pulse(drive_strength=drive_strength, sigma=sigma, duration=duration, omega=omega, phase=phase, expected_state=expected_state,
                 current_state=current_state,
                 plot=plot_prob, bool_blochsphere=plot_blochsphere)


def RX_pulse(theta, drive_strength, sigma, current_state, plot_prob=False, plot_blochsphere=False):
    omega = 5.0
    duration = 120
    phase = 0
    expected_state = np.array([np.cos(theta / 2), -1j * np.sin(theta / 2)])
    return pulse(drive_strength=drive_strength, sigma=sigma, duration=duration, omega=omega, phase=phase, expected_state=expected_state,
                 current_state=current_state,
                 plot=plot_prob, bool_blochsphere=plot_blochsphere)


def pulse(drive_strength, sigma, duration, omega, phase, current_state, expected_state, plot, bool_blochsphere):
    final_probs = np.zeros(2)

    amp = 1.0  # Amplitude, height gaussian bell at peak, default Max
    _dt = 0.1  # Time step in ns
    t_span = np.linspace(0, duration * _dt, duration + 1)  # Tells solver when to check the qubits state
    H_static = static_hamiltonian(omega=omega)
    H_drive = drive_hamiltonian(drive_strength=drive_strength)

    ham_solver = Solver(
        static_hamiltonian=H_static,
        hamiltonian_operators=[H_drive],
        rotating_frame=H_static
    )

    def gaussian_envelope(t):
        center = duration * _dt / 2
        return amp * jnp.exp(-((t - center) ** 2) / (2 * sigma ** 2))

    # https://qiskit-community.github.io/qiskit-dynamics/stubs/qiskit_dynamics.signals.Signal.html
    gaussian_signal = Signal(
        envelope=gaussian_envelope,
        carrier_freq=omega,
        phase=phase
    )

    result = ham_solver.solve(
        t_span=t_span,
        y0=current_state,
        method='jax_odeint',
        signals=[gaussian_signal]
    )

    state_probs = prob(result.y)
    final_state = result.y[-1]
    ol = overlap(expected_state, final_state)

    final_probs[0] = state_probs[-1, 0]
    final_probs[1] = state_probs[-1, 1]

    if plot:
        plot_probabilities(t_span, state_probs)
    if bool_blochsphere:
        plot_bloch_sphere(result.y)

    return drive_strength, sigma, final_probs, ol, result.y
