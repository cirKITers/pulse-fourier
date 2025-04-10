import jax
import matplotlib
from qiskit_dynamics import Solver, Signal
from scipy.integrate import quad
from qiskit.circuit.library import RXGate, HGate, RZGate
from scipy.linalg import expm  # For matrix exponentiation

from utils.definitions import *
from utils.helpers import *
from visuals.bloch_sphere import bloch_sphere_trajectory
from visuals.probabilites import plot_probabilities
from constants import *

matplotlib.use('TkAgg')  # Force TkAgg backend
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')


# new
# CNOT gate
# Cluster for validation

# old
# Duration always assuming _dt=0.1, Number of time steps (samples) -> constants.py


def H_pulse(sigma, current_state, plot_prob=False, plot_blochsphere=False):
    drive_strength = 0.042780849440995694
    omega = 5.0
    duration = 120
    phase = np.pi / 2
    expected_state = current_state.evolve(HGate())
    probs, trajectory = pulse(drive_strength=drive_strength, sigma=sigma, duration=duration, omega=omega, phase=phase,
                              current_state=current_state, plot_probs=plot_prob, plot_blochsphere=plot_blochsphere)
    result = swap_amplitudes(trajectory)
    final_probs = swap_probs(probs)
    return drive_strength, final_probs, overlap(expected_state, result[-1]), result


def RZ_pulse(theta, sigma, current_state, plot_prob=False, plot_blochsphere=False):
    expected_state = current_state.evolve(RZGate(theta))

    t_span = np.linspace(0, duration * dt_, duration + 1)
    t_max = t_span[-1]

    k = 5.524648297886591
    drive_strength = (theta / 2 - 5.0 / 2 * 12 + 2 * np.pi * k) / 12

    H_static = static_hamiltonian(omega=omega)
    H_drive_Z = Z  # Pauli Z operator

    ham_solver = Solver(
        static_hamiltonian=H_static,
        hamiltonian_operators=[H_drive_Z],
        rotating_frame=H_static
    )

    def constant_envelope(t):
        return drive_strength

    signal = Signal(
        envelope=constant_envelope,
        carrier_freq=0.0,  # No oscillation for Z drive
        phase=0.0
    )

    result = ham_solver.solve(
        t_span=t_span,
        y0=current_state,
        method='jax_odeint',
        signals=[signal]
    )

    result.y = quadrant_selective_conjugate_transpose_OneQ(result.y)

    # Transform final state from rotating frame to lab frame
    U_static = expm(-1j * H_static * t_max)  # Transformation matrix (NumPy array)
    final_state_rot = result.y[-1].data  # Convert Statevector to NumPy array
    final_state_lab = U_static @ final_state_rot  # Lab frame final state as NumPy array
    final_state_lab = Statevector(final_state_lab)  # Convert back to Statevector for consistency

    # Compute probabilities (frame-invariant)
    state_probs = prob(result.y)
    final_probs = np.zeros(2)
    final_probs[0] = state_probs[-1, 0]
    final_probs[1] = state_probs[-1, 1]

    # Compute overlap in lab frame
    ol = overlap(expected_state, final_state_lab)

    # Optionally transform all states for plotting (lab frame)
    if plot_prob:
        plot_probabilities(t_span, state_probs)  # Probabilities are frame-invariant
    if plot_blochsphere:
        # Transform entire trajectory to lab frame for plotting
        trajectory_lab = []
        for state in result.y:
            trajectory_lab.append(Statevector(U_static @ state.data))

        bloch_sphere_trajectory(trajectory_lab, False)

    return drive_strength, final_probs, ol, result.y


# uses a time-dependent gaussian envelope
def RX_pulse(theta, sigma, current_state, plot_prob=False, plot_blochsphere=False):
    omega = 5.0
    duration = 120
    t_span = np.linspace(0, duration * dt_, duration + 1)
    t_max = t_span[-1]
    center = duration * dt_ / 2

    # Define the gaussian envelope function
    def envelope(t):
        return np.exp(-((t - center) ** 2) / (2 * sigma ** 2))

    # Calculate the integral of the envelope numerically
    integral, _ = quad(envelope, t_span[0], t_span[-1])

    # Calculate drive_strength
    strength_scale = 0.3183109217857033
    drive_strength = theta / integral
    drive_strength = drive_strength * strength_scale

    final_probs, result_y = pulse(drive_strength, sigma, duration, omega, 0, current_state, plot_prob, plot_blochsphere)

    # Calculate the correct expected_state
    rx_gate = RXGate(theta)
    expected_state_correct = current_state.evolve(rx_gate)

    # Transform the final state to the lab frame
    H_static = static_hamiltonian(omega=omega)
    U_static = expm(-1j * H_static * t_max)
    final_state_rot = result_y[-1].data
    final_state_lab = U_static @ final_state_rot
    final_state_lab = Statevector(final_state_lab)

    # If plot_blochsphere, transform the entire trajectory
    if plot_blochsphere:
        trajectory_lab = []
        for state in result_y:
            state_lab = U_static @ state.data
            trajectory_lab.append(Statevector(state_lab))
        bloch_sphere_trajectory(trajectory_lab)

    return drive_strength, final_probs, overlap(expected_state_correct, final_state_lab), result_y


def pulse(drive_strength, sigma, duration, omega, phase, current_state, plot_probs, plot_blochsphere):
    final_probs = np.zeros(2)

    t_span = np.linspace(0, duration * dt_, duration + 1)  # Tells solver when to check the qubits state
    H_static = static_hamiltonian(omega=omega)
    H_drive = drive_hamiltonian(drive_strength=drive_strength)

    ham_solver = Solver(
        static_hamiltonian=H_static,
        hamiltonian_operators=[H_drive],
        rotating_frame=H_static
    )

    def gaussian_envelope(t):
        center = duration * dt_ / 2
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

    final_probs[0] = state_probs[-1, 0]
    final_probs[1] = state_probs[-1, 1]

    if plot_probs:
        plot_probabilities(t_span, state_probs)
    if plot_blochsphere:
        bloch_sphere_trajectory(result.y)

    return final_probs, result.y


