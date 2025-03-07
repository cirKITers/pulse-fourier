import jax
from qiskit_dynamics import Solver, Signal
import matplotlib
from scipy.linalg import expm

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
    result = swap_amplitudes(trajectory)
    final_probs = swap_probs(probs)
    return drive_strength, sigma, final_probs, ol, result

import numpy as np
from qiskit.quantum_info import Statevector
from qiskit_dynamics import Solver, Signal
from scipy.linalg import expm  # For matrix exponentiation
import jax.numpy as jnp

# Assuming these are defined in your utils.helpers or other modules
from utils.helpers import static_hamiltonian, prob, overlap, Z
from utils.visualize.bloch_sphere import plot_bloch_sphere
from utils.visualize.probabilites import plot_probabilities

def RZ_pulse(theta, drive_strength, sigma, current_state, plot_prob=False, plot_blochsphere=False):
    """
    Implements an RZ gate (rotation around Z-axis) by angle theta on a single qubit.
    Uses a constant drive Hamiltonian with Z operator, with final state transformed to lab frame.
    """
    omega = 5.0
    duration = 120  # Same as other pulses for consistency
    _dt = 0.1
    t_span = np.linspace(0, duration * _dt, duration + 1)
    t_max = t_span[-1]  # Total time for lab frame transformation

    k = 5.524648297886591
    drive_strength = (theta / 2 - 5.0 / 2 * 12 + 2 * np.pi * k) / 12

    # Correct expected state calculation in lab frame
    expected_state = Statevector([np.exp(-1j * theta / 2) * current_state[0],
                                 np.exp(1j * theta / 2) * current_state[1]])

    # Static Hamiltonian (same as single-qubit pulses)
    H_static = static_hamiltonian(omega=omega)

    # Drive Hamiltonian for Z rotation
    H_drive_Z = Z  # Pauli Z operator

    ham_solver = Solver(
        static_hamiltonian=H_static,
        hamiltonian_operators=[H_drive_Z],
        rotating_frame=H_static
    )

    # Constant envelope scaled by drive_strength
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

    result.y = quadrant_selective_conjugate_transpose(result.y)

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

        plot_bloch_sphere(trajectory_lab)

    return drive_strength, sigma, final_probs, ol, result.y

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


def CNOT_Pulse(drive_strength, sigma, current_state, plot_prob=False, plot_blochsphere=False):
    """
    Implements a CNOT gate with qubit 0 as control and qubit 1 as target.
    Sequence: H on control, conditional pi pulse on target, H on control.
    """
    omega = 5.0
    duration_H = 120  # Duration for each H pulse
    duration_pi = 120  # Duration for conditional pi pulse
    total_duration = 2 * duration_H + duration_pi
    _dt = 0.1
    t_span = np.linspace(0, total_duration * _dt, total_duration + 1)

    # Expected state for CNOT (for |00> -> |00>, |10> -> |11>, etc.)
    # Depends on input, so we'll compute overlap with ideal CNOT matrix applied to initial state
    cnot_matrix = jnp.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)
    expected_state = cnot_matrix @ current_state

    # Two-qubit static Hamiltonian
    H_static_2q = 0.5 * omega * (Z0 + Z1)

    # Drive Hamiltonians
    H_drive_control = X0  # Drive for H pulses on control qubit
    H_drive_target = kron(P1, X)  # Conditional drive: |1><1| ⊗ X on target

    ham_solver = Solver(
        static_hamiltonian=H_static_2q,
        hamiltonian_operators=[H_drive_control, H_drive_target],
        rotating_frame=H_static_2q
    )

    # Signal definitions
    amp = 1.0

    def gaussian_H1(t):  # First H pulse, centered at duration_H/2
        center = duration_H * _dt / 2
        return amp * jnp.exp(-((t - center) ** 2) / (2 * sigma ** 2)) * (t <= duration_H * _dt)

    def gaussian_pi(t):  # Conditional pi pulse, centered at duration_H + duration_pi/2
        center = (duration_H + duration_pi / 2) * _dt
        return amp * jnp.exp(-((t - center) ** 2) / (2 * sigma ** 2)) * \
               (t > duration_H * _dt) * (t <= (duration_H + duration_pi) * _dt)

    def gaussian_H2(t):  # Second H pulse, centered at duration_H + duration_pi + duration_H/2
        center = (duration_H + duration_pi + duration_H / 2) * _dt
        return amp * jnp.exp(-((t - center) ** 2) / (2 * sigma ** 2)) * (t > (duration_H + duration_pi) * _dt)

    # Combine signals for control qubit
    def control_signal(t):
        return gaussian_H1(t) + gaussian_H2(t)

    signals = [
        Signal(envelope=control_signal, carrier_freq=omega, phase=np.pi/2),  # Control qubit H pulses
        Signal(envelope=gaussian_pi, carrier_freq=omega, phase=0)  # Target qubit conditional pi pulse
    ]

    # Solve dynamics
    result = ham_solver.solve(
        t_span=t_span,
        y0=current_state,
        method='jax_odeint',
        signals=signals
    )

    # Compute probabilities and overlap
    state_probs = prob(result.y)  # Assumes prob is updated for 4D states (|00>, |01>, |10>, |11>)
    final_state = result.y[-1]
    ol = overlap(expected_state, final_state)

    final_probs = np.zeros(4)  # Two-qubit probabilities
    final_probs[0] = state_probs[-1, 0]  # |00>
    final_probs[1] = state_probs[-1, 1]  # |01>
    final_probs[2] = state_probs[-1, 2]  # |10>
    final_probs[3] = state_probs[-1, 3]  # |11>

    if plot_prob:
        plot_probabilities(t_span, state_probs)
    if plot_blochsphere:
        plot_bloch_sphere(result.y)  # Note: May need adjustment for two-qubit visualization

    return drive_strength, sigma, final_probs, ol, result.y