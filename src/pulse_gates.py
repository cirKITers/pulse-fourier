import jax
import matplotlib
from qiskit_dynamics import Solver, Signal
from scipy.integrate import quad
from qiskit.circuit.library import RXGate, HGate, RZGate
from scipy.linalg import expm

from src.utils.definitions import *
# from scipy.linalg import expm  # For matrix exponentiation

from utils.helpers import *
from utils.visualize.bloch_sphere import *
from utils.visualize.probabilites import plot_probabilities
from constants import *

matplotlib.use('TkAgg')  # Force TkAgg backend
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

# TASK simplify drive strength but save derivation

# Decoherence accounting (?)

# TODO conjugate transpose for RZ
# TODO plot probabilities, plot entanglement
def RZ_pulseMULT(theta, current_state, plot_prob=False, plot_blochsphere=False):
    """
    Implements an RZ gate (rotation around Z-axis) by angle theta on all qubits in current_state.
    Uses a constant drive Hamiltonian with Z operator, with final state transformed to lab frame.
    """
    num_qubits = int(np.log2(current_state.dim))  # Determine number of qubits from state vector dimension
    # expected_state = current_state.evolve(RZGate(theta).control(num_qubits))

    t_span = np.linspace(0, duration * dt_, duration + 1)
    t_max = t_span[-1]

    # RX:
    # center = duration * dt_ / 2
    # def envelope(t):
    #     return np.exp(-((t - center) ** 2) / (2 * sigma ** 2))
    # integral, _ = quad(envelope, t_span[0], t_span[-1])
    # # Calculate drive_strength
    # strength_scale = 0.3183109217857033
    # drive_strength = theta / integral
    # drive_strength = drive_strength * strength_scale

    # RZ:
    k = 5.524648297886591
    drive_strength = (theta / 2 - 5.0 / 2 * 12 + 2 * np.pi * k) / 12
    # RZdrive_strength = theta / (2 * t_max)

    # Construct multi-qubit static and drive Hamiltonians
    H_static_single = static_hamiltonian(omega=omega)
    H_drive_Z_single = Z

    H_static_multi = sum_operator(H_static_single, num_qubits)
    H_drive_Z_multi = sum_operator(H_drive_Z_single, num_qubits)

    ham_solver = Solver(
        static_hamiltonian=H_static_multi,
        hamiltonian_operators=[H_drive_Z_multi],
        rotating_frame=H_static_multi
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

    # TODO very wrong approach to correct:
    # result.y = quadrant_selective_conjugate_transpose(result.y, num_qubits)

    # Transform final state from rotating frame to lab frame
    U_static = expm(-1j * H_static_multi * t_max)
    # final_state_rot = result.y[-1].data
    # final_state_lab = U_static @ final_state_rot
    # final_state_lab = Statevector(final_state_lab)

    # Compute probabilities (frame-invariant)
    # state_probs = prob(result.y)
    # final_probs = prob(result.y[-1])

    # Compute overlap in lab frame
    # ol = overlap(expected_state, final_state_lab)

    # Optionally transform all states for plotting (lab frame)
    # if plot_prob:
    #     plot_probabilities
    # print(f"initial state: {current_state}")
    # print(f"result.y[-1]: {result.y[-1]}")
    # print(f"U_static shape: {U_static.shape}")
    if plot_blochsphere:
        trajectory_lab = []
        for state in result.y:
            # print(f"state.data shape: {state.data.shape}")
            trajectory_lab.append(Statevector(U_static @ state.data))
        bloch_sphere_multiqubit_trajectory(trajectory_lab, list(range(num_qubits)), False)

    return None, None, result.y


def RX_pulseMULT(theta, current_state, plot_prob=False, plot_blochsphere=False):
    """
    Implements an RX gate (rotation around X-axis) by angle theta on all qubits in current_state.
    Uses a constant drive Hamiltonian with X operator, with final state transformed to lab frame.
    """
    num_qubits = int(np.log2(current_state.dim))  # Determine number of qubits from state vector dimension
    # expected_state = current_state.evolve(RZGate(theta).control(num_qubits))

    t_span = np.linspace(0, duration * dt_, duration + 1)
    t_max = t_span[-1]

    # RX:
    center = duration * dt_ / 2
    def envelope(t):
        return np.exp(-((t - center) ** 2) / (2 * sigma ** 2))
    integral, _ = quad(envelope, t_span[0], t_span[-1])
    # Calculate drive_strength
    strength_scale = 0.3183109217857033
    drive_strength = theta / integral
    drive_strength = drive_strength * strength_scale

    # Construct multi-qubit static and drive Hamiltonians
    H_static_single = static_hamiltonian(omega=omega)
    H_drive_X_single = drive_hamiltonian(drive_strength)

    H_static_multi = sum_operator(H_static_single, num_qubits)
    H_drive_X_multi = sum_operator(H_drive_X_single, num_qubits)

    ham_solver = Solver(
        static_hamiltonian=H_static_multi,
        hamiltonian_operators=[H_drive_X_multi],
        rotating_frame=H_static_multi
    )

    def gaussian_envelope(t):
        center = duration * dt_ / 2
        return amp * jnp.exp(-((t - center) ** 2) / (2 * sigma ** 2))

    gaussian_signal = Signal(
        envelope=gaussian_envelope,
        carrier_freq=omega,  # No oscillation for Z drive
        phase=0.0
    )

    result = ham_solver.solve(
        t_span=t_span,
        y0=current_state,
        method='jax_odeint',
        signals=[gaussian_signal]
    )

    # Transform final state from rotating frame to lab frame
    U_static = expm(-1j * H_static_multi * t_max)
    # final_state_rot = result.y[-1].data
    # final_state_lab = U_static @ final_state_rot
    # final_state_lab = Statevector(final_state_lab)

    # Compute probabilities (frame-invariant)
    # state_probs = prob(result.y)
    # final_probs = prob(result.y[-1])

    # Compute overlap in lab frame
    # ol = overlap(expected_state, final_state_lab)

    # Optionally transform all states for plotting (lab frame)
    # if plot_prob:
    #     plot_probabilities
    if plot_blochsphere:
        trajectory_lab = []
        for state in result.y:
            # print(f"state.data shape: {state.data.shape}")
            trajectory_lab.append(Statevector(U_static @ state.data))
        bloch_sphere_multiqubit_trajectory(trajectory_lab, list(range(num_qubits)), False)

    return None, None, result.y


def CNOT_pulse(current_state, control_qubit, target_qubit, omega_list, g, drive_strength):
    """
    Implements a CNOT gate at the pulse level using cross-resonance.

    Args:
        current_state (Statevector): Initial quantum state.
        control_qubit (int): Index of the control qubit.
        target_qubit (int): Index of the target qubit.
        omega_list (list): Frequencies of all qubits.
        g (float): Coupling strength between control and target qubits.
        drive_strength (float): Strength of the drive pulse (needs calibration).

    Returns:
        tuple: (None, None, evolved_state) to match RX_pulseMULT output format.
    """
    num_qubits = int(np.log2(current_state.dim))  # Number of qubits from state dimension

    # Time span for evolution
    t_span = np.linspace(0, duration * dt_, duration + 1)

    # Static Hamiltonian: Sum of individual qubit terms + coupling
    H_static_multi = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex)
    for k in range(num_qubits):
        H_static_multi += operator_on_qubit(static_hamiltonian(omega_list[k]), k, num_qubits)
    H_static_multi += coupling_hamiltonian(num_qubits, control_qubit, target_qubit, g)

    # Drive Hamiltonian: Applied only to the control qubit
    H_drive_control = operator_on_qubit(drive_hamiltonian(drive_strength), control_qubit, num_qubits)

    # Define the solver
    ham_solver = Solver(
        static_hamiltonian=H_static_multi,
        hamiltonian_operators=[H_drive_control],
        rotating_frame=H_static_multi
    )

    # Gaussian envelope for the pulse
    def gaussian_envelope(t):
        center = duration * dt_ / 2
        return amp * jnp.exp(-((t - center) ** 2) / (2 * sigma ** 2))

    # Signal with carrier frequency set to target qubit's frequency
    gaussian_signal = Signal(
        envelope=gaussian_envelope,
        carrier_freq=omega_list[target_qubit],  # Cross-resonance: drive at target frequency
        phase=0.0
    )

    # Solve the time evolution
    result = ham_solver.solve(
        t_span=t_span,
        y0=current_state,
        method='jax_odeint',
        signals=[gaussian_signal]
    )

    return None, None, result.y
