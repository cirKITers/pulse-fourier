import jax
import matplotlib
from qiskit_dynamics import Solver, Signal
from scipy.integrate import quad
from qiskit.circuit.library import RXGate, HGate, RZGate
from scipy.linalg import expm

from src.utils.definitions import *
# from scipy.linalg import expm  # For matrix exponentiation

from src.utils.helpers import *
from utils.visualize.bloch_sphere import *
from utils.visualize.probabilites import plot_probabilities
from constants import *

matplotlib.use('TkAgg')  # Force TkAgg backend
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')


# TASK simplify drive strength but save derivation

# Decoherence accounting (?)

# TODO omega_list = [5.0, 4.9, 4.85, 4.95, ...] for n qubit systems

# TODO conjugate transpose for RZ
# TODO plot probabilities, plot entanglement
def RZ_pulseMULT(theta, current_state, plot_prob=False, plot_blochsphere=False):
    """
    Implements an RZ gate (rotation around Z-axis) by angle theta on all qubits in current_state.
    Uses a constant drive Hamiltonian with Z operator, with final state transformed to lab frame.
    """
    num_qubits = int(np.log2(current_state.dim))
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


def RX_pulseSPEC(theta, current_state, target_qubits='all', plot_prob=False, plot_blochsphere=False):
    """
    Implements an RX gate (rotation around X-axis) by angle theta on specified qubits in current_state.
    Uses a constant drive Hamiltonian with X operator on target qubits, with final state transformed to lab frame.

    Parameters:
    - theta: Rotation angle in radians.
    - current_state: Initial quantum state (state vector or density matrix).
    - target_qubits: List of qubit indices (integers) to apply the RX gate to.
    - plot_prob: Boolean to plot probabilities (not implemented here).
    - plot_blochsphere: Boolean to plot Bloch sphere (not implemented here).

    Returns:
    - result: Simulation result from Solver.solve(), later maybe also testing logic
    """

    num_qubits = int(np.log2(current_state.dim))

    if target_qubits == 'all':
        target_qubits = list(range(num_qubits))
    elif isinstance(target_qubits, int):
        target_qubits = [target_qubits]
    invalid = [k for k in target_qubits if not 0 <= k < num_qubits]
    assert not invalid, f"Target qubit(s) {invalid} are out of range [0, {num_qubits - 1}]."

    # expected_state = current_state.evolve(RZGate(theta).control(num_qubits))

    t_span = np.linspace(0, duration * dt_, duration + 1)

    # RX:
    def envelope(t):
        center = duration * dt_ / 2
        return np.exp(-((t - center) ** 2) / (2 * sigma ** 2))

    integral, _ = quad(envelope, t_span[0], t_span[-1])
    # Calculate drive_strength
    strength_scale = 0.3183109217857033
    drive_strength = theta / integral
    drive_strength = drive_strength * strength_scale

    # Construct multi-qubit static Hamiltonian (applied to all qubits)
    H_static_single = static_hamiltonian(omega=omega)
    H_static_multi = sum_operator(H_static_single, num_qubits)

    H_drive_X_single = drive_hamiltonian(drive_strength)
    # Construct drive Hamiltonian (applied only to target_qubits)
    if not target_qubits:  # If empty, no rotation
        H_drive_X_multi = Operator(np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex))
    else:
        H_drive_X_multi = Operator(np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex))
        for k in target_qubits:
            H_drive_X_multi += operator_on_qubit(H_drive_X_single, k, num_qubits)

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
    # t_max = t_span[-1]
    # U_static = expm(-1j * H_static_multi * t_max)
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
    # if plot_blochsphere:
    #     trajectory_lab = []
    #     for state in result.y:
    #         # print(f"state.data shape: {state.data.shape}")
    #         trajectory_lab.append(Statevector(U_static @ state.data))
    #     bloch_sphere_multiqubit_trajectory(trajectory_lab, list(range(num_qubits)), False)

    return None, None, result.y


# TODO find out if trajectory might be needed and return all together
def CNOT_pulseEcho(current_state, control_qubit, target_qubit, omega_list, g, drive_strength, cnot_duration=120, cnot_phase=0.0, cnot_sigma=15):
    """
    CNOT gate at pulse level using cross-resonance with compensating single-qubit gates (echoing: Pulse-Level Optimization of Parameterized Quantum Circuits for Variational Quantum Algorithms)
    """
    num_qubits = int(np.log2(current_state.dim))
    t_span = np.linspace(0, duration * dt_, duration + 1)

    # RX(π/2) on control qubit
    theta = np.pi / 2
    _, _, resulty = RX_pulseSPEC(theta, current_state, control_qubit)
    current_state = resulty[-1]

    # RX(-π/2) on target qubit
    theta = - np.pi / 2
    _, _, resulty = RX_pulseSPEC(theta, current_state, target_qubit)
    current_state = resulty[-1]

    center = duration * dt_ / 2

    # Cross-resonance pulse
    H_static_multi = Operator(np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex))
    for k in range(num_qubits):
        H_static_multi += operator_on_qubit(static_hamiltonian(omega_list[k]), k, num_qubits)
    H_static_multi += coupling_hamiltonian(num_qubits, control_qubit, target_qubit, g)
    H_drive_control = operator_on_qubit(drive_hamiltonian(drive_strength), control_qubit, num_qubits)
    ham_solver = Solver(static_hamiltonian=H_static_multi, hamiltonian_operators=[H_drive_control], rotating_frame=H_static_multi)

    def gaussian_envelope(t): return amp * jnp.exp(-((t - center) ** 2) / (2 * cnot_sigma ** 2))

    # TODO !! modified phase and t_span
    t_span = np.linspace(0, cnot_duration * dt_, cnot_duration + 1)
    gaussian_signal = Signal(envelope=gaussian_envelope, carrier_freq=omega_list[target_qubit], phase=cnot_phase)
    current_state = ham_solver.solve(t_span=t_span, y0=current_state, method='jax_odeint', signals=[gaussian_signal]).y[-1]

    # RX(π/2) on control
    theta = np.pi / 2
    _, _, resulty = RX_pulseSPEC(theta, current_state, control_qubit)
    current_state = resulty[-1]

    # RX(-π/2) on target
    theta = - np.pi / 2
    _, _, resulty = RX_pulseSPEC(theta, current_state, target_qubit)
    current_state = resulty[-1]

    return None, None, current_state

def CNOT_pulseNoEcho(current_state, control_qubit, target_qubit, omega_list, g, drive_strength, cnot_duration=120, cnot_phase=0.0, cnot_sigma=15):

    num_qubits = int(np.log2(current_state.dim))

    # Cross-resonance pulse
    H_static_multi = Operator(np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex))
    for k in range(num_qubits):
        H_static_multi += operator_on_qubit(static_hamiltonian(omega_list[k]), k, num_qubits)
    H_static_multi += coupling_hamiltonian(num_qubits, control_qubit, target_qubit, g)
    H_drive_control = operator_on_qubit(drive_hamiltonian(drive_strength), control_qubit, num_qubits)
    ham_solver = Solver(static_hamiltonian=H_static_multi, hamiltonian_operators=[H_drive_control], rotating_frame=H_static_multi)
    center = duration * dt_ / 2
    def gaussian_envelope(t): return amp * jnp.exp(-((t - center) ** 2) / (2 * cnot_sigma ** 2))

    # TODO !! modified phase and t_span
    t_span = np.linspace(0, cnot_duration * dt_, cnot_duration + 1)
    gaussian_signal = Signal(envelope=gaussian_envelope, carrier_freq=omega_list[target_qubit], phase=cnot_phase)
    current_state = ham_solver.solve(t_span=t_span, y0=current_state, method='jax_odeint', signals=[gaussian_signal]).y[-1]

    return None, None, current_state


def CNOT_pulse(current_state, control_qubit, target_qubit, omega_list, g, ds, phase_test):
    """
    CNOT gate at pulse level using cross-resonance.
    """
    num_qubits = int(np.log2(current_state.dim))

    t_span = np.linspace(0, durationCNOT * dt_, durationCNOT + 1)

    H_static_multi = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex)
    for k in range(num_qubits):
        H_static_multi += operator_on_qubit(static_hamiltonian(omega_list[k]), k, num_qubits)
    H_static_multi += coupling_hamiltonian(num_qubits, control_qubit, target_qubit, g)

    H_drive_control = operator_on_qubit(drive_hamiltonian(ds), control_qubit, num_qubits)

    ham_solver = Solver(
        static_hamiltonian=H_static_multi,
        hamiltonian_operators=[H_drive_control],
        rotating_frame=H_static_multi
    )

    def gaussian_envelope(t):
        center = durationCNOT * dt_ / 2
        return amp * jnp.exp(-((t - center) ** 2) / (2 * sigmaCNOT ** 2))

    gaussian_signal = Signal(
        envelope=gaussian_envelope,
        carrier_freq=omega_list[target_qubit],  # Cross-resonance: drive at target frequency
        phase=phase_test
    )

    result = ham_solver.solve(
        t_span=t_span,
        y0=current_state,
        method='jax_odeint',
        signals=[gaussian_signal]
    )

    return None, None, result.y
