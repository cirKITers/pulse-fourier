import jax
from qiskit_dynamics import Solver, Signal
from scipy.integrate import quad

from visuals.bloch_sphere import *
from constants import *

import matplotlib
matplotlib.use('TkAgg')  # Force TkAgg backend
import matplotlib.pyplot as plt #import after setting backend

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

# TODO plot the pulses themselves


# TODO h pulse probably needs phase correction and higher accuracy

# TODO pre calc common values like np.pi /2 for comp acceleration


# not working after opt phases

def cnot2(
        current_state,
        ryp,
        rxp,
):
    ryp = ryp
    rxp = rxp

    _, _, current_state = H_pulseSPEC2(current_state, 1, ryp, rxp)

    _, _, current_state = cz(current_state[-1])

    _, _, current_state = H_pulseSPEC2(current_state[-1], 1, ryp, rxp)

    # _, _, current_state = cz(current_state[-1])         # for adjusting the global phase error

    return None, None, current_state


# works for [0, 1] and [1, 0] but hadamard1 problem in phase
# second hadamard phase = pi / 2 gives PSI PLUS, - pi / 2 gives PHI MINUS
# find one phase that works always, already tried?
def cnot(
        current_state,
        phase
):

    thetaOne = phase   # np.pi / 2
    # _, _, current_state = H_pulseSPEC(current_state, 1, thetaOne)

    _, _, current_state = cz(current_state)

    thetaTwo = phase   # - np.pi / 2
    _, _, current_state = H_pulseSPEC(current_state[-1], 1, thetaTwo)

    # _, _, current_state = cz(current_state[-1])         # for adjusting the global phase error

    return None, None, current_state


def cz(
        current_state,
        num_qubits=2,
        control_qubit=0,
        target_qubit=1,
):

    if control_qubit == target_qubit or control_qubit >= num_qubits or target_qubit >= num_qubits:
        raise ValueError("Invalid control or target qubit indices")

    # Static Hamiltonian: sum of ω * σ_z/2 for each qubit
    H_static_single = static_hamiltonian(omega)
    H_static_multi = sum_operator(H_static_single, num_qubits)

    # Drive Hamiltonian: (π/4) * Z ⊗ Z for CZ interaction
    H_drive_zz_single = Operator(np.kron(Z, Z))

    diagonal_values = [0, 0, 0, np.pi]      # (1/4)(I ⊗ I - I ⊗ Z - Z ⊗ I + Z ⊗ Z)
    matrix = np.diag(diagonal_values)

    # H_drive_zz_multi = cs*np.pi/2*H_drive_zz_single  # Scale by drive strength

    ham_solver = Solver(
        static_hamiltonian=H_static_multi,
        hamiltonian_operators=[matrix],
        rotating_frame=H_static_multi
    )

    t_span = np.arange(0, duration * dt_, dt_)

    def constant_envelope(t):
        return 0.11884147603542072   # for phase = np.pi/4  (drive_strength)

    signal = Signal(
        envelope=constant_envelope,
        carrier_freq=0.0,  # No oscillation for Z ⊗ Z drive
        phase=np.pi/4       # TODO why does it converge much better?
    )

    result = ham_solver.solve(
        t_span=t_span,
        y0=current_state,
        method='jax_odeint',
        signals=[signal]
    )

    return None, None, result.y


def H_pulseSPEC3(current_state, target_qubits, k1, k2, plot=False, bool_blochsphere=False):
    # num_qubits = int(np.log2(current_state.dim))

    # if target_qubits == 'all':
    #     target_qubits = list(range(num_qubits))
    # elif isinstance(target_qubits, int):
    #     target_qubits = [target_qubits]
    # invalid = [k for k in target_qubits if not 0 <= k < num_qubits]
    # assert not invalid, f"Target qubit(s) {invalid} are out of range [0, {num_qubits - 1}]."

    # phase = pi / 2 gives PSI PLUS, - pi / 2 gives PHI MINUS
    phase = 0.0

    # well calibrated for drive_hamiltonian function
    # k = 0.042780586392198006

    H_static_single = static_hamiltonian(omega=omega)
    # H_static_multi = sum_operator(H_static_single, num_qubits)

    # # ds = k*np.pi/2
    # H_drive_single = drive_hamiltonian(drive_strength=k)
    #
    # # Construct drive Hamiltonian (applied only to target_qubits)
    # if not target_qubits:  # If empty, no rotation
    #     H_drive_multi = Operator(np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex))
    # else:
    #     H_drive_multi = Operator(np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex))
    #     for q in target_qubits:
    #         H_drive_multi += operator_on_qubit(H_drive_single, q, num_qubits)

    # matrix = (1 / np.sqrt(2)) * (k1*X + k2*Z)

    theta = np.pi
    t_span = np.linspace(0, duration * dt_, duration + 1)  # Tells solver when to check the qubits state

    # RX:
    def envelope(t):
        center = duration * dt_ / 2
        return np.exp(-((t - center) ** 2) / (2 * sigma ** 2))

    integral, _ = quad(envelope, t_span[0], t_span[-1])
    # Calculate drive_strength
    strength_scale = 0.3183109217857033
    drive_strength = theta / integral
    drive_strength = drive_strength * strength_scale
    kx = drive_strength


    # RZ
    k = 0.04166666094977508
    drive_strength = theta * k
    kz = drive_strength

    X_part = drive_hamiltonian(kx)
    Z_part = SIGMA_Z

    ham_solver = Solver(
        static_hamiltonian=H_static_single,
        hamiltonian_operators=[X_part, Z_part],
        rotating_frame=H_static_single
    )

    def gaussian_envelope(t):
        center = duration * dt_ / 2
        return amp * jnp.exp(-((t - center) ** 2) / (2 * sigma ** 2))

    def constant_envelope(t):
        return kz

    gaussian_signal = Signal(
        envelope=gaussian_envelope,
        carrier_freq=omega,
        phase=phase
    )

    constant_signal = Signal(
        envelope=constant_envelope,
        carrier_freq=omega,
        phase=phase
    )

    result = ham_solver.solve(
        t_span=t_span,
        y0=current_state,
        method='jax_odeint',
        signals=[gaussian_signal, constant_signal]
    )

    # final_probs = np.zeros(2)
    # state_probs = prob(result.y)
    # final_state = result.y[-1]
    # overlap = np.abs(np.vdot(expected_state, final_state)) ** 2

    # final_probs[0] = state_probs[-1, 0]
    # final_probs[1] = state_probs[-1, 1]
    #
    # if plot:
    #     plot_probabilities(t_span, state_probs)
    # if bool_blochsphere:
    #     plot_bloch_sphere(result.y)

    return None, None, result.y


# TODO: At the end: build pulse machine as class with stored current_state, num_qubits etc; bundle calculation in init

# TODO omega_list = [5.0, 4.9, 4.85, 4.95, ...] for n qubit systems

# TODO plot probabilities, plot entanglement, plot blochsphere in lab frame

# TODO clean pulse parameter and isolate as far as possible in k, to minimize pulse parameters

# TODO use sparse pauli opt where possible to accelerate code

def H_pulseSPEC2(current_state, target_qubits, ryp, rxp, plot=False, bool_blochsphere=False):
    rx_phase = rxp

    _, _, current_trajectory = RX_pulseSPEC(np.pi, current_state, rx_phase, target_qubits)

    ry_phase = ryp
    _, _, current_trajectory = RY_pulseSPEC(np.pi/2, current_trajectory[-1], ry_phase, target_qubits)

    _, _, current_trajectory = RX_pulseSPEC(-np.pi, current_trajectory[-1], rx_phase, target_qubits)

    # _, _, current_trajectory = RZ_pulseSPEC(np.pi, current_trajectory[-1], target_qubits)

    # final_probs = np.zeros(2)
    # state_probs = prob(result.y)
    # final_state = result.y[-1]
    # overlap = np.abs(np.vdot(expected_state, final_state)) ** 2

    # final_probs[0] = state_probs[-1, 0]
    # final_probs[1] = state_probs[-1, 1]
    #
    # if plot:
    #     plot_probabilities(t_span, state_probs)
    # if bool_blochsphere:
    #     plot_bloch_sphere(result.y)

    return None, None, current_trajectory


def H_pulseSPEC(current_state, target_qubits, phase, plot=False, bool_blochsphere=False):
    num_qubits = int(np.log2(current_state.dim))

    if target_qubits == 'all':
        target_qubits = list(range(num_qubits))
    elif isinstance(target_qubits, int):
        target_qubits = [target_qubits]
    invalid = [k for k in target_qubits if not 0 <= k < num_qubits]
    assert not invalid, f"Target qubit(s) {invalid} are out of range [0, {num_qubits - 1}]."

    # phase = pi / 2 gives PSI PLUS, - pi / 2 gives PHI MINUS
    phase = phase

    # well calibrated for drive_hamiltonian function
    k = 0.042780586392198006

    H_static_single = static_hamiltonian(omega=omega)
    H_static_multi = sum_operator(H_static_single, num_qubits)

    # ds = k*np.pi/2
    H_drive_single = drive_hamiltonian(drive_strength=k)

    # Construct drive Hamiltonian (applied only to target_qubits)
    if not target_qubits:  # If empty, no rotation
        H_drive_multi = Operator(np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex))
    else:
        H_drive_multi = Operator(np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex))
        for q in target_qubits:
            H_drive_multi += operator_on_qubit(H_drive_single, q, num_qubits)

    ham_solver = Solver(
        static_hamiltonian=H_static_multi,
        hamiltonian_operators=[H_drive_multi],
        rotating_frame=H_static_multi
    )

    t_span = np.linspace(0, duration * dt_, duration + 1)   # Tells solver when to check the qubits state

    def gaussian_envelope(t):
        center = duration * dt_ / 2
        return amp * jnp.exp(-((t - center) ** 2) / (2 * sigma ** 2))

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

    # final_probs = np.zeros(2)
    # state_probs = prob(result.y)
    # final_state = result.y[-1]
    # overlap = np.abs(np.vdot(expected_state, final_state)) ** 2

    # final_probs[0] = state_probs[-1, 0]
    # final_probs[1] = state_probs[-1, 1]
    #
    # if plot:
    #     plot_probabilities(t_span, state_probs)
    # if bool_blochsphere:
    #     plot_bloch_sphere(result.y)

    _, _, result = RZ_pulseSPEC(np.pi, result.y[-1], target_qubits)

    return None, None, result


# worst similarity around 0.99
def RY_pulseSPEC(theta, current_state, temp_p, target_qubits='all', plot_prob=False, plot_blochsphere=False):
    num_qubits = int(np.log2(current_state.dim))

    if target_qubits == 'all':
        target_qubits = list(range(num_qubits))
    elif isinstance(target_qubits, int):
        target_qubits = [target_qubits]
    invalid = [k for k in target_qubits if not 0 <= k < num_qubits]
    assert not invalid, f"Target qubit(s) {invalid} are out of range [0, {num_qubits - 1}]."

    t_span = np.linspace(0, duration * dt_, duration + 1)

    drive_strength = 0.027235035038834040234 * theta                   # 0.042780692999130815               # temporarily not high accuracy

    # Construct multi-qubit static Hamiltonian (applied to all qubits)
    H_static_single = static_hamiltonian(omega=omega)
    H_static_multi = sum_operator(H_static_single, num_qubits)

    H_drive_X_single = drive_Y_hamiltonian(drive_strength)
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

    delta = 0.05

    def DRAG_gaussian_envelope(t):
        """Gaussian envelope with DRAG correction."""
        center = duration * dt_ / 2
        gaussian = amp * jnp.exp(-((t - center) ** 2) / (2 * sigma ** 2))
        derivative = -amp * (t - center) / sigma ** 2 * jnp.exp(-((t - center) ** 2) / (2 * sigma ** 2))
        return gaussian + delta * derivative

    gaussian_signal = Signal(
        envelope=gaussian_envelope,
        carrier_freq=omega,  # No oscillation for Z drive
        phase=temp_p        # 0.0
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


# similarity of 1.0
def RZ_pulseSPEC(theta, current_state, target_qubits, plot_prob=False, plot_blochsphere=False):
    """
    Implements an RZ gate (rotation around Z-axis) by angle theta on all qubits in current_state.
    Uses a constant drive Hamiltonian with Z operator, with final state transformed to lab frame.
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
    t_max = t_span[-1]

    # RZ:
    # k_old = 5.524648297886591 # transpose conjugate
    # k = 6.87777037921978  # best k, but bad

    # 1. drive_strength = (theta/2 - omega/2 * t + 2 * np.pi * k) / t  # t = 12
    # drive_strength = (theta / 2 - 5.0 / 2 * 12 + 2 * np.pi * k) / 12
    # 2. RZdrive_strength = theta / (2 * t_max)

    k = 0.04166666094977508
    drive_strength = theta * k

    # static
    H_static_single = static_hamiltonian(omega=omega)
    H_static_multi = sum_operator(H_static_single, num_qubits)

    # drive
    H_drive_Z_single = SIGMA_Z
    if not target_qubits:
        H_drive_Z_multi = Operator(np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex))
    else:
        H_drive_Z_multi = Operator(np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex))
        for k in target_qubits:
            H_drive_Z_multi += operator_on_qubit(H_drive_Z_single, k, num_qubits)

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

    # print("drawing...")
    # # T = duration * dt_
    # signal.draw(t0=0, tf=duration * dt_, n=100)
    # plt.show()

    result = ham_solver.solve(
        t_span=t_span,
        y0=current_state,
        method='jax_odeint',
        signals=[signal]
    )

    # Transform final state from rotating frame to lab frame
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
    # print(f"initial state: {current_state}")
    # print(f"result.y[-1]: {result.y[-1]}")
    # print(f"U_static shape: {U_static.shape}")
    # if plot_blochsphere:
    #     trajectory_lab = []
    #     for state in result.y:
    #         # print(f"state.data shape: {state.data.shape}")
    #         trajectory_lab.append(Statevector(U_static @ state.data))
    #     bloch_sphere_multiqubit_trajectory(trajectory_lab, list(range(num_qubits)), False)

    return None, None, result.y

# works duration independent!
def RX_pulseSPEC(theta, current_state, temp_p, target_qubits='all', plot_prob=False, plot_blochsphere=False):
    """
    Implements an RX gate (rotation around X-axis) by angle theta on specified qubits in current_state.
    Uses a constant drive Hamiltonian with X operator on target qubits

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
        phase=temp_p,    # 0.0
    )

    # print("drawing...")
    # # T = duration * dt_
    # gaussian_signal.draw(t0=0, tf=duration * dt_, n=1000, function="signal")
    # plt.show()

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

    # pop = [np.abs(y.data[0]) ** 2 for y in result.y]
    # start = 50
    # stop = 58
    # plt.plot(t_span[start:stop], pop[start:stop])
    # plt.show()

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
