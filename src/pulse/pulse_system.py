import jax
from qiskit_dynamics import Solver, Signal
from scipy.integrate import quad

from visuals.bloch_sphere import *
from constants import *

import matplotlib

matplotlib.use('TkAgg')  # Force TkAgg backend
import matplotlib.pyplot as plt  # import after setting backend

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')


# TODO test soon: Effect of Pulse-Level parameter on Fourier series (and like that on its coefficients)

# TODO plot the pulses themselves

# TODO: At the end: build pulse machine as class with stored current_state, num_qubits etc; bundle calculation in init

# TODO: play with gate level parameter

# TODO find out if trajectory might be needed and return all together


# TODO plot probabilities, plot entanglement, plot blochsphere in lab frame

# TODO clean pulse parameter and isolate as far as possible in k, to minimize pulse parameters

# TODO use sparse pauli opt where possible to accelerate code


# maybe TODO mix X and Z rotation and plot, probably acceleration when mixing pulses for hadamard

# TODO pre calc common values like np.pi /2 for comp acceleration


class PulseSystem:

    def __init__(self, num_qubits, initial_state):
        self.num_qubits = num_qubits
        self.current_state = initial_state


    def cnot(
            self,
            wires
    ):
        self.h([wires[1]])

        self.cz(wires)

        self.h([wires[1]])

        return self.current_state

    def cz(
            self,
            wires
    ):
        num_qubits = self.num_qubits

        control_qubit = wires[0]
        target_qubit = wires[1]

        if control_qubit == target_qubit or control_qubit >= num_qubits or target_qubit >= num_qubits:
            raise ValueError("Invalid control or target qubit indices")

        # Static Hamiltonian: sum of ω * σ_z/2 for each qubit
        H_static_single = static_hamiltonian(omega)
        H_static_multi = sum_operator(H_static_single, num_qubits)

        # (1/4)(I ⊗ I - I ⊗ Z - Z ⊗ I + Z ⊗ Z)
        diagonal_values = [0, 0, 0, np.pi]
        matrix = np.diag(diagonal_values)

        H_drive = np.diag([0, 0, 0, 0, 0, 0, np.pi, np.pi])

        H_drive = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex)  # TODO precompute

        indices = binary_c_t(num_qubits, control_qubit, target_qubit)
        H_drive[indices, indices] = np.pi

        ham_solver = Solver(
            static_hamiltonian=H_static_multi,
            hamiltonian_operators=[Operator(H_drive)],
            rotating_frame=H_static_multi
        )

        t_span = np.arange(0, duration * dt_, dt_)

        def constant_envelope(t):
            return 0.11884147603542072  # for phase = np.pi/4  (drive_strength)

        signal = Signal(
            envelope=constant_envelope,
            carrier_freq=0.0,  # No oscillation for Z ⊗ Z drive
            phase=np.pi / 4  # TODO why does it converge much better?
        )

        result = ham_solver.solve(
            t_span=t_span,
            y0=self.current_state,
            method='jax_odeint',
            signals=[signal]
        )

        self.current_state = result.y[-1]

    def h(self, target_qubits, correction=True, plot=False, bool_blochsphere=False):
        num_qubits = self.num_qubits

        if target_qubits == 'all':
            target_qubits = list(range(num_qubits))
        elif isinstance(target_qubits, int):
            target_qubits = [target_qubits]
        invalid = [k for k in target_qubits if not 0 <= k < num_qubits]
        assert not invalid, f"Target qubit(s) {invalid} are out of range [0, {num_qubits - 1}]."

        k = 0.042780586392198006

        H_static_single = static_hamiltonian(omega=omega)
        H_static_multi = sum_operator(H_static_single, num_qubits)

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

        t_span = np.linspace(0, duration * dt_, duration + 1)  # Tells solver when to check the qubits state

        def gaussian_envelope(t):
            center = duration * dt_ / 2
            return amp * jnp.exp(-((t - center) ** 2) / (2 * sigma ** 2))

        gaussian_signal = Signal(
            envelope=gaussian_envelope,
            carrier_freq=omega,
            phase=-np.pi / 2
        )
        result = ham_solver.solve(
            t_span=t_span,
            y0=self.current_state,
            method='jax_odeint',
            signals=[gaussian_signal]
        )

        self.current_state = result.y[-1]

        self.rz(np.pi, target_qubits)

        def global_phase_correction(target_q):
            global_phase_correction_angle = 0.0
            case = len(target_q) % 4
            if case == 1:
                global_phase_correction_angle = -np.pi / 2
            elif case == 2:
                global_phase_correction_angle = -np.pi
            elif case == 3:
                global_phase_correction_angle = np.pi / 2
            return global_phase_correction_angle

        if correction:
            glob_phase = global_phase_correction(target_qubits)
            if glob_phase != 0.0:
                glob_hamiltonian_single = glob_phase * I

                if not target_qubits:  # If empty, no rotation
                    glob_hamiltonian = Operator(np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex))
                else:
                    glob_hamiltonian = Operator(np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex))
                    for q in target_qubits:
                        glob_hamiltonian += operator_on_qubit(glob_hamiltonian_single, q, num_qubits)

                glob_hamiltonian = glob_phase * np.eye(2 ** num_qubits, dtype=complex)
                solver_phase = Solver(
                    static_hamiltonian=H_static_multi,
                    hamiltonian_operators=[glob_hamiltonian],
                    rotating_frame=H_static_multi  # No rotating frame for global phase
                )
                phase_correction_time = 1.0  # Choose a time for the phase evolution
                t_span_phase = np.array([0., phase_correction_time])

                result = solver_phase.solve(
                    t_span=t_span_phase,
                    y0=self.current_state,
                    method='jax_odeint',
                    signals=[Signal(envelope=1.0, carrier_freq=0.0)]
                )

                self.current_state = result.y[-1]

    # worst similarity around 0.99
    def ry(self, theta, target_qubits='all', plot_prob=False, plot_blochsphere=False):
        num_qubits = self.num_qubits

        if target_qubits == 'all':
            target_qubits = list(range(num_qubits))
        elif isinstance(target_qubits, int):
            target_qubits = [target_qubits]
        invalid = [k for k in target_qubits if not 0 <= k < num_qubits]
        assert not invalid, f"Target qubit(s) {invalid} are out of range [0, {num_qubits - 1}]."

        t_span = np.linspace(0, duration * dt_, duration + 1)

        drive_strength = 0.027235035038834040234 * theta  # 0.042780692999130815               # temporarily not high accuracy

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

        gaussian_signal = Signal(
            envelope=gaussian_envelope,
            carrier_freq=omega,
            phase=0.0
        )

        result = ham_solver.solve(
            t_span=t_span,
            y0=self.current_state,
            method='jax_odeint',
            signals=[gaussian_signal]
        )

        self.current_state = result.y[-1]

    # similarity of 1.0
    def rz(self, theta, target_qubits, plot_prob=False, plot_blochsphere=False):
        """
        Implements an RZ gate (rotation around Z-axis) by angle theta on all qubits in current_state.
        Uses a constant drive Hamiltonian with Z operator, with final state transformed to lab frame.
        """
        num_qubits = self.num_qubits

        if target_qubits == 'all':
            target_qubits = list(range(num_qubits))
        elif isinstance(target_qubits, int):
            target_qubits = [target_qubits]
        invalid = [k for k in target_qubits if not 0 <= k < num_qubits]
        assert not invalid, f"Target qubit(s) {invalid} are out of range [0, {num_qubits - 1}]."

        t_span = np.linspace(0, duration * dt_, duration + 1)
        t_max = t_span[-1]

        # drive_strength = (theta/2 - omega/2 * t + 2 * np.pi * k) / t  # t = 12

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
            y0=self.current_state,
            method='jax_odeint',
            signals=[signal]
        )

        self.current_state = result.y[-1]

    # works duration independent!
    def rx(self, theta, target_qubits='all', plot_prob=False, plot_blochsphere=False):
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
        num_qubits = self.num_qubits

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
            phase=0.0,  # 0.0
        )

        # print("drawing...")
        # # T = duration * dt_
        # gaussian_signal.draw(t0=0, tf=duration * dt_, n=1000, function="signal")
        # plt.show()

        result = ham_solver.solve(
            t_span=t_span,
            y0=self.current_state,
            method='jax_odeint',
            signals=[gaussian_signal]
        )

        # plot probs
        # pop = [np.abs(y.data[0]) ** 2 for y in result.y]
        # start = 50
        # stop = 58
        # plt.plot(t_span[start:stop], pop[start:stop])
        # plt.show()

        self.current_state = result.y[-1]
