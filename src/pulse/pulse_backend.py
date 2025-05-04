import jax
from scipy.integrate import quad

from pulse.operator import *

from utils.definitions import *

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')


# TODO test: Effect of Pulse-Level parameter on Fourier series (and like that on its coefficients)

# TODO plot the pulses themselves

# TODO: play with gate level parameter

# TODO find out if trajectory might be needed and return all together

# TODO plot probabilities, plot entanglement, plot blochsphere in lab frame

# TODO clean pulse parameter and isolate as far as possible in k, to minimize pulse parameters

# TODO use sparse pauli opt where possible to accelerate code


# maybe TODO mix X and Z rotation and plot, probably acceleration when mixing pulses for hadamard

# TODO pre calc common values like np.pi /2 for comp acceleration


class PulseBackend:

    def __init__(self, num_qubits, initial_state):
        self.num_qubits = num_qubits
        self.current_state = initial_state
        self.operator = PulseOperator(num_qubits)

    # fod 0.9998
    def h(self, target_qubits, correction=True, plot=False, bool_blochsphere=False):

        target_qubits = self.operator.verify_wires(target_qubits, "H")

        k = 0.042780586392198006 * np.pi

        H_static_single = static_hamiltonian(vu=nu)
        H_drive_single = drive_X_hamiltonian(drive_strength=k)

        H_static_multi = self.operator.parallel_hamiltonian("static", "all", H_static_single)

        print(H_static_single)
        print(H_static_multi)

        H_drive_multi = self.operator.parallel_hamiltonian("drive", target_qubits, H_drive_single)

        ham_solver = Solver(
            static_hamiltonian=H_static_multi,
            hamiltonian_operators=[H_drive_multi],
            rotating_frame=H_static_multi
        )

        gaussian_signal = Signal(
            envelope=gaussian_envelope,
            carrier_freq=nu,
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

        if correction:
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

            glob_phase = global_phase_correction(target_qubits)
            if glob_phase != 0.0:
                glob_hamiltonian = glob_phase * np.eye(2 ** self.num_qubits, dtype=complex)
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
        target_qubits = self.operator.verify_wires(target_qubits, "RX")

        # RX:
        integral, _ = quad(gaussian_envelope, t_span[0], t_span[-1])  # this equals out exactly the gauss deviations

        # strength_scale = 0.3183109217857033     # close to 1/pi
        drive_strength = theta / integral

        # drive_strength = drive_strength * strength_scale

        H_static_single = static_hamiltonian(vu=nu)
        H_static = self.operator.parallel_hamiltonian("static", "all", H_static_single)

        H_drive_X_single = drive_X_hamiltonian(drive_strength)
        H_drive = self.operator.parallel_hamiltonian("drive", target_qubits, H_drive_X_single)

        ham_solver = Solver(
            static_hamiltonian=H_static,
            hamiltonian_operators=[H_drive],
            rotating_frame=H_static
        )

        # tests
        # const = gaussian_envelope(T * dt)
        # print(T*dt)
        # print(const)

        gaussian_signal = Signal(
            envelope=gaussian_envelope,
            carrier_freq=nu,
            phase=0.0,
        )

        # print("drawing...")
        # gaussian_signal.draw(t0=0, tf=60, n=1000000, function="signal")
        # plt.axvline(x=12, color='red', linestyle='--')
        # plt.show()

        result = ham_solver.solve(
            t_span=t_span,
            y0=self.current_state,
            method='jax_odeint',
            signals=[gaussian_signal]
        )

        # Does not work for multi qubit
        # plot probs
        # pop = [np.abs(y.data[0]) ** 2 for y in result.y]
        # start = 50
        # stop = 58
        # plt.plot(t_span[start:stop], pop[start:stop])
        # plt.show()

        self.current_state = result.y[-1]

        # worst similarity at around 0.99

    def ry(self, theta, target_qubits='all', plot_prob=False, plot_blochsphere=False):
        target_qubits = self.operator.verify_wires(target_qubits, "RY")

        integral, _ = quad(gaussian_envelope, t_span[0], t_span[-1])
        drive_strength = theta / integral

        # Construct multi-qubit static Hamiltonian (applied to all qubits)
        H_static_single = static_hamiltonian(vu=nu)
        H_drive_Y_single = drive_Y_hamiltonian(drive_strength)

        H_static = self.operator.parallel_hamiltonian("static", "all", H_static_single)
        H_drive = self.operator.parallel_hamiltonian("drive", target_qubits, H_drive_Y_single)

        ham_solver = Solver(
            static_hamiltonian=H_static,
            hamiltonian_operators=[H_drive],
            rotating_frame=H_static
        )

        gaussian_signal = Signal(
            envelope=gaussian_envelope,
            carrier_freq=nu,
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
        target_qubits = self.operator.verify_wires(target_qubits, "RZ")

        # k = 0.04166666094977508 old k used for tuning
        drive_strength = theta / (2 * T * dt)

        # static
        H_static_single = static_hamiltonian(vu=nu)
        H_drive_Z_single = SIGMA_Z

        H_static = self.operator.parallel_hamiltonian("static", "all", H_static_single)
        H_drive = self.operator.parallel_hamiltonian("drive", target_qubits, H_drive_Z_single)

        ham_solver = Solver(
            static_hamiltonian=H_static,
            hamiltonian_operators=[H_drive],
            rotating_frame=H_static
        )

        signal = Signal(
            envelope=drive_strength,  # constant envelope
            carrier_freq=0.0,  # No oscillation for Z drive
            phase=0.0
        )

        # print("drawing...")
        # signal.draw(t0=0, tf=T / 2, n=100, function="signal")
        # plt.show()

        result = ham_solver.solve(
            t_span=t_span,
            y0=self.current_state,
            method='jax_odeint',
            signals=[signal]
        )

        self.current_state = result.y[-1]

    def cz(
            self,
            wires
    ):
        num_qubits = self.num_qubits

        control_qubit = wires[0]
        target_qubit = wires[1]

        if control_qubit == target_qubit or control_qubit >= num_qubits or target_qubit >= num_qubits:
            raise ValueError("Invalid control or target qubit indices")

        H_static_single = static_hamiltonian(nu)
        H_static_multi = self.operator.parallel_hamiltonian("static", "all", H_static_single)

        # (pi/4)(I ⊗ I - I ⊗ Z - Z ⊗ I + Z ⊗ Z)
        # diagonal_values = [0, 0, 0, np.pi]
        # matrix = np.diag(diagonal_values)
        # H_drive = np.diag([0, 0, 0, 0, 0, 0, np.pi, np.pi])

        H_drive = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex)  # TODO precompute

        indices = binary_c_t(num_qubits, control_qubit, target_qubit)
        H_drive[indices, indices] = np.pi

        ham_solver = Solver(
            static_hamiltonian=H_static_multi,
            hamiltonian_operators=[Operator(H_drive)],
            rotating_frame=H_static_multi
        )

        drive_strength = 1 / (T * dt)  # making time independence
        signal = Signal(
            envelope=drive_strength,
            carrier_freq=0.0,
            phase=0.0
        )

        result = ham_solver.solve(
            t_span=t_span,
            y0=self.current_state,
            method='jax_odeint',
            signals=[signal]
        )

        self.current_state = result.y[-1]

    def cnot(
            self,
            wires
    ):
        self.h([wires[1]])
        # self.ry(np.pi / 2, wires[1])
        self.cz(wires)
        # self.ry(np.pi / 2, wires[1])
        self.h([wires[1]])

        return self.current_state
