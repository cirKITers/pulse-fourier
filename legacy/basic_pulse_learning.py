import numpy as np
import jax.numpy as jnp
from qiskit_dynamics import Solver, Signal
from qiskit.quantum_info import Statevector, Operator
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Force TkAgg backend


# drive_strength = 0.02086
def h_pulse(drive_strength, sigma, plot):

    final_probs = np.zeros(2)

    # Step 1: Define the qubit Hamiltonian

    # OMEGA:
    # Typical value for superconducting qubits, natural frequency in GHz
    # How fast spins between |0> and |1> without push, defines natural behavior in H_static
    omega = 5.0

    # DRIVE STRENGTH:
    # Found through Trial and Error, multiplies pulse's effect in H_drive
    drive_strength = drive_strength

    # Pauli X Operator: How to flip qubit's phase around X-Axis, used in H_drive
    # Flips states: ∣0⟩=[1,0]→[0,1]=∣1⟩
    #           and ∣1⟩=[0,1]→[1,0]=∣0⟩
    # For smaller tilts (like π/2), it mixes them (e.g., ∣0⟩→[0.707,0.707])
    sigma_x = Operator(np.array([[0, 1], [1, 0]]))

    # Pauli Z Operator: How to flip qubit's phase around Z-Axis, used in H_static
    # ∣0⟩=[1,0]→[1,0]=∣0⟩ (no change)
    # ∣1⟩=[0,1]→[0,−1]=−∣1⟩ (phase flip)
    sigma_z = Operator(np.array([[1, 0], [0, -1]]))

    # Static Hamiltonian (σz term), natural rulebook for how it spins on its own without pulse
    # Used in solver
    H_static = 2 * np.pi * omega * sigma_z / 2

    # Drive operator (X-control term), the controlled push scaled by drive_strength
    H_drive = 2 * np.pi * drive_strength * sigma_x

    # Step 2: Define Gaussian envelope as a Signal
    # GAUSSIAN PULSE VARIABLES
    duration = 120  # Number of time steps (samples)
    sigma = sigma    # Standard deviation, how wide or narrow the gaussian bell
    amp = 1.0      # Amplitude, height gaussian bell at peak, default Max
    dt = 0.1       # Time step in ns
    t_span = np.linspace(0, duration * dt, duration + 1)   # Tells solver when to check the qubits state

    # Gaussian signal
    def gaussian_envelope(t):
        center = duration * dt / 2
        return amp * jnp.exp(-((t - center) ** 2) / (2 * sigma ** 2))

    gaussian_signal = Signal(
        envelope=gaussian_envelope,
        carrier_freq=omega  # Resonant with qubit frequency
    )
    # Step 3: Solver
    solver = Solver(
        static_hamiltonian=H_static,
        hamiltonian_operators=[H_drive],
        rotating_frame=H_static
    )

    # Step 4: Define initial state
    initial_state = Statevector([1.0, 0.0])  # |0⟩ state

    # Step 5: Solve the dynamics with the signal
    result = solver.solve(
        t_span=t_span,
        y0=initial_state,
        method='jax_odeint',
        signals=[gaussian_signal]
    )

    # Step 6: Final state analysis
    final_state = result.y[-1]
    # print("Final state:", final_state)

    # Plot state evolution
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

    expected_state = np.array([1, 1]) / np.sqrt(2)

    # squared overlap or fidelity between two quantum states
    overlap = np.abs(np.vdot(expected_state, final_state)) ** 2
    # print(f"Overlap with (|0⟩ + |1⟩)/√2: {overlap:.4f}")

    state_probs = np.abs(result.y) ** 2
    final_probs[0] = state_probs[-1, 0]  # P(|0⟩)
    final_probs[1] = state_probs[-1, 1]

    return drive_strength, sigma, final_probs[0] - final_probs[1], overlap, final_state


# # Noise parameters
# T1 = 50e3  # 50 µs (energy relaxation time, in ns)
# T2 = 40e3  # 40 µs (dephasing time, in ns)
# noise_scale = 0.05  # 5% amplitude noise
#
# # Decoherence operators
# gamma_1 = 1 / T1  # Relaxation rate
# gamma_2 = 1 / T2  # Dephasing rate
# L1 = Operator([[0, 1], [0, 0]])  # Collapse operator for T1 (σ-)
# L2 = sigma_z  # Dephasing operator (σz)
