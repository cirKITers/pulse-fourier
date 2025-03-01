import numpy as np
import jax.numpy as jnp
from qiskit_dynamics import Solver, Signal
from qiskit.quantum_info import Statevector, Operator
import matplotlib.pyplot as plt

# print("Qiskit Dynamics version:", qiskit_dynamics.__version__)

# Qubit setup
omega = 5.0  # GHz
sigma_x = Operator([[0, 1], [1, 0]])  # X-axis tool
sigma_z = Operator([[1, 0], [0, -1]])  # Z-axis tool
H_static = 2 * np.pi * omega * sigma_z / 2  # Natural spin

# Pulse parameters
dt = 0.1  # ns per step
sigma = 40
amp = 1.0

# Define Gaussian envelope
def gaussian_envelope(t, center):
    return amp * jnp.exp(-((t - center) ** 2) / (2 * sigma ** 2))


# Gate 1: RX(π/2) ~ Hadamard-like (4 ns)
rx_duration = 40  # 4 ns = 40 * 0.1
rx_drive_strength = 0.0625  # TUNABLE
rx_center = rx_duration * dt / 2  # 2 ns
rx_signal = Signal(
    envelope=lambda t: gaussian_envelope(t, rx_center),
    carrier_freq=omega
)
H_rx = 2 * np.pi * rx_drive_strength * sigma_x
t_span_rx = np.linspace(0, rx_duration * dt, rx_duration + 1)  # 0 to 4 ns

# Gate 2: RZ(π) (2 ns)
rz_duration = 20  # 2 ns = 20 * 0.1
rz_drive_strength = 0.1  # Adjust for π rotation
rz_center = rx_duration * dt + rz_duration * dt / 2  # Center at 5 ns
rz_signal = Signal(
    envelope=lambda t: gaussian_envelope(t, rz_center),
    carrier_freq=0  # Z doesn’t need frequency
)
H_rz = 2 * np.pi * rz_drive_strength * sigma_z
t_span_rz = np.linspace(rx_duration * dt, (rx_duration + rz_duration) * dt, rz_duration + 1)  # 4 to 6 ns

# Solver with initial operators (we’ll update later)
solver = Solver(
    static_hamiltonian=H_static,
    hamiltonian_operators=[H_rx],  # Start with RX operator
    rotating_frame=H_static
)

# Initial state
state = Statevector([1.0, 0.0])

# Apply RX(π/2)
result_rx = solver.solve(
    t_span=t_span_rx,
    y0=state,
    method='jax_odeint',
    signals=[rx_signal]  # Matches H_rx
)
state = result_rx.y[-1]  # Update state after RX
print("After RX(π/2) at 4 ns:", state)

# Update solver for RZ(π)
solver.hamiltonian_operators = [H_rz]  # Switch to RZ operator
result_rz = solver.solve(
    t_span=t_span_rz,
    y0=state,
    method='jax_odeint',
    signals=[rz_signal]  # Matches H_rz
)
final_state = result_rz.y[-1]
print("After RZ(π) at 6 ns:", final_state)

# Combine time spans and states for plotting
t_span_full = np.concatenate([t_span_rx, t_span_rz[1:]])  # 0 to 6 ns
state_probs = np.abs(np.concatenate([result_rx.y, result_rz.y[1:]])) ** 2

plt.figure(figsize=(12, 6))
plt.plot(t_span_full, state_probs[:, 0], label="P(|0⟩)")
plt.plot(t_span_full, state_probs[:, 1], label="P(|1⟩)")
plt.axvline(x=4, color='red', linestyle='--', label='RX end (4 ns)')
plt.axvline(x=6, color='green', linestyle='--', label='RZ end (6 ns)')
plt.xlabel("Time (ns)")
plt.ylabel("Probability")
plt.title("Pulse Sequence: RX(π/2) then RZ(π)")
plt.legend()
plt.grid()
plt.savefig('pulse_sequence.png')
print("Plot saved as 'pulse_sequence.png'")

# Check RX(π/2)
expected_hadamard = np.array([1, 1]) / np.sqrt(2)
overlap_rx = np.abs(np.vdot(expected_hadamard, result_rx.y[-1])) ** 2
print(f"Overlap with (|0⟩ + |1⟩)/√2 after RX: {overlap_rx:.3f}")
