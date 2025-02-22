import numpy as np
import jax.numpy as jnp
from qiskit_dynamics import Solver, Signal
from qiskit.quantum_info import Statevector, Operator
import matplotlib.pyplot as plt
import matplotlib
from utils.helpers import *
from utils.visualize.bloch_sphere import *
matplotlib.use('TkAgg')  # Force TkAgg backend

# Define Pauli matrices
sigma_x = SIGMA_X
sigma_y = SIGMA_Y
sigma_z = SIGMA_Z

# Qubit parameters
omega_q = 5.0  # Qubit frequency in GHz (example value)
omega_d = omega_q  # Drive frequency (on-resonance)

# Pulse parameters for Hadamard (pi/2 rotation around y-axis)
T = 1.0  # Total gate time in ns (arbitrary, adjust based on Rabi freq)
rabi_freq = np.pi / (2 * T)  # Rabi frequency to achieve pi/2 rotation
sigma = T / 4  # Gaussian width (adjustable)
dt = T / 99

# Define the pulse envelope (Gaussian)
def gaussian_envelope(t):
    return rabi_freq * jnp.exp(-(t - T/2)**2 / (2 * sigma**2))


# Define the signal for the y-drive
drive_signal = Signal(
    envelope=gaussian_envelope,
    carrier_freq=omega_d,
    phase=np.pi/2  # y-axis rotation
)

# Define the Hamiltonian
# Static part: H_0 = -omega_q/2 * sigma_z
# Drive part: H_d = gaussian(t) * cos(omega_d t + phase) * sigma_y
H_0 = -0.5 * omega_q * sigma_z
H_d_operators = [sigma_y]

# Set up the solver
solver = Solver(
    static_hamiltonian=H_0,
    hamiltonian_operators=H_d_operators,
    rotating_frame=H_0,
    array_library='jax'
)


initial_state = INIT_STATE

# Time points for simulation
t_vals = np.linspace(0, T, 100)

# Solve the dynamics
solution = solver.solve(
    t_span=np.linspace(0, T),
    y0=INIT_STATE,
    method='jax_odeint',
    signals=[drive_signal]
)

# Extract states and compute Bloch coordinates
states = solution.y

plot_bloch_sphere(states)
