from qiskit.quantum_info import Operator
from qiskit_dynamics import Signal, Solver

from constants import *
from utils.definitions import *

# Integration of Lindblad equation due to supply of dissipators in solve function


drive_strength = 0.04
gamma_amp = 0.01  # Amplitude damping rate
gamma_phase = 0.02  # Phase damping rate
amp_dissipator = np.sqrt(gamma_amp) * SIGMA_MINUS
phase_dissipator = np.sqrt(gamma_phase) * SIGMA_Z

hamiltonian_solver = Solver(
    static_hamiltonian=Operator(omega * SIGMA_Z),
    hamiltonian_operators=[Operator(drive_strength * SIGMA_X)],
    dissipator_operators=[amp_dissipator, phase_dissipator],
)

gauss_signal = Signal(envelope=lambda t: 1.0, carrier_freq=omega)


t_span = [0.0, 10.0]
y0 = np.array([[1, 0], [0, 0]], dtype=complex)

results = hamiltonian_solver.solve(t_span=t_span, y0=y0, signals=([gauss_signal], [Signal(1.0), Signal(1.0)]))

print(results.y[-1])
