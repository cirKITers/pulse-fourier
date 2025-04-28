import numpy as np
from qiskit.quantum_info import Operator, partial_trace
from qiskit_dynamics import Solver, Signal

from pulse.envelope import *
from utils.definitions import I


# OPERATOR CONSTRUCTION

class PulseOperator:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits

    def parallel_hamiltonian(self, hamiltonian_type, target_qubits, single_hamiltonian):
        # Construct multi-qubit static Hamiltonian (applied to all qubits)
        if hamiltonian_type == 'static':
            return self.sum_operator(single_hamiltonian, "all")

        # Construct drive Hamiltonian (applied only to target_qubits)
        if hamiltonian_type == 'drive':
            return self.sum_operator(single_hamiltonian, target_qubits)

    # TODO sources
    def sum_operator(self, operator, target_qubits):
        """Creates a sum of the operator applied to each qubit in target."""
        total = Operator(np.zeros((2 ** self.num_qubits, 2 ** self.num_qubits), dtype=complex))
        if target_qubits == "all":
            target_qubits = range(self.num_qubits)
        for q in target_qubits:
            total += self.operator_on_qubit(operator, q)

        return total

    def operator_on_qubit(self, operator, qubit_index):
        """Places an operator on the specified qubit with identities on others."""
        ops = [I] * self.num_qubits
        ops[qubit_index] = operator
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return Operator(result)

    def verify_wires(self, target_qubits, gate_name):
        """Verifies if the target qubits are valid for the given gate."""
        if target_qubits == 'all':
            return list(range(self.num_qubits))
        elif isinstance(target_qubits, int):
            return [target_qubits]
        elif isinstance(target_qubits, list):
            invalid = [k for k in target_qubits if not 0 <= k < self.num_qubits]
            assert not invalid, f"Target qubit(s) {invalid} for gate '{gate_name}' are out of range [0, {self.num_qubits - 1}]."
            return target_qubits
        else:
            raise ValueError(f"Invalid target_qubits specification for gate '{gate_name}'. Must be 'all', an integer, or a list of integers.")

    # def solve_dynamics(self, static_hamiltonian, drive_hamiltonian, signal, y0):
    #     ham_solver = Solver(
    #         static_hamiltonian=static_hamiltonian,
    #         hamiltonian_operators=[drive_hamiltonian],
    #         rotating_frame=static_hamiltonian
    #     )
    #
    #     gaussian_signal = Signal(
    #         envelope=gaussian_envelope,
    #         carrier_freq=omega,  # No oscillation for Z drive
    #         phase=0.0,  # 0.0
    #     )
    #     result = ham_solver.solve(
    #         t_span=t_span,
    #         y0=y0,
    #         method='jax_odeint',
    #         signals=[gaussian_signal]
    #     )






