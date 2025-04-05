import qiskit_aer
from qiskit import QuantumCircuit

from src.utils.definitions import *


class QuickCircuit:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits

    def run_quick_circuit(self, initial_state):
        qc = QuantumCircuit(self.num_qubits)

        if initial_state is not None:
            qc.initialize(initial_state, range(self.num_qubits))  # Initialize with the custom state

        qc.h(0)
        # qc.rx(np.pi / 2, 0)
        # qc.rx(np.pi / 2, 1)
        # qc.cx(1, 0)
        simulator = 'statevector_simulator'
        backend = qiskit_aer.Aer.get_backend(simulator)
        result = backend.run(qc).result()
        statevector = result.get_statevector()
        return statevector


num_q = 2
init_state = GROUND_STATE(num_qubits=num_q)

qm = QuickCircuit(num_qubits=num_q)

statevector = qm.run_quick_circuit(init_state)

print(statevector)

