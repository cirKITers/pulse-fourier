from qiskit import QuantumCircuit, transpile, assemble
import qiskit_aer
from qiskit.circuit.library import HGate, RXGate
from qiskit.quantum_info import Statevector

from src.utils.helpers import prob
from src.utils.visualize.bloch_sphere import *

def q_circuit(qubits):
    circuit = QuantumCircuit(qubits)
    simulator = 'statevector_simulator'
    backend = qiskit_aer.Aer.get_backend(simulator)
    return backend, circuit


# theta = np.pi / 2
# init_state = EXCITED_STATE

def run_basic_model(backend, circuit, init_state, theta, plot_bloch=False):
    def basic_model(qc):
        qc.rz(theta, 0)
        return qc

    circuit.initialize(init_state, 0)
    result = backend.run(basic_model(circuit)).result()
    result_vector = Statevector(result.get_statevector())
    probability = prob(result_vector)
    if plot_bloch:
        plot_bloch_sphere([init_state, result_vector])

    return result_vector, probability

# Check correctness
# expected_vector = Statevector(init_state).evolve(RXGate(theta))
# if np.allclose(result_vector.data, expected_vector.data):
#     print("match")
# else:
#     print("mismatch")
