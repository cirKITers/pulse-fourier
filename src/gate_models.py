from qiskit import QuantumCircuit, transpile, assemble
import qiskit_aer
from qiskit.circuit.library import HGate, RXGate
from qiskit.quantum_info import Statevector

from src.utils.helpers import prob
from src.utils.visualize.bloch_sphere import *

circuit = QuantumCircuit(1)
simulator = 'statevector_simulator'
backend = qiskit_aer.Aer.get_backend(simulator)

theta = np.pi / 2
init_state = GROUND_STATE

def basic_model(qc):
    qc.rz(theta, 0)
    return qc


circuit.initialize(init_state, 0)
result = backend.run(basic_model(circuit)).result()
result_vector = Statevector(result.get_statevector())
probability = prob(result_vector)

plot_bloch_sphere([init_state, result_vector])
print("result_vector", result_vector.data)
print("probability", probability)

# Check correctness
# expected_vector = Statevector(init_state).evolve(RXGate(theta))
# if np.allclose(result_vector.data, expected_vector.data):
#     print("match")
# else:
#     print("mismatch")
