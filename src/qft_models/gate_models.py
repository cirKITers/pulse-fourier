from qiskit import QuantumCircuit
import qiskit_aer

from visuals.bloch_sphere import *

def q_circuit(qubits):
    circuit = QuantumCircuit(qubits)
    simulator = 'statevector_simulator'
    backend = qiskit_aer.Aer.get_backend(simulator)
    return backend, circuit


# theta = np.pi / 2
# init_state = EXCITED_STATE

def run_basic_model(backend, circuit, init_state, theta, plot_bloch=False):
    def basic_model(qc):
        qc.h(0)
        return qc

    circuit.initialize(init_state, 0)
    result = backend.run(basic_model(circuit)).result()
    result_vector = Statevector(result.get_statevector())
    probability = prob(result_vector)
    if plot_bloch:
        bloch_sphere_trajectory([init_state, result_vector])

    return result_vector, probability

# Check correctness
# expected_vector = Statevector(init_state).evolve(RXGate(theta))
# if np.allclose(result_vector.data, expected_vector.data):
#     print("match")
# else:
#     print("mismatch")
