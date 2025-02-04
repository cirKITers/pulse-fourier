import numpy as np
from qiskit import QuantumCircuit
import qiskit_aer
from qiskit.circuit import Parameter
from qiskit.visualization import plot_histogram


def classical_function(x):
    return 2 * x + 1

n_qubits = 1

qc = QuantumCircuit(n_qubits)

theta = Parameter('θ')

qc.rx(theta, 0)

qc.measure_all()

backend = qiskit_aer.Aer.get_backend('qasm_simulator')

x_input = 1.0

compiled_qc = qc.assign_parameters({theta: x_input})

shots = 1024
result = backend.run(compiled_qc, shots=shots).result()

counts = result.get_counts()

plot_histogram(counts)

measured_value = sum([int(bit, 2) * counts[bit] for bit in counts]) / shots

print(f"Measured value: {measured_value}")
print(f"Classical function value: {classical_function(x_input)}")
