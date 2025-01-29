from qiskit import QuantumCircuit, transpile
import qiskit_aer
from qiskit.circuit import Parameter
from qiskit.visualization import plot_histogram
import numpy as np

from helpers import prob

simulation_type = 'qasm'    # qasm or state

theta1 = Parameter('θ1')
theta2 = Parameter('θ2')
theta3 = Parameter('θ3')
theta4 = Parameter('θ4')

# QC
qc = QuantumCircuit(2)
qc.rx(theta1, 0)
qc.ry(theta2, 1)
qc.rx(theta3, 0)
qc.ry(theta4, 1)
qc.cx(0, 1)
print(qc)
qc = qc.assign_parameters({theta1: 1.23, theta2: 2.34, theta3: 3.45, theta4: 4.56})

if simulation_type == 'state':
    # Statevector output
    simulator = qiskit_aer.Aer.get_backend('statevector_simulator')
    result = simulator.run(qc).result()
    statevector = result.get_statevector()
    print("Statevector: ", statevector)
    probability = prob(statevector)
    print("Probability: ", probability)

if simulation_type == 'qasm':
    # Qasm probability
    qc.measure_all()
    simulator = qiskit_aer.Aer.get_backend('qasm_simulator')
    job = simulator.run(qc, shots=6)
    result = job.result()
    counts = result.get_counts()
    print("Measurement results:", counts)
