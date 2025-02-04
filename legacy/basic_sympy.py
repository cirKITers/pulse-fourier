from qiskit import QuantumCircuit
from sympy import symbols
from qiskit.quantum_info import Statevector

theta1, theta2, theta3, theta4 = symbols('θ1 θ2 θ3 θ4')

qc = QuantumCircuit(2)

qc.rx(theta1, 0)
qc.ry(theta2, 1)
qc.rx(theta3, 0)
qc.ry(theta4, 1)

statevector = Statevector.from_instruction(qc)

print("Symbolic Statevector:")
print(statevector)
