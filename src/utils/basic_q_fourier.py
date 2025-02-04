import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
import qiskit_aer
from qiskit.circuit import Parameter
matplotlib.use('TkAgg')  # more general backend, works for PyCharm

# Only one qubit possible sor far for simplification purposes
class BasicQFourier1:
    def __init__(self, num_qubits, num_layer, parameter):
        self.num_q = num_qubits
        self.L = num_layer
        self.params = parameter
        self.param_label = [Parameter(f'theta_{i}') for i in range(self.L)]

    @staticmethod
    def data_encoding(qc, qubit, x):
        qc.rx(2 * np.pi * x, qubit)

    @staticmethod
    def trainable_block(qc, qubit, theta):
        qc.rz(theta, qubit)

    def define_circuit(self, x):
        qc = QuantumCircuit(self.num_q)
        for lay in range(self.L):
            self.data_encoding(qc, 0, x)
            self.trainable_block(qc, 0, self.param_label[lay])
        for i, theta in enumerate(self.param_label):
            qc = qc.assign_parameters({theta: self.params[i % len(self.params)]})
        return qc

    @staticmethod
    def compute_expectations(qc, simulator, shots):
        backend = qiskit_aer.Aer.get_backend(simulator)
        if simulator == 'qasm_simulator':
            qc.measure_all()
            result = backend.run(qc, shots=shots).result()
            counts = result.get_counts()
            probability_0 = counts.get('0', 0) / shots   # Probability of measuring |0> state - expectation value
        else:   # if simulator == 'statevector_simulator':
            result = backend.run(qc).result()
            statevector = result.get_statevector()
            probability_0 = abs(statevector.data[0]) ** 2
        return 2 * probability_0 - 1  # Map to [-1, 1]

    def predict_interval(self, simulator, shots, interval, points, plot=True):
        x_values = np.linspace(interval[0], interval[1], points, endpoint=True)  # Set False to avoid duplicate samples at start-end of period
        f_x_values = np.zeros(points)
        for i, x in enumerate(x_values):
            qc = self.define_circuit(x)
            f_x = self.compute_expectations(qc, simulator, shots)
            f_x_values[i] = f_x

        if plot:
            plt.figure(figsize=(8, 6))
            plt.plot(x_values, f_x_values, label="Quantum Model Prediction", color='b')
            plt.title("Quantum Model Prediction Function")
            plt.xlabel("Input x")
            plt.ylabel("Predicted f(x)")
            plt.grid(True)
            plt.legend()
            plt.show()

        return x_values, f_x_values

    def predict_single(self, simulator, shots, x):
        qc = self.define_circuit(x)
        f_x = self.compute_expectations(qc, simulator, shots)
        return f_x




