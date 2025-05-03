import numpy as np
import pennylane as qml

from pulse.pulse_system import PulseSystem
from utils.helpers import normalized_ground_state_prob


class Circuit15:

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits

    def run(self, x, params, draw=False):

        dev = qml.device("default.qubit", wires=self.num_qubits)

        def Ansatz(theta):

            for i in range(self.num_qubits):
                qml.RY(theta[i], wires=i)

            for i in range(self.num_qubits):
                control_qubit = i
                target_qubit = (i + 1) % self.num_qubits
                qml.CNOT(wires=[control_qubit, target_qubit])

            for i in range(self.num_qubits):
                qml.RY(theta[self.num_qubits + i], wires=i)

            for i in range(self.num_qubits):
                control_qubit = i
                target_qubit = (i + 1) % self.num_qubits
                qml.CNOT(wires=[control_qubit, target_qubit])

        def Encoding(feature):
            for i in range(self.num_qubits):
                qml.RX(feature, wires=i)

        @qml.qnode(dev)
        def circuit():

            Ansatz(theta=params[0])  # 2*num_qubits

            Encoding(x)

            Ansatz(theta=params[1])  # 2*num_qubits

            return qml.state()

        if draw:
            print(qml.draw(circuit)())
        return circuit()

    def sample_fourier(self, x, parameter_set, num_samples):
        fx_set = []
        for sample in range(num_samples):
            # Make fourier series for this sample
            fx = []
            for x_val in x:
                feature = np.array([x_val] * self.num_qubits)
                param = parameter_set[sample]
                final_state = self.run(feature, param, draw=True)
                fx_val = normalized_ground_state_prob(final_state)
                fx.append(fx_val)

            fx_set.append(np.array(fx))
        return fx_set
