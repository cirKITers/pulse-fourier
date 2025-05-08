import numpy as np
import pennylane as qml

from utils.helpers import normalized_ground_state_prob, prob, ground_state_prob


class Circuit9:

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits

    def run(self, x, params, draw=False):

        dev = qml.device("default.qubit", wires=self.num_qubits)

        def Ansatz(theta):

            for i in range(self.num_qubits):
                qml.Hadamard(wires=i)

            for i in range(self.num_qubits - 1):
                control_qubit = i
                target_qubit = i + 1
                qml.CZ(wires=[control_qubit, target_qubit])
            #
            for i in range(self.num_qubits):
                qml.RX(theta[i], wires=i)

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
            # Print progress every 500 samples
            if (sample + 1) % 500 == 0:
                print(f"Processed sample: {sample + 1} / {num_samples}")

            # Make fourier series for this sample
            fx = []
            for x_val in x:

                feature = x_val

                param = parameter_set[sample]

                final_state = self.run(feature, param, draw=False)

                fx_val = normalized_ground_state_prob(final_state)

                fx.append(fx_val)

            fx_set.append(np.array(fx))

        return fx_set

