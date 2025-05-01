import numpy as np
import pennylane as qml

from pulse.pulse_system import PulseSystem
from utils.helpers import normalized_ground_state_prob

class Circuit10:

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits

    def run(self, features, trainable_params, init_state=None, draw=False):
        """Runs the custom PennyLane quantum circuit based on the diagram structure.

        Args:
            trainable_params (np.ndarray): Shape (num_layers, num_qubits, 2)
            features (np.ndarray): Shape (num_qubits,)
            init_state (np.ndarray, optional): Optional initial state vector.
            draw (bool): If True, prints the circuit.

        Returns:
            np.ndarray: Final state of the quantum system.
        """
        num_layers = trainable_params.shape[0]
        assert features.shape == (self.num_qubits,), "Number of features must match number of qubits."
        assert trainable_params.shape == (num_layers, self.num_qubits, 2), \
            "Trainable parameters must have shape (num_layers, num_qubits, 2)."

        dev = qml.device('default.qubit', wires=self.num_qubits)

        @qml.qnode(dev)
        def circuit():
            if init_state is not None:
                qml.StatePrep(init_state, wires=range(self.num_qubits))

            for layer_idx in range(num_layers):
                # Apply RY rotations using trainable parameters
                for q in range(self.num_qubits):
                    qml.RY(trainable_params[layer_idx, q, 0], wires=q)

                # Fully connected ring
                for i in range(self.num_qubits):
                    qml.CZ(wires=[i, (i + 1) % self.num_qubits])

                for q in range(self.num_qubits):
                    qml.RY(features[q], wires=q)

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
                final_state = self.run(feature, param, draw=False)
                fx_val = normalized_ground_state_prob(final_state)
                fx.append(fx_val)

            fx_set.append(np.array(fx))
        return fx_set

