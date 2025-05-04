import numpy as np
import pennylane as qml

from pulse.pulse_backend import PulseBackend
from utils.helpers import normalized_ground_state_prob


# with num_qubits > 2 the states become entangled
class PennyHubregtsen:

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits

    def run(self, features, trainable_params, init_state=None, draw=False):
        """Runs the quantum circuit with the specified number of layers.

        Args:
            trainable_params (np.ndarray): Trainable parameters for the circuit
                with shape (num_layers, num_qubits, 1).
            features (np.ndarray): Input features for each qubit, with shape (num_qubits,).
            init_state (np.ndarray, optional): Initial state of the qubits. Defaults to None.

        Returns:
            np.ndarray: The final state of the quantum system.
        """
        num_layers = trainable_params.shape[0]
        assert features.shape == (self.num_qubits,), "Number of features must match number of qubits."
        assert trainable_params.shape == (num_layers, self.num_qubits, 1), \
            "Trainable parameters must have shape (num_layers, num_qubits, 1)."

        dev = qml.device('default.qubit', wires=self.num_qubits)

        @qml.qnode(dev)
        def circuit():
            if init_state is not None:
                qml.StatePrep(init_state, wires=range(self.num_qubits))

            for layer_idx in range(num_layers):
                for q in range(self.num_qubits):
                    qml.Hadamard(wires=q)
                    qml.RZ(features[q], wires=q)
                    qml.RY(trainable_params[layer_idx, q, 0], wires=q)

                # Ring of controlled Pauli-Z rotations
                for i in range(self.num_qubits):
                    control_qubit = i
                    target_qubit = (i + 1) % self.num_qubits
                    qml.CZ(wires=[control_qubit, target_qubit])

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


class PulseHubregtsen:

    def __init__(self, num_qubits, initial_state):
        self.num_qubits = num_qubits
        self.pulse_system = PulseBackend(num_qubits, initial_state)

    def one_layer(self, features, trainable_params):
        """Implements one layer of the quantum circuit using pulse-level control."""
        assert len(features) == self.num_qubits, "Number of features must match number of qubits."
        assert len(trainable_params) == self.num_qubits, "Number of trainable parameters must match number of qubits."

        # Layer of Hadamard gates
        for q in range(self.num_qubits):
            self.pulse_system.h([q])

        # Layer of single-qubit Pauli-Z rotations (embedding features)
        for q in range(self.num_qubits):
            self.pulse_system.rz(features[q], [q])

        # Layer of trainable single-qubit Pauli-Y rotations
        for q in range(self.num_qubits):
            self.pulse_system.ry(trainable_params[q], [q])

        # Ring of controlled Pauli-Z rotations
        for i in range(self.num_qubits):
            control_qubit = i
            target_qubit = (i + 1) % self.num_qubits
            self.pulse_system.cz([control_qubit, target_qubit])

    def run_layered_circuit(self, features, trainable_params_list):
        """Runs the quantum circuit with multiple layers using pulse-level control."""
        num_layers = len(trainable_params_list)
        assert num_layers > 0, "At least one layer of trainable parameters must be provided."
        assert all(len(params) == self.num_qubits for params in trainable_params_list), \
            "Each layer must have a set of trainable parameters equal to the number of qubits."
        assert len(features) == self.num_qubits, "Number of features must match number of qubits."

        current_state = self.pulse_system.current_state  # Start with the initial state

        for layer_idx in range(num_layers):
            self.one_layer(features, trainable_params_list[layer_idx])

        return self.pulse_system.current_state
