import pennylane as qml

from pulse.pulse_gates import H_pulseSPEC
from utils.definitions import GROUND_STATE
from utils.helpers import prints, statevector_similarity


class PennyCircuit:

    def __init__(self, num_qubits):

        self.num_qubits = num_qubits

    def run_quick_circuit(self, init_state=None):
        dev = qml.device('default.qubit', wires=self.num_qubits)

        @qml.qnode(dev)
        def general_circuit():
            if init_state is not None:
                qml.StatePrep(init_state, wires=range(self.num_qubits))

            qml.Hadamard(1)
            # qml.Hadamard(1)
            # qml.Hadamard(2)
            # qml.CNOT(wires=[0, 1])  # big endian
            # qml.RX(np.pi/2, 0)
            # qml.RX(np.pi/2, 1)
            # qml.CZ([0, 1])
            # qml.RZ(theta, 0)
            # qml.RZ(theta, 1)
            return qml.state()

        return general_circuit()


num_q = 10
c = PennyCircuit(num_q)

penny_state = c.run_quick_circuit()
prints(penny_state)

_, _, current_state = H_pulseSPEC(GROUND_STATE(num_q), 1)
prints(current_state[-1])

print(statevector_similarity(penny_state, current_state[-1]))

