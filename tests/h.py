import numpy as np
import pennylane as qml

from pulse.pulse_gates import H_pulseSPEC, H_pulseSPEC2
from utils.definitions import GROUND_STATE, EXCITED_STATE, SUPERPOSITION_STATE_H, RANDOM_STATE, PHASESHIFTED_STATE
from utils.helpers import prints, statevector_similarity, statevector_fidelity


class PennyCircuit:

    def __init__(self, num_qubits):

        self.num_qubits = num_qubits

    def run_quick_circuit(self, init_state=None):
        dev = qml.device('default.qubit', wires=self.num_qubits)

        @qml.qnode(dev)
        def general_circuit():
            if init_state is not None:
                qml.StatePrep(init_state, wires=range(self.num_qubits))

            qml.Hadamard(0)
            qml.Hadamard(1)
            # qml.Hadamard(2)
            # qml.Hadamard(3)
            # qml.CNOT(wires=[0, 1])  # big endian
            # qml.RX(np.pi/2, 0)
            # qml.RX(np.pi/2, 1)
            # qml.CZ([0, 1])
            # qml.RZ(theta, 0)
            # qml.RZ(theta, 1)
            return qml.state()

        return general_circuit()


num_q = 2
c = PennyCircuit(num_q)



# fidelity 0, means orthogonal

for i in range(1):
    init = RANDOM_STATE(num_q)

    penny_state = c.run_quick_circuit(init.data)
    prints(penny_state)

    _, _, current_state = H_pulseSPEC(init, [0, 1], -np.pi/2, 0.0, 0.0)
    # _, _, current_state = H_pulseSPEC2(init, [0, 1, 3], 0.0, 0.0, 0.0)
    prints(current_state[-1])


    sim = statevector_similarity(penny_state, current_state[-1])
    fid = statevector_fidelity(penny_state, current_state[-1].data)
    print(f"sim = {sim}, fid = {fid}")


