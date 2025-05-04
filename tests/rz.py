import numpy as np
from qiskit.quantum_info import Statevector
import pennylane as qml

from pulse.pulse_backend import RZ_pulseSPEC
from utils.helpers import random_theta, prints, overlap_components, statevector_fidelity


# ALL PASSED, two qubit random theta

class PennyCircuit:

    def __init__(self, num_qubits):

        self.num_qubits = num_qubits

    def run_quick_circuit(self, theta, init_state=None):
        dev = qml.device('default.qubit', wires=self.num_qubits)

        @qml.qnode(dev)
        def general_circuit():
            if init_state is not None:
                qml.StatePrep(init_state, wires=range(self.num_qubits))
            # qml.CNOT(wires=[0, 1])  # big endian
            qml.RX(np.pi/2, 0)
            qml.RX(np.pi/2, 1)
            qml.RZ(theta, 0)
            qml.RZ(theta, 1)
            return qml.state()

        return general_circuit()


num_q = 2
c = PennyCircuit(num_q)


tries = 1



for t in range(1):

    for i in range(tries):

        theta = random_theta()

        penny_state = c.run_quick_circuit(theta)

        _, _, current_state = RZ_pulseSPEC(theta, Statevector([0.50000000 + 0.00000000j, 0.00000000 - 0.50000000j, 0.00000000 - 0.50000000j, -0.50000000 + 0.00000000j]), "all")

        sim = overlap_components(current_state[-1], penny_state)
        fid = statevector_fidelity(current_state[-1], penny_state)
        if sim > 0.99:

            print(f"works for theta = {theta}")
            print(f"similarity: {sim}")
            print(f"fidelity: {fid}")
            prints(penny_state)
            prints(current_state[-1])
            print("\n")

        else:
            print(f"############################################################################DID NOT work for {theta}")
            print(sim)
            prints(penny_state)
            prints(current_state[-1])
            print("\n")