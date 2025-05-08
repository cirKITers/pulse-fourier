import numpy as np

from pulse.pulse_backend import PulseBackend
from utils.definitions import GROUND_STATE
from utils.helpers import normalized_ground_state_prob, ground_state_prob


class PulseHE:

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits

    def run(self, x, params, draw=False):

        pls = PulseBackend(self.num_qubits, GROUND_STATE(self.num_qubits))

        def Ansatz(theta):

            for i in range(self.num_qubits):
                pls.ry(theta[i], i)

            for i in range(self.num_qubits):
                pls.rz(theta[self.num_qubits + i], i)

            for i in range(self.num_qubits):
                pls.ry(theta[self.num_qubits*2 + i], i)

            for i in range(self.num_qubits):
                control_qubit = i
                target_qubit = (i + 1) % self.num_qubits
                pls.cnot(wires=[control_qubit, target_qubit])

        def Encoding(feature):
            for i in range(self.num_qubits):
                pls.rx(feature, i)

        def circuit():

            Ansatz(theta=params[0])  # 2*num_qubits

            Encoding(x)

            Ansatz(theta=params[1])  # 2*num_qubits

            return pls.current_state

        if draw:
            print("no drawing on pulse level.")
        return circuit()

    def sample_fourier(self, x, parameter_set, num_samples):
        # print("Starting Pulse HEA eval...")

        fx_set = []
        for sample in range(num_samples):

            # Print progress every 100 samples
            if (sample + 1) % 100 == 0:
                print(f"Processed sample: {sample + 1} / {num_samples}")

            # Make fourier series for this sample
            fx = []
            for x_val in x:

                # print("discrete point:", x_val, flush=True)

                feature = x_val

                param = parameter_set[sample]

                final_state = self.run(feature, param, draw=False)

                fx_val = normalized_ground_state_prob(final_state)

                fx.append(fx_val)

            fx_set.append(np.array(fx))
        return fx_set


