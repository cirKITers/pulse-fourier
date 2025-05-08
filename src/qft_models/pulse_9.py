import numpy as np
from pulse.pulse_backend import PulseBackend
from utils.definitions import GROUND_STATE
from utils.helpers import normalized_ground_state_prob, ground_state_prob


class Pulse9:

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.pls = PulseBackend(num_qubits, GROUND_STATE(num_qubits))

    def run(self, x, params, draw=False):

        def Ansatz(theta):

            for i in range(self.num_qubits):
                self.pls.h(i)

            for i in range(self.num_qubits - 1):
                control_qubit = i
                target_qubit = i + 1
                self.pls.cz(wires=[control_qubit, target_qubit])

            for i in range(self.num_qubits):
                self.pls.rx(theta[i], i)

        def Encoding(feature):
            for i in range(self.num_qubits):
                self.pls.rx(feature, i)

        def circuit():

            Ansatz(theta=params[0])  # 2*num_qubits

            Encoding(x)

            Ansatz(theta=params[1])  # 2*num_qubits

            return self.pls.current_state

        if draw:
            print("no drawing on pulse level.")
        return circuit()

    def sample_fourier(self, x, parameter_set, num_samples):
        fx_set = []
        for sample in range(num_samples):
            print("Starting Pulse 9 eval...", flush=True)

            # Print progress every 500 samples
            if (sample + 1) % 500 == 0:
                print(f"Processed sample: {sample + 1} / {num_samples}", flush=True)

            # Make fourier series for this sample
            fx = []
            for x_val in x:

                # print("discrete point:", x_val, flush=True)

                feature = x_val

                param = parameter_set[sample]

                final_state = self.run(feature, param, draw=False)

                fx_val = ground_state_prob(final_state)

                fx.append(fx_val)

            fx_set.append(np.array(fx))

        return fx_set

