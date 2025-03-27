from qiskit.circuit import Parameter

import pulse_gates as plsg
from src.utils.visualize.fx import plot_fx
from constants import *
from utils.helpers import *

class PulseONEQFourier:
    model_name = "PulseONEQFourier"

    def __init__(self, num_qubits, num_layer, parameter):
        self.num_qubits = num_qubits
        self.L = num_layer
        self.params = parameter
        # self.model_name = "PulseONEQFourier"

    @staticmethod
    def data_encoding(current_state, x):
        theta = 2 * np.pi * x
        _, _, _, next_state = plsg.RX_pulse(theta, sigma, current_state)
        return next_state[-1]

    @staticmethod
    def trainable_block(current_state, theta):
        _, _, _, next_state = plsg.RZ_pulse(theta, sigma, current_state)
        return next_state[-1]

    def predict_single(self, x):
        current_state = GROUND_STATE
        for lay in range(self.L):
            next_state = self.data_encoding(current_state, x)
            next_state = self.trainable_block(next_state, self.params[0][lay].item())
            current_state = next_state
        probability_0 = prob(current_state.data[0])
        return 2 * probability_0 - 1  # Map to [-1, 1]

    def predict_interval(self, x_interval, plot=True):
        fx = []
        for t in range(len(x_interval)):
            fx.append(self.predict_single(x_interval[t]))

        if plot:
            plot_fx(x_interval, fx, "Fourier prediction on pulse level")
        return fx
