from qiskit.circuit import Parameter

import pulse_gates as pls
from utils.visualize import fx
from constants import *
from utils.helpers import *

class PulseONEQFourier:
    def __init__(self, num_layer, parameter):
        self.L = num_layer
        self.params = parameter

    @staticmethod
    def data_encoding(current_state, x):
        theta = 2 * np.pi * x
        _, _, _, next_state = pls.RX_pulse(theta, sigma, current_state)
        return next_state[-1]

    @staticmethod
    def trainable_block(current_state, theta):
        _, _, _, next_state = pls.RZ_pulse(theta, sigma, current_state)
        return next_state[-1]

    def predict_single(self, x):
        current_state = GROUND_STATE
        for lay in range(self.L):
            next_state = self.data_encoding(current_state, x)
            next_state = self.trainable_block(next_state, self.params[lay])
            current_state = next_state
        probability_0 = prob(current_state.data[0])
        return 2 * probability_0 - 1  # Map to [-1, 1]



