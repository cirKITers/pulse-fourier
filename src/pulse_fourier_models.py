import oneQ_pulse_gates as plsg
from src.visualize.fx import plot_fx
from constants import *
from utils.helpers import *

class MultiQubitPulseFourier:
    model_name = "MultiQubitPulseFourier"
    num_qubits = 2
    num_layer = 1
    num_gates = 2

    def __init__(self, parameter):
        self.num_layers = parameter.shape[0]
        self.num_qubits = parameter.shape[1]
        self.num_gates = parameter.shape[2]

        if isinstance(parameter, np.ndarray):
            self.params = parameter.flatten().tolist()
        else:
            raise ValueError("Parameter for init is not a nparray.")

        # sanity check
        total_params_expected = MultiQubitPulseFourier.num_gates * MultiQubitPulseFourier.num_layer * MultiQubitPulseFourier.num_qubits
        if not len(self.params) == total_params_expected:
            raise ValueError(f"The number of parameter ({len(self.params)}) does not match the expected number of parameters ({total_params_expected}).")

        self.current_state = GROUND_STATE(self.num_qubits)

    @staticmethod
    def data_encoding(current_state, theta, x):
        theta = theta * x
        _, _, _, next_states = plsg.RX_pulse(theta, sigma, current_state)
        next_state = next_states[-1]
        return next_state

    @staticmethod
    def trainable_block(current_state, theta):
        _, _, _, next_states = plsg.RZ_pulse(theta, sigma, current_state)
        next_state = next_states[-1]
        return next_state

    def predict_single(self, x):
        current_state = GROUND_STATE_OneQ
        for lay in range(self.num_layer):
            next_state = self.data_encoding(current_state, self.params[0][lay].item(), x)
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

class PulseONEQFourier:
    model_name = "PulseONEQFourier"
    num_qubits = 1
    num_layer = 1
    num_gates = 2

    def __init__(self, parameter):
        if isinstance(parameter, np.ndarray):
            self.params = parameter.flatten().tolist()
        else:
            raise ValueError("Parameter for init is not a nparray.")

        self.current_state = None

    @staticmethod
    def data_encoding(current_state, theta, x):
        theta = theta * x
        _, _, _, next_states = plsg.RX_pulse(theta, sigma, current_state)
        next_state = next_states[-1]
        return next_state

    @staticmethod
    def trainable_block(current_state, theta):
        _, _, _, next_states = plsg.RZ_pulse(theta, sigma, current_state)
        next_state = next_states[-1]
        return next_state

    def predict_single(self, x):
        current_state = GROUND_STATE_OneQ
        for lay in range(self.num_layer):
            next_state = self.data_encoding(current_state, self.params[0][lay].item(), x)
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
