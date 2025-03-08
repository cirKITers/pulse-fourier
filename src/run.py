from gate_fourier_models import GateONEQFourier
from pulse_fourier_models import PulseONEQFourier
from utils.helpers import *
from data.load import *
from constants import *
from src.utils.visualize.fx import *


x, pulse_fx, parameter, parameter_shape = load_data_from_jsonl(file_name_pulse, 9)


num_repetitions = parameter_shape[0]
num_layer = parameter_shape[1]
num_qubits = parameter_shape[2]

simulator = 'statevector_simulator'
shots = 32768

gate_qm = GateONEQFourier(num_qubits, num_layer, parameter)

gate_f_x = gate_qm.predict_interval(simulator, shots, x, plot=False)

plot_2fx_advanced(x, pulse_fx, gate_f_x, "Pulse Fourier", "Gate Fourier", "Comparison", "X", "Y", None, None, True)



