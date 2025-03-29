import json

from data.save import *
from pulse_fourier_models import PulseONEQFourier
from src.oneQ_pulse_gates import *
from utils.helpers import *
from constants import *

# theta = np.pi
current_state = GROUND_GROUND
drive_strength = 0.04

drive_strength, final_probs, result = CNOT_Pulse(drive_strength, sigma, current_state, plot_prob=True, plot_blochsphere=True)


# this takes 20 minutes
# num_layer = 3
# parameter = random_parameter(1, num_layer, 1)
# qm = PulseONEQFourier(num_layer, parameter)
#
# points = 200
# x_ = np.linspace(0, 2, points)
#
# fx = qm.predict_interval(x_)


# SAVE DATA
# data_to_save = {
#     "model_name": qm.model_name+"("+str(parameter.shape).replace("(", "").replace(")", "").replace(" ", "")+")",
#     "x": x_.tolist(),  #
#     "fx": [val.tolist() if isinstance(val, np.ndarray) else val for val in fx],
#     "parameters": parameter.tolist(),
# }
#
# save_to(data_to_save, pulse_file)
