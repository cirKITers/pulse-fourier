import json

from data.save import *
from pulse_fourier_models import PulseONEQFourier
from utils.helpers import *
from constants import *

num_layer = 3
parameter = random_parameter(1, num_layer, 1)
qm = PulseONEQFourier(num_layer, parameter)

points = 200
x_ = np.linspace(0, 2, points)

fx = qm.predict_interval(x_)

# SAVE DATA
data_to_save = {
    "model_name": qm.model_name+"("+str(parameter.shape).replace("(", "").replace(")", "").replace(" ", "")+")",
    "x": x_.tolist(),  #
    "fx": [val.tolist() if isinstance(val, np.ndarray) else val for val in fx],
    "parameters": parameter.tolist(),
}

save(data_to_save, file_name_pulse)
