import numpy as np

from pulse_fourier_models import PulseONEQFourier
from utils.helpers import *
from utils.visualize.fx import *

num_layer = 3
parameter = random_parameter(1, num_layer, 1)
qm = PulseONEQFourier(num_layer, parameter)


points = 100
x = np.linspace(0, 1, points)

fx = []
for t in range(points):
    fx.append(qm.predict_single(x[t]))

plot_fx(x, fx, "pulse fourier analysis")


