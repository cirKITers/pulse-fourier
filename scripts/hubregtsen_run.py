import numpy as np
# import time

from qft_models.hubregtsen import *
from utils.helpers import *
from utils.definitions import *
from utils.load import *
from visuals.fx import *
from visuals.coefficients import *
from fft.coefficients import *

num_qubits = 3
num_layers = 2

model = PennyHubregtsen(num_qubits)

p = 400
# x = np.linspace(-4, 4, p)
num_samples = 3

# Every sample with different fixed parameter theta (gradual)
parameter_set = gradual_parameter_set(num_layers, num_qubits, len(["RZ"]), num_samples)


# fx_set = model.sample_fourier(x, parameter_set, num_samples)
# save_set("PennyHubregtsen", num_qubits, num_layers, num_samples, x, fx_set, "../results/fx_data.txt")

# plot_nfx(x, fx_set)

x, fx_set = load_set("../results/fx_data.txt", 0)

a, b = coefficient_set(fx_set)

subplot(a, b)


