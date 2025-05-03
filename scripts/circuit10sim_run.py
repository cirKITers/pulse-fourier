import numpy as np

from qft_models.circuit10sim import Circuit10
# import time

from qft_models.hubregtsen import *
from utils.helpers import *
from utils.definitions import *
from utils.load import *
from visuals.correlation_heatmap import complex_correlation
from visuals.fx import *
from visuals.coefficients import *
from fft.coefficients import *


num_qubits = 4
num_layers = 1

# model = Circuit10(num_qubits)
#
# p = 400
# x = np.linspace(0, 8, p)
# num_samples = 100

# Every sample with different fixed parameter theta (gradual)
# parameter_set = random_parameter_set(num_layers, num_qubits, len(["RZ", "RZ"]), num_samples)
#
# fx_set = model.sample_fourier(x, parameter_set, num_samples)
# save_set("Circuit10_Random", num_qubits, num_layers, num_samples, x, fx_set, "../results/fx_data.txt")
#
# plot_nfx(x, fx_set, random_color=True)

x, fx_set = load_set("../results/fx_data.txt", 1)

_, _, c = coefficient_set(fx_set)

magnitude_c = magnitude(c)

print(magnitude_c.shape)



