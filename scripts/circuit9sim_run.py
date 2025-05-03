from qft_models.circuit9sim import *
from utils.helpers import *
from utils.definitions import *
from utils.load import *
from visuals.correlation_heatmap import complex_correlation
from visuals.fx import *
from visuals.coefficients import *
from fft.coefficients import *


num_qubits = 4  # scale
num_ansatz = 2  # const, 1 layer

model = Circuit9(num_qubits)

p = 2
x = np.linspace(0, 20, p)
num_samples = 1


parameter_set = random_parameter_set2(num_samples, 2, num_qubits, len(["RX"]))

print(parameter_set)

fx_set = model.sample_fourier(x, parameter_set, num_samples)
# save_set("Circuit15_Random", num_qubits, num_layers, num_samples, x, fx_set, "../results/fx_data.txt")

plot_nfx(x, fx_set, random_color=True)

