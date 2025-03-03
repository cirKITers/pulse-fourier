import numpy as np

from utils.helpers import *
from gate_models import *
from pulse_gates import *

init_state = RANDOM_STATE()
theta = np.pi/2

# gate level
backend, qc = q_circuit(1)
gate_result, gate_probs = run_basic_model(backend, qc, init_state, theta)
print("Gate level:")
print("result_vector", gate_result)
print("probability", gate_probs)

# pulse level
sigma = 15
RX = 0.04278068369641117
H = 0.042780849440995694
RZ = 10
# H2 = np.pi / (math.sqrt(2*np.pi) * sigma)
ds, s, pulse_probs, ol, pulse_result = H_pulse(H, sigma, init_state, plot_prob=False, plot_blochsphere=False)
print("Pulse level:")
print("result_vector", pulse_result[-1].data)
print("probability", pulse_probs)

# COMPARISON
print("\n")
print("state vector (real): ", diff(pulse_result[-1].data.real, gate_result.real))
print("state vector (imag): ", diff(pulse_result[-1].data.imag, gate_result.imag))
print("probabilities: ", diff(pulse_probs, gate_probs))

