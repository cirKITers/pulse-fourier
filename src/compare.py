import numpy as np

from src.utils.visualize import bloch_sphere
from utils.helpers import *
from gate_models import *
from pulse_gates import *


init_state = RANDOM_STATE()
theta = random_theta()
print("Initial state: ", init_state.data)
print("theta: ", theta)
print("\n")

# gate level
backend, qc = q_circuit(1)
gate_result, gate_probs = run_basic_model(backend, qc, init_state, theta)
print("Gate level:")
print("result_vector", gate_result.data)
print("probability", gate_probs)
# bloch_sphere.plot_bloch_sphere([init_state, gate_result])

# pulse level
sigma = 15
# H2 = np.pi / (math.sqrt(2*np.pi) * sigma)


samples = 10
# ii = np.linspace(0.31831092178570325, 0.3183109217857033, samples)

ds, pulse_probs, ol, pulse_result = RX_pulse(theta, sigma, init_state, plot_prob=False, plot_blochsphere=False)


print("\n")
print("Pulse level:")
print("result_vector", pulse_result[-1].data)
print("probability", pulse_probs)
# bloch_sphere.plot_bloch_sphere(pulse_result)

# COMPARISON
print("\n")
print("state vector (real): ", diff(pulse_result[-1].data.real, gate_result.data.real))
print("state vector (imag): ", diff(pulse_result[-1].data.imag, gate_result.data.imag))
print("probabilities: ", diff(pulse_probs, gate_probs))

