import math
import numpy as np

from src.pulse_gates import *
from utils.visualize import bloch_sphere

from scipy.optimize import minimize

current_state = PHASE_SHIFTED_STATE
samples = 100
sigma = 15
theta = jnp.pi/2

# for sigma = 15
RX = 0.04278068369641117
H = 0.042780849440995694       # does not work when init state is superposition
RZ = 0.4581489313344305
H2 = np.pi / (math.sqrt(2*np.pi) * sigma)
# good estimate for H is np.pi / (math.sqrt(2*np.pi) * sigma)

RX = np.linspace(0.042780631757944214, 0.042780641757944214, samples)

for _ in range(samples):
    ds, s, probs, ol, result = RX_pulse(theta, RX[_], sigma, current_state, plot_prob=False, plot_blochsphere=False)

    print(ds, "final:", result[-1].data, "|0>", probs[0], "---", "|1>", probs[1], "---", )
    # bloch_sphere.plot_bloch_sphere(result)
    # if ol > 0.99995:
    #     print("found")
        # bloch_sphere.plot_bloch_sphere(result)

# final_vector = Statevector([1.0 + 0.0j, 0.0 + 0.0j])
#
# def distance_to_target(rx):
#     """Calculates the Euclidean distance to the target state."""
#     ds, s, probs, ol, result = RX_pulse(theta, rx, sigma, current_state, plot_prob=False, plot_blochsphere=False)
#     return np.linalg.norm(result[-1].data - final_vector.data)
#
# initial_rx = RX

# USING BFGS
# result = minimize(distance_to_target, initial_rx, method='BFGS')
#
# best_rx = result.x[0]
# min_distance = result.fun
#
# print(f"Best RX: {best_rx}, Minimum Distance: {min_distance}")
# -> Scipy minimize found Best RX: 0.042780631757944214, Minimum Distance: 0.0006575298896276235


# Minimize the distance using Nelder-Mead
# result = minimize(distance_to_target, initial_rx, method='Nelder-Mead')
#
# best_rx = result.x[0]
# min_distance = result.fun
#
# print(f"Best RX: {best_rx}, Minimum Distance: {min_distance}")
# -> Best RX: 0.04278068369641117, Minimum Distance: 0.0006575304094206627




