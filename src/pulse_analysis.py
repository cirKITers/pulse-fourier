import math
import numpy as np

from src.pulse_gates import *
from utils.visualize import bloch_sphere

current_state = GROUND_STATE
samples = 1
sigma = 15
drive_strength_samples = np.linspace(0, 0.02277960635661175, samples)
sigma_samples = np.linspace(15, 15, samples)
theta = jnp.pi/2

# for sigma = 15
RX = 0.04278068369641117
H = 0.042780849440995694       # does not work when init state is superposition
RZ = 0
H2 = np.pi / (math.sqrt(2*np.pi) * sigma)
# good estimate for H is np.pi / (math.sqrt(2*np.pi) * sigma)

for _ in range(samples):
    ds, s, probs, ol, result = RZ_pulse(theta, RX, sigma, current_state, plot_prob=False, plot_blochsphere=False)
    print("|0>", probs[0], "---", "|1>", probs[1], "---", ds, s, "final:", result[-1].data)
    bloch_sphere.plot_bloch_sphere(result)
    if ol > 0.99995:
        print("found")
        # bloch_sphere.plot_bloch_sphere(result)

