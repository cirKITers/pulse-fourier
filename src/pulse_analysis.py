import math
import numpy as np

from src.pulse_gates import *
from utils.visualize import bloch_sphere

current_state = RANDOM_STATE()
samples = 1
sigma = 15
theta = jnp.pi/2

# for sigma = 15
RX = 0.04278068369641117
H = 0.042780849440995694       # does not work when init state is superposition
RZ = 0.4581489313344305
H2 = np.pi / (math.sqrt(2*np.pi) * sigma)
# good estimate for H is np.pi / (math.sqrt(2*np.pi) * sigma)

for _ in range(samples):
    ds, s, probs, ol, result = RZ_pulse(1, theta, RZ, sigma, current_state, plot_prob=False, plot_blochsphere=False)
    print("|0>", probs[0], "---", "|1>", probs[1], "---", ds, s, "final:", result[-1].data)
    bloch_sphere.plot_bloch_sphere(result)
    # if ol > 0.99995:
    #     print("found")
        # bloch_sphere.plot_bloch_sphere(result)

