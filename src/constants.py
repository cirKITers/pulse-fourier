import numpy as np

# pulse: tunable
sigma = 15  # standard deviation of gaussian function
dt_ = 0.1   # Time steps. The smaller the time step, the more accurate the simulation, but also the more computationally expensive
amp = 1.0  # Amplitude, height gaussian bell at peak, default Max
duration = 120  # will lead to t_span from 0 to 12 with 120 intervals of 0.1
t_span = np.linspace(0, duration * dt_, duration + 1)

# pulse standards
omega = 5.0     # standard (Frequency of the qubit transition in GHz, also denoted as w)


# qiskit doc standard:
# duration = 128
# amp = 0.1
# sigma = 16

# alternative Abby Mitchell
# amp = 1. # amplitude
# sig = 0.399128/r #sigma
# t0 = 3.5*sig # center of Gaussian
# T = 7*sig # end of signal


shots = 32768   # for qasm simulator

