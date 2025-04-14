shots = 32768   # for qasm simulator

# pulse
sigma = 15
dt_ = 0.1   # The smaller the time step, the more accurate the simulation, but also the more computationally expensive
amp = 1.0  # Amplitude, height gaussian bell at peak, default Max
omega = 5.0
duration = 120

# qiskit doc standard:
# duration = 128
# amp = 0.1
# sigma = 16

# alternative Abby Mitchell
# amp = 1. # amplitude
# sig = 0.399128/r #sigma
# t0 = 3.5*sig # center of Gaussian
# T = 7*sig # end of signal

# CNOT
sigmaCNOT = 1.5
durationCNOT = 300


