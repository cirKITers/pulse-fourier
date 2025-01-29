import numpy as np

#
from basic_q_fourier import QuantumModel

simulator = 'statevector_simulator'
num_layer = 3
num_qubits = 1
shots = 32768
params = [np.pi / 8, np.pi / 8, np.pi / 8]
interval = [-2, 2]
space = 200

qm = QuantumModel(num_qubits, num_layer, params)
x, f_x = qm.predict_interval(simulator, shots, interval, space, plot=True)

f_fft = np.fft.fft(f_x)
freqs = np.fft.fftfreq(len(x), x[1] - x[0])

idx = np.argmax(np.abs(f_fft[1:])) + 1  # Ignore the first element (DC component)
dominant_frequency = np.abs(freqs[idx])

# Calculate the period T (inverse of frequency)
T = 1 / dominant_frequency

print(f"The estimated period is {T}")






