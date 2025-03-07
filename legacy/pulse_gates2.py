from qiskit import pulse
import numpy as np

# Parameters
frequency = 5e9  # Qubit resonance frequency (5 GHz)
theta = np.pi / 2  # Rotation angle for R_X
drive_strength = 50e6  # Drive strength in Hz or rad/s
sigma = 50e-9  # Pulse width in seconds (Gaussian shape)
duration = 1000  # Duration in discrete time steps (integer)

# Check the values of parameters to ensure consistency
print(f"Frequency: {frequency}, Drive strength: {drive_strength}, Sigma: {sigma}, Duration: {duration}")

# Create a Gaussian pulse for R_X(π/2)
# Ensure correct usage of parameters
pulse_sequence = pulse.Schedule(name="RX(π/2)")
gaussian_pulse = pulse.Gaussian(duration=duration,
                                amplitude=drive_strength,
                                frequency=frequency,
                                sigma=sigma,
                                phase=0)  # Phase is 0 for this example

# Add the pulse to the schedule
pulse_sequence += gaussian_pulse

# Display the pulse sequence
print(pulse_sequence)
