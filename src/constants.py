sigma = 15
shots = 32768   # for qasm simulator
dt_ = 0.1   # The smaller the time step, the more accurate the simulation, but also the more computationally expensive
amp = 1.0  # Amplitude, height gaussian bell at peak, default Max
omega = 5.0
duration = 120


pulse_file = "../data/pulse_fourier.jsonl"
gate_file = "../data/gate_fourier.jsonl"
correlation_dir = "../data/correlations"
correlation_dir_detailed = "../data/correlations/detailed"

