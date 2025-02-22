from src.basic_pulse import *
from utils.visualize import bloch_sphere

samples = 1

# sigma is 15

drive_strength = np.linspace(0.04277960635661175, 0.042781180930164746, samples)
sigma = np.linspace(15, 15, samples)


for _ in range(samples):
    ds, s, probs, ol, result = h_pulse(drive_strength[_], sigma[_], plot=False, bool_blochsphere=False)
    print("|0>", probs[0], "---", "|1>", probs[1], "---", "overlap =", ol, ds, s, "final:", result[-1].data)
    if ol > 0.999998:
        print("found")
        # bloch_sphere.plot_bloch_sphere(result)
