from src.basic_pulse import *
from utils.visualize import bloch_sphere

samples = 1
H = 3
# sigma is 15

if H == 1:
    drive_strength = np.linspace(0.04277960635661175, 0.042781180930164746, samples)
    sigma = np.linspace(15, 15, samples)

    for _ in range(samples):
        ds, s, probs, ol, result = h_pulse(drive_strength[_], sigma[_], plot=False, bool_blochsphere=False)
        print("|0>", probs[0], "---", "|1>", probs[1], "---", "overlap =", ol, ds, s, "final:", result[-1].data)
        if ol > 0.999997:
            print("found")
            bloch_sphere.plot_bloch_sphere(result)

elif H == 2:
    drive_strength = np.linspace(0.04277960635661175, 0.042781180930164746, samples)
    sigma = np.linspace(15, 15, samples)

    for _ in range(samples):
        ds, s, probs, ol, result = RX_pulse(drive_strength[_], sigma[_], theta=jnp.pi/2, plot=False, bool_blochsphere=False)
        print("|0>", probs[0], "---", "|1>", probs[1], "---", "overlap =", ol, ds, s, "final:", result[-1].data)
        if ol > 0.1:
            print("found")
            bloch_sphere.plot_bloch_sphere(result)

else:
    drive_strength = np.linspace(0.04277960635661175, 0.042781180930164746, samples)
    sigma = np.linspace(15, 15, samples)

    for _ in range(samples):
        ds, s, probs, ol, result = RZ_pulse(drive_strength[_], sigma[_], theta=jnp.pi / 2, plot=False, bool_blochsphere=False)
        print("|0>", probs[0], "---", "|1>", probs[1], "---", "overlap =", ol, ds, s, "final:", result[-1].data)
        if ol > 0.1:
            print("found")
            bloch_sphere.plot_bloch_sphere(result)


