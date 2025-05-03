import jax.numpy as jnp
from qiskit_dynamics import DiscreteSignal
from qiskit_dynamics.signals import Convolution

from constants import *


def gaussian_envelope(t):
    center = T * dt / 2
    return 1.0 * jnp.exp(-((t - center) ** 2) / (2 * sigma ** 2))


def DRAG_gaussian_envelope(t):
    """Gaussian envelope with DRAG correction."""
    delta = 0.05
    center = T * dt / 2
    gaussian = amp * jnp.exp(-((t - center) ** 2) / (2 * sigma ** 2))
    derivative = -amp * (t - center) / sigma ** 2 * jnp.exp(-((t - center) ** 2) / (2 * sigma ** 2))
    return gaussian + delta * derivative


# define convolution filter
def gaus(t):
    sigma = 15
    _dt = 0.1
    return 2.*_dt/np.sqrt(2.*np.pi*sigma**2)*np.exp(-t**2/(2*sigma**2))


convolution = Convolution(gaus)

# define function mapping parameters to signals
def signal_mapping(params):

    # map samples into [-1, 1]
    bounded_samples = jnp.arctan(params) / (np.pi / 2)

    # pad with 0 at beginning
    padded_samples = jnp.append(jnp.array([0], dtype=complex), bounded_samples)

    # apply filter
    output_signal = convolution(DiscreteSignal(dt=0.1, samples=padded_samples))

    # set carrier frequency to v
    output_signal.carrier_freq = nu

    return output_signal



