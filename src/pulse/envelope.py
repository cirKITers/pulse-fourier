import jax.numpy as jnp

from constants import *


def gaussian_envelope(t):
    center = duration * dt_ / 2
    return amp * jnp.exp(-((t - center) ** 2) / (2 * sigma ** 2))


def DRAG_gaussian_envelope(t):
    """Gaussian envelope with DRAG correction."""
    delta = 0.05
    center = duration * dt_ / 2
    gaussian = amp * jnp.exp(-((t - center) ** 2) / (2 * sigma ** 2))
    derivative = -amp * (t - center) / sigma ** 2 * jnp.exp(-((t - center) ** 2) / (2 * sigma ** 2))
    return gaussian + delta * derivative
