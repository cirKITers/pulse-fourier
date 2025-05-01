import numpy as np
from qiskit.quantum_info import Statevector
from qiskit_dynamics import Signal, Solver

from qiskit.quantum_info.operators import Operator

from utils.helpers import prints


def gaussian(amp, sig, t0, t):
    return amp * np.exp( -(t - t0)**2 / (2 * sig**2) )

r = 0.1
w = 1.
X = Operator.from_label('X')
Y = Operator.from_label('Y')
Z = Operator.from_label('Z')

# specifications for generating envelope
amp = 1. # amplitude
sig = 0.399128/r #sigma
t0 = 3.5*sig # center of Gaussian
T = 7*sig # end of signal

gaussian_envelope = lambda t: gaussian(amp, sig, t0, t)


gauss_signal = Signal(envelope=gaussian_envelope, carrier_freq=w)


# prepare the static hamiltonian (which we call drift here):
drift = 2 * np.pi * w * Z/2
# prepare the hamiltonian operators:
operators = [2 * np.pi * r * X/2]

hamiltonian_solver = Solver(static_hamiltonian=drift,
                            hamiltonian_operators=operators)

y0 = Statevector([0., 1.])
# create an array of 500 points between 0 and T
times = np.linspace(0., T, 500)
sol = hamiltonian_solver.solve(
  t_span=[0., T], # time interval to integrate over
  y0=y0,
  signals=[gauss_signal],  # initial state
  t_eval=times)         # points to integrate over


# Return the full list of Statevectors for every point
# during the entire solution
all_states = sol.y
# Return the Statevector at a specified t_eval point
mid_state = sol.y[249]
# Return the Statevector at the final time
final_state = sol.y[-1]


prints(mid_state)
prints(final_state)

