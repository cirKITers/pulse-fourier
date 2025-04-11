import numpy as np

from pulse.pulse_gates import *
from qft_models.pennylane_models import PennyCircuit
from utils.definitions import *
from utils.helpers import prints


num_q = 2
c = PennyCircuit(num_q)

penny_state = c.run_quick_circuit(PHI_PLUS_NO_CNOT.data)
prints(penny_state)


print("-"*20)
prints(PHI_PLUS)

print("-"*20)

state = cnot(PHI_PLUS_NO_CNOT)
prints(state)

print("-"*20)

print(statevector_similarity(penny_state, state))

