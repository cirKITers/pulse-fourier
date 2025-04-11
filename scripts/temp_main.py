import numpy as np

from pulse.pulse_gates import *
from qft_models.pennylane_models import PennyCircuit
from utils.definitions import *
from utils.helpers import prints


# init_state = EXCITED_STATE(2)
#
# print(bool_valid_state(init_state))
#
# num_q = 2
# c = PennyCircuit(num_q)
#
# penny_state = c.run_quick_circuit(init_state.data)
# prints(penny_state)
#
# dsCZ = 0.11884149043553377
#
# test_state = cz_gate_solver(init_state, dsCZ)
# prints(test_state)

num_q = 2
c = PennyCircuit(num_q)

penny_state = c.run_quick_circuit([0.50000000+0.00000000j, 0.50000000+0.00000000j, 0.50000000+0.00000000j, -0.50000000+0.00000000j])
prints(penny_state)


print("-"*20)
# prints(PHI_PLUS)

print("-"*20)

_, _, state = cnot(Statevector([0.50000000+0.00000000j, 0.50000000+0.00000000j, 0.50000000+0.00000000j, -0.50000000+0.00000000j]))
prints(state[-1])

print("-"*20)

print(statevector_similarity(penny_state, state[-1]))










# tries = 50

# dsCZ = 0.11884147603542072
#
# ds = np.linspace(0.2515993336109954, 0.25231570179092044, tries)
#
# for i in range(len(ds)):
#     test_state = cz_gate_solver(EXCITED_STATE(2), ds[i])
#
#     sim = statevector_similarity(penny_state, test_state)
#
#     if sim > 0.9999999:
#         prints(test_state)
#         print(i, sim, ds[i])
#     else:
#         print(i, sim, ds[i])


