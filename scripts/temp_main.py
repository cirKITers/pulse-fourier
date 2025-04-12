import numpy as np

from pulse.pulse_gates import *
from qft_models.pennylane_models import PennyCircuit
from utils.definitions import *
from utils.helpers import prints





# num_q = 2
# c = PennyCircuit(num_q)
#
# penny_state = c.run_quick_circuit([0.50000000+0.00000000j, 0.50000000+0.00000000j, 0.50000000+0.00000000j, -0.50000000+0.00000000j])
# prints(penny_state)
#
#
# print("-"*20)
# # prints(PHI_PLUS)
#
# print("-"*20)
#
# tries = 400
#
# ds = np.linspace(0.01, 2.5, tries)
#
# # ds = np.arange(- 2, 2 + 1/8, 1/8) * np.pi
#
#
# pair = []
#
# for i in range(len(ds)):
#
#     _, _, state = cnot(Statevector([0.50000000+0.00000000j, 0.50000000+0.00000000j, 0.50000000+0.00000000j, -0.50000000+0.00000000j]), ds[i])
#
#     sim = statevector_similarity(penny_state, state[-1])
#     print(i, sim, ds[i])
#     # prints(state[-1])
#     if sim > 0.95:
#         prints(state[-1])
#         pair.append((sim, ds[i]))
#     if sim < 0.08:
#         prints(state[-1])
#
# print(pair)
# print("-"*20)
#
#
#









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


