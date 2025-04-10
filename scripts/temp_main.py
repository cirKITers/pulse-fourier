from pulse.pulse_gates import H_pulseSPEC
from qft_models.pennylane_models import PennyCircuit
from utils.definitions import *
from utils.helpers import prints


num_q = 3
c = PennyCircuit(num_q)

penny_state = c.run_quick_circuit()
prints(penny_state)

# _, _, current_state = H_pulseSPEC(GROUND_STATE(num_q), 1)
# prints(current_state[-1])

# print(statevector_similarity(penny_state, current_state[-1]))

