from utils.definitions import *
from utils.helpers import *


fid = statevector_fidelity(SUPERPOSITION_STATE_H(1), Statevector([0.5 + 0.5j, 0.5 + 0.5j]).data)

print(fid)

