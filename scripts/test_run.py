from pulse.pulse_backend import PulseBackend
from utils.data_handler import save
from utils.definitions import GROUND_STATE
from utils.helpers import prints

pls = PulseBackend(5, GROUND_STATE(5))

pls.h(0)
pls.cnot([0, 1])
pls.h(0)

prints(pls.current_state)


save("test", 1, 1, 1, 1, 1, 1, [1], [1], "test/")

