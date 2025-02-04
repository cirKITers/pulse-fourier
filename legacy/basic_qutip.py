import numpy as np
from qutip import Qobj, tensor, identity, basis
from qutip.qip.operations import hadamard_transform, rz, ry, controlled_gate

def layer(x, params, wires, i0=0, inc=1):
    num_wires = len(wires)
    i = i0

    ops = []
    for j, wire in enumerate(wires):
        op = hadamard() if wire == 0 else identity(2)
        op = rz(x[i % len(x)]) * op
        i += inc
        op = ry(params[0, j]) * op
        ops.append(op)

    for j in range(num_wires):
        control = wires[j]
        target = wires[(j + 1) % num_wires]
        crz_gate = controlled_gate(rz(params[1][j]), N=num_wires, control=control, target=target)
        ops.append(crz_gate)

    return tensor(*ops)

def ansatz(x, params, wires):
    layers = []
    for j, layer_params in enumerate(params):
        layers.append(layer(x, layer_params, wires, i0=j * len(wires)))
    return tensor(*layers)

def adjoint_ansatz(x, params, wires):
    return ansatz(x, params, wires).dag()

def init_circuit(num_wires):
    wires = list(range(num_wires))

    def kernel_circuit(x1, x2, params):
        psi_x1 = ansatz(x1, params, wires)
        psi_x2_dag = adjoint_ansatz(x2, params, wires)

        result = (psi_x2_dag * psi_x1).tr()
        return result

    return kernel_circuit

def kernel(x1, x2, params, num_wires):
    circuit = init_circuit(num_wires)
    return circuit(x1, x2, params)

num_wires = 3
x1 = [0.5, 0.1, 0.3]
x2 = [0.6, 0.2, 0.4]
params = np.random.random((2, num_wires))

result = kernel(x1, x2, params, num_wires)
print(result)
