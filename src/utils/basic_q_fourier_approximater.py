import numpy as np
import qiskit_aer
from matplotlib import pyplot as plt
from scipy.optimize import minimize


def target_function(x):
    return np.sin(2 * np.pi * x)


def loss(params, x_values, target_values, L, num_q, shots):
    total_loss = 0
    backend = qiskit_aer.Aer.get_backend('qasm_simulator')
    for x, target in zip(x_values, target_values):
        qc = quantum_model(x, params, L, num_q)
        f_x = comp_expectations(qc, backend, shots)
        total_loss += (f_x - target) ** 2

    return total_loss


x_values = np.linspace(0, 1, 100)
target_values = target_function(x_values)
init_params = [np.pi / 8] * L

result = minimize(loss, init_params, args=(x_values, target_values, L, num_q, shots), method='COBYLA', options={'maxiter': 1000, 'disp': True})

trained_params = result.x
print("Trained parameter:", trained_params)

f_x_values_optimized = []
for x in x_values:
    qc = quantum_model(x, trained_params, L, num_q)
    f_x_optimized = comp_expectations(qc, backend, shots)
    f_x_values_optimized.append(f_x_optimized)

plt.figure(figsize=(8, 6))
plt.plot(x_values, target_values, label="Target Function (sin)", color='r')
plt.plot(x_values, f_x_values_optimized, label="Optimized Quantum Model", color='b')
plt.title("Quantum Model vs. Target Function")
plt.xlabel("Input x")
plt.ylabel("Function Value")
plt.legend()
plt.grid(True)
plt.show()
