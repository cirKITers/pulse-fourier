from qiskit import QuantumCircuit
import qiskit_aer
from qiskit.circuit import Parameter

from visuals import fx
from visuals.bloch_sphere import *

# https://quantum.ibm.com/composer/files/new

# math behind it: https://dojo.qulacs.org/en/latest/notebooks/1.3_multiqubit_representation_and_operations.html
matplotlib.use('TkAgg')  # more general backend, works for PyCharm

class EntanglingFourier:
    model_name = "EntanglingFourier"
    num_layer = 4
    num_qubits = 2
    num_gates = 3

    def __init__(self, parameter):
        if not isinstance(parameter, np.ndarray) or parameter.ndim != 3:
            raise ValueError("Parameter for init EntanglingFourier must be a 3D NumPy array with shape (num_layers, num_qubits, num_gates).")
        self.num_layers = parameter.shape[0]
        self.num_qubits = parameter.shape[1]
        self.num_gates = parameter.shape[2]
        if isinstance(parameter, np.ndarray):
            self.params = parameter.flatten().tolist()
        else:
            raise ValueError("Parameter for init QiskitFourier1 is not a nparray.")

        self.param_label = []
        for lay in range(self.num_layers):
            for qub in range(self.num_qubits):
                for gate in range(self.num_gates):
                    self.param_label.append(Parameter(f'theta_layer_{lay}_qubit_{qub}_gate_{gate}'))

        # sanity check
        self.total_params_expected = len(self.params)
        if len(self.param_label) != self.total_params_expected:
            raise ValueError(
                f"The number of parameter labels ({len(self.param_label)}) does not match the expected number of parameters ({self.total_params_expected}).")

    def define_circuit(self, x):
        qc = QuantumCircuit(self.num_qubits)
        param_index = 0
        for lay in range(self.num_layers):
            for qub in range(self.num_qubits):
                # Encoding
                qc.rx(self.param_label[param_index] * x, qub)
                param_index += 1
                # Trainable Block
                qc.rz(self.param_label[param_index], qub)
                param_index += 1
                qc.ry(self.param_label[param_index], qub)
                param_index += 1

            if self.num_qubits > 1:
                qc.cx(0, 1)

        param_binds = dict(zip(self.param_label, self.params))
        qc = qc.assign_parameters(param_binds)
        return qc


# with correct convention for parameter assigning
class BasicFourier1:
    model_name = "QiskitFourier1"
    num_layer = 4
    num_qubits = 2
    num_gates = 3

    # parameter always in the same format! (num_layers, num_qubits, num_gates)
    def __init__(self, parameter):
        self.num_layers = parameter.shape[0]
        self.num_qubits = parameter.shape[1]
        self.num_gates = parameter.shape[2]

        if isinstance(parameter, np.ndarray):
            self.params = parameter.flatten().tolist()
        else:
            raise ValueError("Parameter for init is not a nparray.")

        self.param_label = []
        for lay in range(self.num_layers):
            for qub in range(self.num_qubits):
                for gate in range(self.num_gates):
                    self.param_label.append(Parameter(f'theta_layer_{lay}_qubit_{qub}_gate_{gate}'))

        # sanity check 1
        total_params_expected = len(self.params)
        if len(self.params) != total_params_expected:
            raise ValueError(f"Error 1: The number of parameters provided ({len(self.params)}) does not match the expected number ({total_params_expected}).")

        # sanity check 2
        total_params_expected = BasicFourier1.num_layer * BasicFourier1.num_qubits * BasicFourier1.num_gates
        if len(self.params) != total_params_expected:
            raise ValueError(f"Error 2: The number of parameters provided ({len(self.params)}) does not match the expected number ({total_params_expected}).")

    def define_circuit(self, x):
        qc = QuantumCircuit(self.num_qubits)
        param_index = 0
        for lay in range(self.num_layers):
            for qub in range(self.num_qubits):
                # Encoding
                qc.rx(self.param_label[param_index] * x, qub)
                param_index += 1
                # Trainable Block
                qc.rz(self.param_label[param_index], qub)
                param_index += 1
                qc.ry(self.param_label[param_index], qub)
                param_index += 1

        param_binds = dict(zip(self.param_label, self.params))
        qc = qc.assign_parameters(param_binds)
        # print(qc.draw())
        return qc

class TestCircuit:
    model_name = "TestCircuit"
    num_layer = 1
    num_qubits = 2
    num_gates = 1

    # parameter always in the same format! (num_layers, num_qubits, num_gates)
    def __init__(self, parameter):
        self.num_layers = parameter.shape[0]
        self.num_qubits = parameter.shape[1]
        self.num_gates = parameter.shape[2]

        if isinstance(parameter, np.ndarray):
            self.params = parameter.flatten().tolist()
        else:
            raise ValueError("Parameter for init is not a nparray.")

        self.param_label = []
        for lay in range(self.num_layers):
            for qub in range(self.num_qubits):
                for gate in range(self.num_gates):
                    self.param_label.append(Parameter(f'theta_layer_{lay}_qubit_{qub}_gate_{gate}'))

        # sanity check 1
        # total_params_expected = len(self.params)
        # if len(self.params) != total_params_expected:
        #     raise ValueError(f"Error 1: The number of parameters provided ({len(self.params)}) does not match the expected number ({total_params_expected}).")

        # sanity check 2
        # total_params_expected = BasicFourier1.num_layer * BasicFourier1.num_qubits * BasicFourier1.num_gates
        # if len(self.params) != total_params_expected:
        #     raise ValueError(f"Error 2: The number of parameters provided ({len(self.params)}) does not match the expected number ({total_params_expected}).")

    def define_circuit(self, x):
        qc = QuantumCircuit(self.num_qubits)
        for lay in range(self.num_layers):
            for qub in range(1):
                # qc.h(0)
                # qc.h(1)
                qc.rx(np.pi / 2, 0)
                qc.rx(np.pi / 2, 1)
                qc.cx(0, 1)
                # qc.rx(np.pi / 2, 1)
        #         qc.rx(np.pi / 2, 2)
        #         qc.rx(np.pi / 2, 3)
        #         qc.rx(np.pi / 2, 4)
        # param_binds = dict(zip(self.param_label, self.params))
        # qc = qc.assign_parameters(param_binds)
        return qc

# for quick and easy modifications
class QuickCircuit:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits

    def run_quick_circuit(self, initial_state):
        qc = QuantumCircuit(self.num_qubits)

        if initial_state is not None:
            qc.initialize(initial_state, range(self.num_qubits))  # Initialize with the custom state

        # qc.rx(np.pi / 2, 0)
        # qc.rx(np.pi / 2, 1)
        qc.cx(1, 0)
        simulator = 'statevector_simulator'
        backend = qiskit_aer.Aer.get_backend(simulator)
        result = backend.run(qc).result()
        statevector = result.get_statevector()
        return statevector

def predict_interval(qm, simulator, shots, x, plot=False):
    f_x = []
    for i in range(len(x)):
        qc = qm.define_circuit(x[i])
        mapped_probability0, resulting_statevector = compute_expectations(qc, simulator, shots)
        f_x.append(mapped_probability0.item())
    if plot:
        fx.plot_fx(x, f_x, "Gate level Fourier Model Prediction")
    return f_x

def predict_single(qm, simulator, shots, x):
    qc = qm.define_circuit(x)
    f_x, resulting_statevector = compute_expectations(qc, simulator, shots)
    return f_x, resulting_statevector

def compute_expectations(qc, simulator, shots):
    backend = qiskit_aer.Aer.get_backend(simulator)
    statevector = None
    if simulator == 'qasm_simulator':
        qc.measure_all()
        result = backend.run(qc, shots=shots).result()
        counts = result.get_counts()
        probability_0 = counts.get('0' * qc.num_qubits, 0) / shots  # Probability of measuring |0> state - expectation value
    else:  # if simulator == 'statevector_simulator':
        print(qc.draw())
        result = backend.run(qc).result()
        statevector = result.get_statevector()
        # bloch_sphere_multiqubit_trajectory([Statevector(statevector.data)])
        probability_0 = prob(statevector.data[0])
    return 2 * probability_0 - 1, statevector  # Map to [-1, 1]


# Only one qubit possible sor far for simplification purposes
class GateONEQFourier:
    model_name = "GateONEQFourier"

    def __init__(self, num_qubits, num_layer, parameter):
        self.num_q = num_qubits
        self.L = num_layer
        if isinstance(parameter, np.ndarray):
            self.params = parameter.flatten().tolist()
        elif isinstance(parameter, list):
            self.params = parameter
        else:
            self.params = [parameter]
        self.param_label = [Parameter(f'theta_{i}') for i in range(self.L)]

    @staticmethod
    def data_encoding(qc, qubit, x):
        qc.rx(2 * np.pi * x, qubit)

    @staticmethod
    def trainable_block(qc, qubit, theta):
        qc.rz(theta, qubit)
        qc.ry(theta, qubit)

    def define_circuit(self, x):
        qc = QuantumCircuit(self.num_q)
        print("param_label", self.param_label)
        print("params", self.params)
        for lay in range(self.L):
            for q_i in range(self.num_q):
                self.data_encoding(qc, q_i, x)
                # print("_")
                # print("lay", lay, "qubit", q_i, "params", self.params[lay], "param_label", self.param_label[lay])
                self.trainable_block(qc, q_i, self.param_label[lay])
        for i, theta in enumerate(self.param_label):
            qc = qc.assign_parameters({theta: self.params[i % len(self.params)]})
        print(qc.draw())
        return qc

