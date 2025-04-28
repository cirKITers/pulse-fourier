import unittest
from pulse.pulse_system import *

# Passed all tests
class TestRXPulseSPEC(unittest.TestCase):

    def test_single_qubit_rx(self):
        theta = np.pi / 2
        initial_state = GROUND_STATE_OneQ
        target_qubits = [0]
        pls = PulseSystem(1, initial_state)
        pls.rx(theta, initial_state, 0.0, target_qubits)
        expected_state = SUPERPOSITION_STATE_RX_OneQ
        final_state = pls.current_state
        print(expected_state, final_state)
        # for i in range(len(result)):
        #     prints(result[i])
        print("sim", overlap_components(final_state, expected_state))
        print("fid", statevector_fidelity(final_state, expected_state))
        self.assertTrue(bool_statevector_closeness(final_state, expected_state))

    # def test_two_qubit_rx(self):
    #     theta = np.pi
    #     initial_state = GROUND_STATE(2)
    #     target_qubits = [0, 1]
    #     _, _, resulty = RX_pulseSPEC(theta, initial_state, 0.0, target_qubits)
    #     final_state = resulty[-1]
    #     # print(resulty[-1])
    #     expected_state = Statevector([0, 0, 0, -1])
    #     print("sim", statevector_similarity(final_state, expected_state))
    #     print("fid", statevector_fidelity(final_state, expected_state))
    #     self.assertTrue(bool_statevector_closeness(final_state, expected_state))
    #
    # def test_no_target_qubits(self):
    #     theta = np.pi / 4
    #     initial_state = Statevector.from_label('01')
    #     target_qubits = []
    #     _, _, result = RX_pulseSPEC(theta, initial_state, 0.0, target_qubits)
    #     final_state = Statevector(result[-1])
    #     expected_state = Statevector.from_label('01')
    #     print("sim", statevector_similarity(final_state, expected_state))
    #     print("fid", statevector_fidelity(final_state, expected_state))
    #     self.assertTrue(final_state.equiv(expected_state))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
