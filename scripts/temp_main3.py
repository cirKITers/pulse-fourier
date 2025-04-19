from pulse.pulse_system import *

# FIXED PARAMS
theta = np.pi / 2
num_qubits = 2
omega_list = [5.0, 4.9]
g = 0.05
ds = 1.0728385125463975
cnot_dur = 239
cnot_p = 1.7554873088999543
cnot_sigma = 1.5


# Check echo
_, _, echo = CNOT_pulseEcho(PHI_PLUS_NO_CNOT, 0, 1, omega_list, g, ds, cnot_dur, cnot_p, cnot_sigma)

_, _, no_echo = CNOT_pulseNoEcho(PHI_PLUS_NO_CNOT, 0, 1, omega_list, g, ds, cnot_dur, cnot_p, cnot_sigma)

print(echo, no_echo)

print(statevector_similarity(echo, no_echo))

print(statevector_similarity(echo, PHI_PLUS))

print(statevector_similarity(no_echo, PHI_PLUS))

# GATE LEVEL:
# qm = TestCircuit
# num_layer = qm.num_layer
# num_qubits = qm.num_qubits
# num_gates = qm.num_gates
# simulator = 'statevector_simulator'
#
#
# samples = 1
#
# num_coeffs = 5
# points = 1
#
# x = np.linspace(1, 1, points)
# parameter = fixed_parameter_set(num_layer, num_qubits, num_gates, samples, theta)
#
#
# gate_qm = qm(parameter[0])
# _, expected_statevector = predict_single(gate_qm, simulator, shots, None)
#
# print(expected_statevector.data)

# PULSE LEVEL
#
# init_state = PLUS_ZERO_STATE
# expected_state = PHI_PLUS
#
# omega_list = [5.0, 4.9]
#
# tries = 50
#
# gs = np.random.uniform(0.02, 2, tries)     # 0.937-0.939 highest fid
# dss = np.linspace(0.1, 1.6, tries)  # Needs calibration 0.288-0.29 highest fid. phase = pi/2: np.linspace(0.26842105263157895, 0.322, tries)
#
# ps = np.random.uniform(0, 2*np.pi, tries)
# durations = np.random.randint(120, 300, tries)
#
# for try_ in range(tries):
#     g = 0.05
#     ds = 1.0728385125463975
#     cnot_dur = 239
#     cnot_p = 1.7554873088999543
#     cnot_sigma = 1.5
#
#     # g = gs[try_]
#     ds = dss[try_]
#     # p = ps[try_]
#     # d = durations[try_]
#
#     print("#"*20)
#     _, _, state_afterCNOT = CNOT_pulseEcho(init_state, 0, 1, omega_list, g=g, drive_strength=ds, cnot_duration=cnot_dur, cnot_phase=cnot_p,
#                                            cnot_sigma=cnot_sigma)
#     sim = statevector_similarity(state_afterCNOT, expected_state.data)
#
#     if sim > 0.5:
#         print("#"*20)
#         print("found!")
#         print(f"similarity: {sim}, drive strength: {ds}, g: {g}, phase: {cnot_p}, duration: {cnot_dur}, sigma: {cnot_sigma}")
#         print(state_afterCNOT)
#     elif sim > 0.2:
#         print(f"similarity: {sim}, drive strength: {ds}, g: {g}, phase: {cnot_p}, duration: {cnot_dur}, sigma: {cnot_sigma}")
#     else:
#         print(None)



