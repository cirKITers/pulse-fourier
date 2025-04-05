# import torch
#
# from src.pulse_gates import *
#
#
# def obj_function(x, params):
#     g, ds, cnot_dur, cnot_p, cnot_sigma = params
#     current_state = x
#     control_qubit = 0
#     target_qubit = 1
#     omega_list = [5.0, 4.9]
#     _, _, state = CNOT_pulseEcho(current_state, control_qubit, target_qubit, omega_list, g, ds, cnot_dur, cnot_p, cnot_sigma)
#     return torch.tensor(state.data, dtype=torch.complex128)
#
#
# def loss(x, y):
#     return 1 - statevector_similarity(x.detach().numpy(), y.detach().numpy())
#
# def batch_loss(params, batch):
#     total_batch_loss = 0.0
#     for x, y in batch:
#         output_state = obj_function(x, params)
#         total_batch_loss += loss(output_state, y)
#     return total_batch_loss / len(batch)
#
#
# data_pairs = [
#     (torch.tensor(PHI_PLUS_NO_CNOT.data, dtype=torch.complex128), torch.tensor(PHI_PLUS.data, dtype=torch.complex128)),
#     (torch.tensor(PSI_PLUS_NO_CNOT.data, dtype=torch.complex128), torch.tensor(PSI_PLUS.data, dtype=torch.complex128)),
#     (torch.tensor(PHI_MINUS_NO_CNOT.data, dtype=torch.complex128), torch.tensor(PHI_MINUS.data, dtype=torch.complex128)),
#     (torch.tensor(PSI_MINUS_NO_CNOT.data, dtype=torch.complex128), torch.tensor(PSI_MINUS.data, dtype=torch.complex128)),
# ]
#
#
# def generate_dataset(num_samples):
#     dataset = []
#     for i in range(num_samples):
#         dataset.append(data_pairs[np.random.randint(len(data_pairs))])
#     return dataset
#
#
# parameter = torch.tensor([0.05, 1.0728385125463975, 239, 1.7554873088999543, 1.5], requires_grad=True)  # Initial parameters
#
# optimizer = torch.optim.Adam([parameter], lr=0.01)
#
#
# num_epochs = 100
# batch_size = 4
# for epoch in range(num_epochs):
#     batch = generate_dataset(batch_size)
#     total_loss = 0.0
#     for x, y in batch:
#         output_state = obj_function(x, parameter)
#         loss_val = loss(output_state, y)
#         total_loss += loss_val.item()       #.item() converts tensor to scalar.
#
#     total_loss /= batch_size
#     optimizer.zero_grad()
#     total_loss.backward()
#     optimizer.step()
#     print(f"Epoch {epoch}, Loss: {total_loss}")
#
# print("Optimized Parameters:", parameter)
#
#
#
#
