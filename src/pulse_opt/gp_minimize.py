import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
import joblib

from pulse.pulse_gates import *

expected_states = [
    PHI_PLUS,
    PSI_PLUS,
    PHI_MINUS,
    PSI_MINUS
]

init_states = [
    PHI_PLUS_NO_CNOT,
    PSI_PLUS_NO_CNOT,
    PHI_MINUS_NO_CNOT,
    PSI_MINUS_NO_CNOT
]


def CNOT_correctness(g, ds, cnot_dur, cnot_p, cnot_sigma):
    similarities = []
    for init_state, exp_state in zip(init_states, expected_states):
        _, _, state_afterCNOT = CNOT_pulseEcho(init_state, 0, 1, [5.0, 4.9], g=g, drive_strength=ds, cnot_duration=cnot_dur, cnot_phase=cnot_p,
                                               cnot_sigma=cnot_sigma)
        sim_score = statevector_similarity(state_afterCNOT, exp_state.data)
        similarities.append(sim_score)
    return np.mean(similarities)  # try other aggregations...


search_space = [
    Real(0.02, 1, name='g'),
    Real(0.05, 1.7, name='ds'),
    Integer(120, 300, name='cnot_dur'),
    Real(0, 2 * np.pi, name='cnot_p'),
    Real(0.5, 5, name='cnot_sigma')
]


def objective_function(params):
    g, ds, cnot_dur, cnot_p, cnot_sigma = params
    return -CNOT_correctness(g, ds, cnot_dur, cnot_p, cnot_sigma)  # Negate for maximization


n_calls_per_batch = 200  # Adjust as needed to fit your memory constraints
n_initial_points = 12
initial_points = []
initial_values = []

print("Computing initial values...")

for _ in range(n_initial_points):
    initial_params = [
        np.random.uniform(0.02, 1),
        np.random.uniform(0.05, 1.7),
        np.random.randint(120, 300),
        np.random.uniform(0, 2 * np.pi),
        np.random.uniform(0.5, 5),
    ]
    initial_points.append(initial_params)
    initial_values.append(objective_function(initial_params))

num_restarts = 3
best_results = []

for seed in range(num_restarts):
    print("seed: ", seed)
    current_x0 = initial_points.copy()
    current_y0 = initial_values.copy()
    total_calls = 0

    batch_number = 1  # Initialize batch counter

    while total_calls < 1000:
        remaining_calls = min(n_calls_per_batch, 1000 - total_calls)  # Ensure you don't exceed 1000 calls
        print(f"Running batch {batch_number} with {remaining_calls} calls (total: {total_calls})...")

        result = gp_minimize(
            objective_function,
            search_space,
            n_calls=remaining_calls,
            random_state=seed,
            n_jobs=-1,
            x0=current_x0,
            y0=current_y0,
        )

        total_calls += remaining_calls
        current_x0 = result.x_iters  # Use the last points from the batch as the next starting points.
        current_y0 = result.func_vals  # use the last objective function values from the batch.
        best_results.append(result)
        joblib.dump(best_results, f"batch_results_seed_{seed}.joblib")  # safe each batch result

        best_params = result.x
        best_value = -result.fun

        print(f"Restart {seed + 1}, Batch {batch_number}:")
        print(f"  Best parameters: g={best_params[0]}, ds={best_params[1]}, cnot_dur={best_params[2]}, cnot_p={best_params[3]}, cnot_sigma={best_params[4]}")
        print(f"  Maximum average similarity: {best_value}")
        print("-" * 20)
        print("\nIndividual results for the best parameters:")
        for initial_statevector, expected_statevector in zip(init_states, expected_states):
            _, _, CNOTEcho_state = CNOT_pulseEcho(initial_statevector, 0, 1, [5.0, 4.9], g=best_params[0], drive_strength=best_params[1],
                                                  cnot_duration=best_params[2], cnot_phase=best_params[3], cnot_sigma=best_params[4])
            sim = statevector_similarity(CNOTEcho_state, expected_statevector.data)
            print("Expected:")
            prints(expected_statevector)
            print("CNOTEcho:")
            prints(CNOTEcho_state)
            print(f"Similarity: {sim}")
        print("-" * 20)
        print("\n")

        batch_number += 1

    # find the best one
    loaded_results = joblib.load(f"batch_results_seed_{seed}.joblib")
    final_best_result = min(loaded_results, key=lambda r: r.fun)

    final_best_params = final_best_result.x
    final_best_value = -final_best_result.fun

    print(f"Restart {seed + 1}:")
    print(
        f"Final Best parameters: g={final_best_params[0]}, ds={final_best_params[1]}, cnot_dur={final_best_params[2]}, cnot_p={final_best_params[3]}, cnot_sigma={final_best_params[4]}")
    print(f"Final Maximum average similarity: {final_best_value}")




def objective_function(k, current_state):
    """Objective function to maximize statevector similarity."""
    theta = typical_theta()
    _, _, final_state = RZ_pulseSPEC(theta, current_state, "all", k)
    return -statevector_similarity(c.run_quick_circuit(theta), final_state[-1])  # Negate for maximization

def optimize_k2(initial_state, k2_bounds=(0.1, 8.0), n_calls=1000, n_init_values=20, seeds=[0, 1, 2]):
    """Optimizes k2 with multiple initial values and seeds."""

    search_space = [Real(k2_bounds[0], k2_bounds[1], name='k2')]
    best_overall_k2 = None
    best_overall_similarity = -float('inf')

    for seed in seeds:
        print(f"\nStarting optimization with seed: {seed}")
        initial_points = [[np.random.uniform(k2_bounds[0], k2_bounds[1])] for _ in range(n_init_values)]
        initial_values = [objective_function(k2, initial_state) for k2 in [point[0] for point in initial_points]]

        result = gp_minimize(
            lambda k2: objective_function(k2[0], initial_state),
            search_space,
            n_calls=n_calls,
            random_state=seed,
            n_jobs=-1,
            x0=initial_points,
            y0=initial_values,
        )

        best_k2 = result.x[0]
        best_similarity = -result.fun

        print(f"Seed {seed}: Best k: {best_k2}, Best Similarity: {best_similarity}")
        print("Best parameters from this seed:")
        print(result)

        if best_similarity > best_overall_similarity:
            best_overall_similarity = best_similarity
            best_overall_k2 = best_k2

    return best_overall_k2, best_overall_similarity
#
# num_q = 2
# c = PennyCircuit(num_q)
# current_state = GROUND_STATE(num_q)
# _, _, current_state = RX_pulseSPEC(np.pi/2, current_state, "all")
#
# best_k, best_similarity = optimize_k2(current_state[-1])
#
# print(f"Best k: {best_k}")
# print(f"Best Similarity: {best_similarity}")
#
# # Test for every theta:
# theta_values = [-np.pi, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, np.pi]
# print("\nTesting similarity for each theta:")
#
#
# for theta in theta_values:
#     _, _, final_state = RZ_pulseSPEC(theta, current_state[-1], "all", best_k)
#     similarity = statevector_similarity(c.run_quick_circuit(theta), final_state[-1])
#     print(f"Theta: {theta}, Similarity: {similarity}")

