import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
import joblib

from src.pulse_gates import *

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


