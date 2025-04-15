import numpy as np
import random
import logging
import matplotlib.pyplot as plt
from func_plot import plot_lhd_improvement, plot_individual_fitness_histories,plot_aggregated_fitness_stats, compute_stats,plot_mean_std,plot_boxplot
from fitness_functions import fitness_phi_p, fitness_maxmin, PHI_P_VALUE
from aoga import ao_ga_algorithm, apply_move_within_shuffle_window
from parameters import *
from initialization import initialize_pop

# Set up logging
logging.basicConfig(level=logging.INFO)

# Main function to run the algorithm and plot results
def main():
    global iteration
    global moves

    population = initialize_pop(NUM_ROWS, NUM_COLUMNS, POP_SIZE)
    #fitness_phi_p_values = [fitness_phi_p(eagle) for eagle in population]
    #fitness_maxmin_values = [fitness_maxmin(eagle) for eagle in population]
    iteration = 1

    # Print initial fitness values
    initial_fitness_phi_p = fitness_phi_p(population[0])
    initial_fitness_maxmin = fitness_maxmin(population[0])
    logging.info(f"Initial Phi-p Fitness: {initial_fitness_phi_p}, Initial Maxmin Fitness: {initial_fitness_maxmin}")

    moves = []
    moves.append(
        lambda lhd, best_lhd, iteration, total_iterations, num_rows: apply_move_within_shuffle_window(lhd, best_lhd,
                                                                                                      iteration,
                                                                                                      total_iterations,
                                                                                                      num_rows))

    # Run the GA-AO algorithm
    initial_population, best_lhd_phi_p, best_lhd_maxmin, best_fitness_phi_p_history, global_best_fitness_phi_p_history, best_fitness_maxmin_history, global_best_fitness_maxmin_history, lhd_history = ao_ga_algorithm(
        moves)

    best_fitness_phi_p_value = fitness_phi_p(best_lhd_phi_p)
    best_fitness_maxmin_value = fitness_maxmin(best_lhd_maxmin)
    logging.info(f"Best Phi-p Fitness: {best_fitness_phi_p_value}, Best Maxmin Fitness: {best_fitness_maxmin_value}")

    return best_fitness_phi_p_value, best_fitness_maxmin_value, global_best_fitness_phi_p_history, global_best_fitness_maxmin_history

from parameters import POP_SIZE, NUM_COLUMNS, NUM_ROWS, NUM_ITERATIONS, SHUFFLE_WINDOW_SIZE, CROSS_WINDOW_SIZE
def run_multiple_times(n_runs):
    phi_p_fitnesses = []
    maxmin_fitnesses = []

    all_phi_p_histories = []
    all_maxmin_histories = []

    best_phi_p = float('inf')
    worst_phi_p = float('-inf')
    best_maxmin = float('-inf')
    worst_maxmin = float('inf')

    for run in range(n_runs):
        print(f"Run {run + 1}/{n_runs}")
        best_fitness_phi_p, best_fitness_maxmin, global_best_fitness_phi_p_history, global_best_fitness_maxmin_history = main()
        phi_p_fitnesses.append(best_fitness_phi_p)
        maxmin_fitnesses.append(best_fitness_maxmin)
        all_phi_p_histories.append(global_best_fitness_phi_p_history)
        all_maxmin_histories.append(global_best_fitness_maxmin_history)

        # Track the best and worst fitness values
        if best_fitness_phi_p < best_phi_p:
            best_phi_p = best_fitness_phi_p
        if best_fitness_phi_p > worst_phi_p:
            worst_phi_p = best_fitness_phi_p
        if best_fitness_maxmin > best_maxmin:
            best_maxmin = best_fitness_maxmin
        if best_fitness_maxmin < worst_maxmin:
            worst_maxmin = best_fitness_maxmin

        if (run + 1) % 10 == 0:
            max_phi_p = np.max(phi_p_fitnesses)
            min_phi_p = np.min(phi_p_fitnesses)
            mean_phi_p = np.mean(phi_p_fitnesses)
            std_phi_p = np.std(phi_p_fitnesses)

            print(f"After {run + 1} runs:")
            print(f"Max Phi-p Fitness: {max_phi_p:.4e}")
            print(f"Min Phi-p Fitness: {min_phi_p:.4e}")
            print(f"Mean Phi-p Fitness: {mean_phi_p:.4e}")
            print(f"STD Phi-p Fitness: {std_phi_p:.4e}")

    phi_p_mean = np.mean(phi_p_fitnesses)
    phi_p_std = np.std(phi_p_fitnesses)
    maxmin_mean = np.mean(maxmin_fitnesses)
    maxmin_std = np.std(maxmin_fitnesses)

    # Printing the most and the least fitness values
    print(f"\nMost and Least Fitness Values after {n_runs} runs:")
    print(f"Most Phi-p Fitness: {worst_phi_p:.4e}")
    print(f"Least Phi-p Fitness: {best_phi_p:.4e}")
    print(f"Most Maxmin Fitness: {best_maxmin:.4e}")
    print(f"Least Maxmin Fitness: {worst_maxmin:.4e}")

    print(f"Phi-p Fitness: Mean = {phi_p_mean}, STD = {phi_p_std}")
    print(f"Maxmin Fitness: Mean = {maxmin_mean}, STD = {maxmin_std}")

    # Plotting individual fitness histories for all runs
    plot_individual_fitness_histories(all_phi_p_histories, title='Fitness Evolution Over Iterations for All Runs (Phi-p)')
    return all_phi_p_histories, all_maxmin_histories

if __name__ == "__main__":
    all_phi_p_histories, all_maxmin_histories = run_multiple_times(n_runs)
