import numpy as np
import random
import logging
from fitness_functions import fitness_phi_p, fitness_maxmin
from rov_rule import apply_rov_rule
from operations import crossover, mutation, shuffle, shuffle_window
from operations import expanded_exploration, narrowed_exploration, expanded_exploitation, narrowed_exploitation
from parameters import *
from initialization import initialize_pop



# move function incorporating the four AO optimization moves
def move(position, best_position, iteration, total_iterations, num_rows):
    moves = [expanded_exploration, narrowed_exploration, expanded_exploitation, narrowed_exploitation]
    probabilities = [EXPANDED_EXPLORATION_PROB, NARROWED_EXPLORATION_PROB, EXPANDED_EXPLOITATION_PROB,
                     NARROWED_EXPLOITATION_PROB]
    chosen_move = np.random.choice(moves, p=probabilities)

    alpha = 1 - (iteration / total_iterations)

    new_position = chosen_move(position, best_position, num_rows)
    for j in range(position.shape[1]):
        new_position[:, j] = apply_rov_rule(new_position[:, j].reshape(-1, 1), range(num_rows)).flatten()
    if fitness_phi_p(new_position) < fitness_phi_p(position):
        position = new_position
    position = alpha * position + (1 - alpha) * best_position

    for j in range(position.shape[1]):
        position[:, j] = apply_rov_rule(position[:, j].reshape(-1, 1), range(num_rows)).flatten()
    return position


# Function to apply the move function within the shuffle window
def apply_move_within_shuffle_window(lhd, best_lhd, iteration, total_iterations, num_rows):
    shuffled_lhd = shuffle_window(lhd, SHUFFLE_WINDOW_SIZE)
    return move(shuffled_lhd, best_lhd, iteration, total_iterations, num_rows)


# Roulette wheel selection process for LHDs (GA operation)
def roulette_wheel_selection(population, fitness):
    total_fitness = sum(fitness)
    selection_probs = [f / total_fitness for f in fitness]
    cumulative_probs = np.cumsum(selection_probs)

    selected_individuals = []
    for _ in range(len(population) // 2):
        r = random.random()
        for i, cp in enumerate(cumulative_probs):
            if r < cp:
                selected_individuals.append(population[i])
                break

    return selected_individuals

# Replacement process for LHDs (GA operation)
def replace(population, new_generation, fitness, new_fitness, num_rows):
    combined = list(zip(population, fitness)) + list(zip(new_generation, new_fitness))
    combined_sorted = sorted(combined, key=lambda x: x[1])
    new_population = [x[0] for x in combined_sorted[:min(POP_SIZE, len(combined_sorted))]]
    # Apply ROV rule to ensure LHD properties in the new population
    for lhd in new_population:
        for j in range(lhd.shape[1]):
            lhd[:, j] = apply_rov_rule(lhd[:, j].reshape(-1, 1), range(num_rows)).flatten()
    return new_population


# Function to calculate the current diversification rate
def get_diversification_rate(current_iteration, total_iterations):
    return DIVERSIFICATION_RATE_INITIAL - (DIVERSIFICATION_RATE_INITIAL - DIVERSIFICATION_RATE_FINAL) * (
            current_iteration / total_iterations)


# Function to compute stats for fitness values
def compute_stats(fitness_histories):
    stats = {}
    for move, fitness_history in fitness_histories.items():
        mean_fitness = np.mean(fitness_history)
        std_fitness = np.std(fitness_history)
        stats[move] = (mean_fitness, std_fitness)
    return stats

# Main AO-GA loop
def ao_ga_algorithm(moves, iterations=NUM_ITERATIONS, print_every=PRINT_EVERY):
    population = initialize_pop(NUM_ROWS, NUM_COLUMNS, POP_SIZE)
    initial_population = np.copy(population)
    fitness_phi_p_values = [fitness_phi_p(eagle) for eagle in population]
    fitness_maxmin_values = [fitness_maxmin(eagle) for eagle in population]
    iteration = 1
    best_fitness_phi_p_history = []
    best_fitness_maxmin_history = []
    global_best_fitness_phi_p_history = []
    global_best_fitness_maxmin_history = []
    lhd_history = []

    global_best_fitness_phi_p = min(fitness_phi_p_values)
    global_best_lhd_phi_p = population[fitness_phi_p_values.index(global_best_fitness_phi_p)]

    while iteration <= iterations:
        selected = roulette_wheel_selection(population, fitness_phi_p_values)
        if random.random() < CROSSOVER_PROB:
            offspring = crossover(selected, NUM_ROWS, NUM_COLUMNS)
        else:
            offspring = selected.copy()

        if iteration > 1:
            prev_best_fitness = best_fitness_phi_p_history[-1]
            curr_best_fitness = min(fitness_phi_p_values)
            if curr_best_fitness >= prev_best_fitness:
                mut_rate = INITIAL_MUT_RATE * 5
            else:
                mut_rate = INITIAL_MUT_RATE * 0.2
        else:
            mut_rate = INITIAL_MUT_RATE

        if random.random() < MUTATION_PROB:
            mutated_offspring = mutation(offspring, mut_rate, NUM_ROWS)
        else:
            mutated_offspring = offspring

        new_fitness_phi_p_values = [fitness_phi_p(eagle) for eagle in mutated_offspring]
        new_fitness_maxmin_values = [fitness_maxmin(eagle) for eagle in mutated_offspring]

        for move in moves:
            new_offspring = []
            for i in range(len(mutated_offspring)):
                if move == shuffle:
                    if random.random() < SHUFFLE_PROB:
                        new_offspring.extend(move(mutated_offspring[i]))
                    else:
                        new_offspring.append(mutated_offspring[i])
                else:
                    new_offspring.append(
                        move(mutated_offspring[i], population[np.argmin(fitness_phi_p_values)], iteration,
                             NUM_ITERATIONS, NUM_ROWS))
            mutated_offspring = new_offspring

        for i in range(len(mutated_offspring)):
            for j in range(mutated_offspring[i].shape[1]):
                mutated_offspring[i][:, j] = apply_rov_rule(mutated_offspring[i][:, j].reshape(-1, 1),
                                                            range(NUM_ROWS)).flatten()
            new_fitness_phi_p_values[i] = fitness_phi_p(mutated_offspring[i])
            new_fitness_maxmin_values[i] = fitness_maxmin(mutated_offspring[i])

        # Descending Diversification Rate
        diversification_rate = get_diversification_rate(iteration, iterations)

        if iteration % 3 == 0:
            num_new_individuals = int(diversification_rate * POP_SIZE)
            new_individuals = initialize_pop(NUM_ROWS, NUM_COLUMNS, num_new_individuals)
            new_fitness_phi_p_values.extend([fitness_phi_p(eagle) for eagle in new_individuals])
            new_fitness_maxmin_values.extend([fitness_maxmin(eagle) for eagle in new_individuals])
            mutated_offspring.extend(new_individuals)

        population = replace(population, mutated_offspring, fitness_phi_p_values, new_fitness_phi_p_values, NUM_ROWS)
        fitness_phi_p_values = [fitness_phi_p(eagle) for eagle in population]
        fitness_maxmin_values = [fitness_maxmin(eagle) for eagle in population]

        best_fitness_phi_p = min(fitness_phi_p_values)
        best_fitness_maxmin = max(fitness_maxmin_values)
        best_lhd_phi_p = population[fitness_phi_p_values.index(best_fitness_phi_p)]
        best_lhd_maxmin = population[fitness_maxmin_values.index(best_fitness_maxmin)]

        best_fitness_phi_p_history.append(best_fitness_phi_p)
        best_fitness_maxmin_history.append(best_fitness_maxmin)

        # Update global best fitness and LHD if current best is better
        if best_fitness_phi_p < global_best_fitness_phi_p:
            global_best_fitness_phi_p = best_fitness_phi_p
            global_best_lhd_phi_p = best_lhd_phi_p.copy()

        global_best_fitness_phi_p_history.append(global_best_fitness_phi_p)
        global_best_fitness_maxmin_history.append(fitness_maxmin(global_best_lhd_phi_p))

        if iteration % print_every == 0 or iteration == iterations:
            logging.info(
                f"Iteration: {iteration}, Best Phi-p Fitness: {best_fitness_phi_p}, Best Maxmin Fitness: {best_fitness_maxmin}")

        if iteration % (iterations // 4) == 0:
            lhd_history.append((iteration, best_lhd_phi_p.copy()))

        iteration += 1

    return initial_population, best_lhd_phi_p, best_lhd_maxmin, best_fitness_phi_p_history, global_best_fitness_phi_p_history, best_fitness_maxmin_history, global_best_fitness_maxmin_history, lhd_history
