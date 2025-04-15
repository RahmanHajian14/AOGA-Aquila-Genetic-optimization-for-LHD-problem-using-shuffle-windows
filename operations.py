import numpy as np
import random
from rov_rule import apply_rov_rule
from parameters import *


# Crossover function for LHDs (GA operation) with cross_window
def crossover(selected_chromosomes, num_rows, num_columns):
    offspring = []
    for _ in range(POP_SIZE // 2):
        parent1 = random.choice(selected_chromosomes)
        parent2 = random.choice(selected_chromosomes)
        # CW (only apply crossover in the selected window area)
        child1 = parent1.copy()
        child2 = parent2.copy()
        # Define the starting point for CW
        row_start = np.random.randint(0, num_rows - CROSS_WINDOW_SIZE[0] + 1)
        col_start = np.random.randint(0, num_columns - CROSS_WINDOW_SIZE[1] + 1)
        # Extract CW from both parents
        window1 = child1[row_start:row_start + CROSS_WINDOW_SIZE[0], col_start:col_start + CROSS_WINDOW_SIZE[1]].copy()
        window2 = child2[row_start:row_start + CROSS_WINDOW_SIZE[0], col_start:col_start + CROSS_WINDOW_SIZE[1]].copy()
        # Perform crossover by swapping the windows between parents
        child1[row_start:row_start + CROSS_WINDOW_SIZE[0], col_start:col_start + CROSS_WINDOW_SIZE[1]] = window2
        child2[row_start:row_start + CROSS_WINDOW_SIZE[0], col_start:col_start + CROSS_WINDOW_SIZE[1]] = window1

        # Apply the ROV rule to ensure discrete values in the affected columns
        for j in range(col_start, col_start + CROSS_WINDOW_SIZE[1]):
            child1[:, j] = apply_rov_rule(child1[:, j].reshape(-1, 1), range(num_rows)).flatten()
            child2[:, j] = apply_rov_rule(child2[:, j].reshape(-1, 1), range(num_rows)).flatten()
        # Add the new offspring to the offspring list
        offspring.extend([child1, child2])

    return offspring


# Mutation function for LHDs (GA operation)
def mutation(offspring, mut_rate, num_rows):
    num_columns = NUM_COLUMNS
    mutated_offspring = []
    for lhd in offspring:
        for j in range(num_columns):
            if random.random() < mut_rate:
                rows = np.random.choice(num_rows, 2, replace=False)
                lhd[rows[0], j], lhd[rows[1], j] = lhd[rows[1], j], lhd[rows[0], j]
        mutated_offspring.append(lhd)
    return mutated_offspring


# Shuffle window function
def shuffle_window(lhd, k):
    n, m = lhd.shape
    k_rows, k_cols = k
    # Ensure the shuffle window size does not exceed matrix dimensions
    if k_rows > n or k_cols > m:
        raise ValueError("Shuffle window size exceeds matrix dimensions.")

    new_lhd = lhd.copy()
    row_start = np.random.randint(0, n - k_rows + 1)
    col_start = np.random.randint(0, m - k_cols + 1)
    window = new_lhd[row_start:row_start + k_rows, col_start:col_start + k_cols].copy()
    np.random.shuffle(window.ravel())
    new_lhd[row_start:row_start + k_rows, col_start:col_start + k_cols] = window.reshape(k_rows, k_cols)
    for j in range(col_start, col_start + k_cols):
        new_lhd[:, j] = np.random.permutation(new_lhd[:, j])
    return new_lhd


# Shuffle function
def shuffle(lhd):
    shuffled_lhds = []
    shuffled_lhd = shuffle_window(lhd, SHUFFLE_WINDOW_SIZE)
    shuffled_lhds.append(shuffled_lhd)
    return shuffled_lhds


# Simplified  AO-based move functions
def expanded_exploration(position, best_position, num_rows):
    scaling_factor = np.random.uniform(0.5, 1.5, position.shape)
    new_position = position + scaling_factor * np.random.randint(-2, 3, position.shape) * (best_position - position)
    new_position = np.clip(new_position, 0, num_rows - 1)
    for j in range(position.shape[1]):
        new_position[:, j] = apply_rov_rule(new_position[:, j].reshape(-1, 1), range(num_rows)).flatten()
    return new_position


def narrowed_exploration(position, best_position, num_rows):
    scaling_factor = np.random.uniform(0.8, 1.2, position.shape)
    new_position = position + scaling_factor * np.random.randint(-1, 2, position.shape) * (best_position - position)
    new_position = np.clip(new_position, 0, num_rows - 1)
    for j in range(position.shape[1]):
        new_position[:, j] = apply_rov_rule(new_position[:, j].reshape(-1, 1), range(num_rows)).flatten()
    return new_position


def expanded_exploitation(position, best_position, num_rows):
    scaling_factor = np.random.uniform(0.5, 1.5, position.shape)
    new_position = position + scaling_factor * np.sin(best_position - position + np.random.rand(*position.shape))
    new_position = np.clip(new_position, 0, num_rows - 1)
    for j in range(position.shape[1]):
        new_position[:, j] = apply_rov_rule(new_position[:, j].reshape(-1, 1), range(num_rows)).flatten()
    return new_position


def narrowed_exploitation(position, best_position, num_rows):
    scaling_factor = np.random.uniform(0.8, 1.2, position.shape)
    new_position = position + scaling_factor * np.cos(best_position - position + np.random.rand(*position.shape))
    new_position = np.clip(new_position, 0, num_rows - 1)
    for j in range(position.shape[1]):
        new_position[:, j] = apply_rov_rule(new_position[:, j].reshape(-1, 1), range(num_rows)).flatten()
    return new_position
