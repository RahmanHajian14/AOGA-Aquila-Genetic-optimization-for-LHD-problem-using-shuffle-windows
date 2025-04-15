import numpy as np

# Function to initialize a random population of LHDs
def initialize_pop(num_rows, num_columns, pop_size):
    population = []
    for _ in range(pop_size):
        lhd = np.zeros((num_rows, num_columns), dtype=int)
        for j in range(num_columns):
            lhd[:, j] = np.random.permutation(num_rows)
        population.append(lhd)
    return population