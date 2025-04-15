

# Parameters
POP_SIZE = 20  # Population size
NUM_ROWS = 10  # Number of rows in each LHD
NUM_COLUMNS = 10  # Number of columns in each LHD
NUM_ITERATIONS = 500 # Number of iterations
INITIAL_MUT_RATE = 0.2  # Initial mutation rate
DIVERSIFICATION_RATE_INITIAL = 0.4  # Initial diversification rate
DIVERSIFICATION_RATE_FINAL = 0.1  # Final diversification rate
SHUFFLE_WINDOW_SIZE = (2, 1)  # Shuffle window size (rows, columns)
CROSS_WINDOW_SIZE = (3,2)  # Cross window size (rows, columns)

PHI_P_VALUE = 50  # Value of p in fitness_phi_p function
PRINT_EVERY = 200  # Print progress every 50 iterations


EXPANDED_EXPLORATION_PROB = 0.25
NARROWED_EXPLORATION_PROB = 0.25
EXPANDED_EXPLOITATION_PROB = 0.25
NARROWED_EXPLOITATION_PROB = 0.25

n_runs = 3

# Hyperparameters for functions
CROSSOVER_PROB = 0.35  # Probability for crossover
MUTATION_PROB = 0.1  # Probability for mutation
SHUFFLE_PROB = 0.55  # Probability for shuffle window

# Ensure the sum of probabilities does not exceed 1
if CROSSOVER_PROB + MUTATION_PROB + SHUFFLE_PROB > 1:
    raise ValueError("The sum of crossover, mutation, and shuffle probabilities must not exceed 1.")