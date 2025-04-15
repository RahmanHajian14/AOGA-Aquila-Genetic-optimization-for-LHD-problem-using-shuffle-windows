import numpy as np

def apply_rov_rule(continuous_column, discrete_values):
    sorted_indices = np.argsort(continuous_column.flatten())
    discrete_col = np.zeros_like(continuous_column, dtype=int).flatten()
    for i, idx in enumerate(sorted_indices):
        discrete_col[idx] = discrete_values[i]
    return discrete_col.reshape(continuous_column.shape)
