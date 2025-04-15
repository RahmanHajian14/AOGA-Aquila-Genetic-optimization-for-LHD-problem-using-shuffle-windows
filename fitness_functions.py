import numpy as np
import scipy.spatial.distance as distance
from parameters import PHI_P_VALUE


# Define the φp criterion function
def phi_p(lhd, p):
    pairwise_distances = distance.pdist(lhd, 'euclidean')
    weighted_distances = np.sum(pairwise_distances ** (-p))
    phi_p_value = (weighted_distances) ** (1 / p)
    return phi_p_value

# Define the maxmin criterion function
def maxmin(lhd):
    pairwise_distances = distance.pdist(lhd, 'euclidean')
    min_distance = np.min(pairwise_distances)
    return min_distance

# Fitness functions
def fitness_phi_p(lhd):
    return phi_p(lhd, PHI_P_VALUE)

def fitness_maxmin(lhd):
    return maxmin(lhd)
