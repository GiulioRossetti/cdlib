#!/usr/bin/env python
# encoding:UTF-8

################################################################################################
#
#    Community Detection via Local Spectral Clustering
#
################################################################################################

# (Our algorithm is also known as "LEMON", which is the short form of Local Expansion via Minimum One Norm)


# LEMON.py
# Yixuan Li
# Last modified: 2015-1-8
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.


import numpy as np
import math
import pulp
from scipy import linalg as splin
import gc


def __set_initial_prob(n, starting_nodes):
    """Precondition: starting_nodes is ndarray which indicate the indices of starting points

    Return: A probability vector with n elements
    """
    v = np.zeros(n)
    v[starting_nodes] = 1.0 / starting_nodes.size

    return v


def __set_initial_prob_proportional(n, degree_sequence, starting_nodes):
    """Precondition: starting_nodes is ndarray which indicate the indices of starting points

    Return: A probability vector with n elements
    """
    v = np.zeros(n)
    vol = 0
    for node in starting_nodes:
        vol += degree_sequence[node]
    for node in starting_nodes:
        v[node] = degree_sequence[node] / float(vol)

    return v


def __adj_to_Laplacian(G):
    """Computes the normalized adjacency matrix of a given graph"""

    n = G.shape[0]
    D = np.zeros((1, n))
    for i in range(n):
        D[0, i] = math.sqrt(G[i, :].sum())

    temp = np.dot(D.T, np.ones((1, n)))
    horizontal = G / temp
    normalized_adjacency_matrix = horizontal / (temp.T)
    gc.collect()

    return normalized_adjacency_matrix


def __cal_conductance(G, cluster):
    """cluster: a list of node id that forms a algorithms. Data type of cluster is given by numpy array

    Calculate the conductance of the cut A and complement of A.
    """

    assert (
        type(cluster) == np.ndarray
    ), "The given algorithms members is not a numpy array"

    temp = G[cluster, :]
    subgraph = temp[:, cluster]
    cutsize = temp.sum() - subgraph.sum()
    denominator = min(temp.sum(), G.sum() - temp.sum())
    conductance = cutsize / denominator if denominator > 0 else 1

    return conductance


def __random_walk(G, initial_prob, subspace_dim=3, walk_steps=3):
    """
    Start a random walk with probability distribution p_initial.
    Transition matrix needs to be calculated according to adjacent matrix G.

    """
    assert (
        type(initial_prob) == np.ndarray
    ), "Initial probability distribution is \
                                             not a numpy array"

    # Transform the adjacent matrix to a laplacian matrix P
    P = __adj_to_Laplacian(G)

    Prob_Matrix = np.zeros((G.shape[0], subspace_dim))
    Prob_Matrix[:, 0] = initial_prob
    for i in range(1, subspace_dim):
        Prob_Matrix[:, i] = np.dot(Prob_Matrix[:, i - 1], P)

    Orth_Prob_Matrix = splin.orth(Prob_Matrix)

    for i in range(walk_steps):
        temp = np.dot(Orth_Prob_Matrix.T, P)
        Orth_Prob_Matrix = splin.orth(temp.T)

    return Orth_Prob_Matrix


def __min_one_norm(B, initial_seed, seed):
    weight_initial = 1 / float(len(initial_seed))
    weight_later_added = weight_initial / float(0.5)
    difference = len(seed) - len(initial_seed)
    [r, c] = B.shape
    prob = pulp.LpProblem("Minimum one norm", pulp.LpMinimize)
    indices_y = range(0, r)
    y = pulp.LpVariable.dicts("y_s", indices_y, 0)
    indices_x = range(0, c)
    x = pulp.LpVariable.dicts("x_s", indices_x)

    f = dict(zip(indices_y, [1.0] * r))

    prob += pulp.lpSum(f[i] * y[i] for i in indices_y)  # objective function

    prob += pulp.lpSum(y[s] for s in initial_seed) >= 1

    prob += pulp.lpSum(y[r] for r in seed) >= 1 + weight_later_added * difference

    for j in range(r):
        temp = dict(zip(indices_x, list(B[j, :])))
        prob += pulp.lpSum(y[j] + (temp[k] * x[k] for k in indices_x)) == 0

    prob.solve()

    result = []
    for var in indices_y:
        result.append(y[var].value())

    return result


def __global_minimum(sequence, start_index):
    detected_size = len(list(sequence))
    seq_length = len(list(sequence))
    cond = sequence[seq_length - 2]
    for x in range(40):
        list(sequence).append(0)
    for i in range(seq_length - 40):
        if sequence[i] < sequence[i - 1] and sequence[i] < sequence[i + 1]:
            count_larger = 0
            count_smaller = 0
            for j in range(1, 32):
                if sequence[i + 1 + j] > sequence[i + 1]:
                    count_larger += 1
            for k in range(1, 32):
                if sequence[i - 1 - k] > sequence[i - 1]:
                    count_smaller += 1
            if count_larger >= 18 and count_smaller >= 18:
                detected_size = i + start_index
                cond = sequence[i]
                break
    return detected_size, cond


def lemon(
    G,
    seedset,
    min_comm_size,
    max_comm_size,
    expand_step=None,
    subspace_dim=None,
    walk_steps=None,
    biased=True,
):
    degree = []
    n = G.shape[0]
    for x in range(n):
        degree.append(G[x].sum())

    # Random walk starting from seed nodes:
    if biased:
        initial_prob = __set_initial_prob_proportional(n, degree, seedset)
    else:
        initial_prob = __set_initial_prob(G.shape[0], seedset)

    Orth_Prob_Matrix = __random_walk(G, initial_prob, subspace_dim, walk_steps)
    initial_seed = seedset

    # Initialization
    detected = list(seedset)
    seed = seedset
    step = expand_step
    detected_comm = []

    global_conductance = np.zeros(30)
    global_conductance[-1] = 1000000  # set the last element to be infinitely large
    global_conductance[-2] = 1000000
    flag = True
    iteration = 0

    while iteration < 30 and flag:
        temp = np.argsort(
            np.array(__min_one_norm(Orth_Prob_Matrix, list(initial_seed), list(seed)))
        )

        sorted_top = list(temp[::-1][:step])

        detected = list(set(list(detected) + sorted_top))
        seed = np.array(detected)

        conductance_record = np.zeros(max_comm_size - min_comm_size + 1)
        conductance_record[-1] = 0

        for i in range(min_comm_size, max_comm_size):
            candidate_comm = np.array(list(temp[::-1][:i]))
            conductance_record[i - min_comm_size] = __cal_conductance(G, candidate_comm)

        detected_size, cond = __global_minimum(conductance_record, min_comm_size)

        step += expand_step

        if biased:
            initial_prob = __set_initial_prob_proportional(n, degree, seedset)
        else:
            initial_prob = __set_initial_prob(G.shape[0], seedset)

        Orth_Prob_Matrix = __random_walk(G, initial_prob, subspace_dim, walk_steps)

        if detected_size != 0:
            current_comm = list(temp[::-1][:detected_size])
            detected_comm = current_comm

        global_conductance[iteration] = cond
        if (
            global_conductance[iteration - 1] <= global_conductance[iteration]
            and global_conductance[iteration - 1] <= global_conductance[iteration - 2]
        ):
            flag = False

        iteration += 1

    return detected_comm
