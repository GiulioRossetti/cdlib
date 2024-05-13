import numpy as np
from scipy.sparse import isspmatrix, dok_matrix, csc_matrix
import sklearn.preprocessing


def sparse_allclose(a, b, rtol=1e-5, atol=1e-8):
    """
    Version of np.allclose for use with sparse matrices
    """
    c = np.abs(a - b) - rtol * np.abs(b)
    # noinspection PyUnresolvedReferences
    return c.max() <= atol


def normalize(matrix):
    """
    Normalize the columns of the given matrix

    :param matrix: The matrix to be normalized
    :returns: The normalized matrix
    """
    return sklearn.preprocessing.normalize(matrix, norm="l1", axis=0)


def inflate(matrix, power):
    """
    Apply cluster inflation to the given matrix by raising
    each element to the given power.

    :param matrix: The matrix to be inflated
    :param power: Cluster inflation parameter
    :returns: The inflated matrix
    """
    if isspmatrix(matrix):
        return normalize(matrix.power(power))

    return normalize(np.power(matrix, power))


def expand(matrix, power):
    """
    Apply cluster expansion to the given matrix by raising
    the matrix to the given power.

    :param matrix: The matrix to be expanded
    :param power: Cluster expansion parameter
    :returns: The expanded matrix
    """

    if isspmatrix(matrix):
        return matrix**power
    else:
        import scipy.sparse as sp

        matrix = sp.csr_matrix(matrix)
        return matrix**power

    return np.linalg.matrix_power(matrix, power)


def add_self_loops(matrix, loop_value):
    """
    Add self-loops to the matrix by setting the diagonal
    to loop_value

    :param matrix: The matrix to add loops to
    :param loop_value: Value to use for self-loops
    :returns: The matrix with self-loops
    """
    shape = matrix.shape
    assert shape[0] == shape[1], "Error, matrix is not square"

    if isspmatrix(matrix):
        new_matrix = matrix.todok()
    else:
        new_matrix = matrix.copy()

    for i in range(shape[0]):
        new_matrix[i, i] = loop_value

    if isspmatrix(matrix):
        return new_matrix.tocsc()

    return new_matrix


def prune(matrix, threshold):
    """
    Prune the matrix so that very small edges are removed.
    The maximum value in each column is never pruned.

    :param matrix: The matrix to be pruned
    :param threshold: The value below which edges will be removed
    :returns: The pruned matrix
    """
    if isspmatrix(matrix):
        pruned = dok_matrix(matrix.shape)
        pruned[matrix >= threshold] = matrix[matrix >= threshold]
        pruned = pruned.tocsc()
    else:
        pruned = matrix.copy()
        pruned[pruned < threshold] = 0

    # keep max value in each column. same behaviour for dense/sparse
    num_cols = matrix.shape[1]
    row_indices = matrix.argmax(axis=0).reshape((num_cols,))
    col_indices = np.arange(num_cols)
    pruned[row_indices, col_indices] = matrix[row_indices, col_indices]

    return pruned


def converged(matrix1, matrix2):
    """
    Check for convergence by determining if
    matrix1 and matrix2 are approximately equal.

    :param matrix1: The matrix to compare with matrix2
    :param matrix2: The matrix to compare with matrix1
    :returns: True if matrix1 and matrix2 approximately equal
    """
    if isspmatrix(matrix1) or isspmatrix(matrix2):
        return sparse_allclose(matrix1, matrix2)

    return np.allclose(matrix1, matrix2)


def iterate(matrix, expansion, inflation):
    """
    Run a single iteration (expansion + inflation) of the mcl algorithm

    :param matrix: The matrix to perform the iteration on
    :param expansion: Cluster expansion factor
    :param inflation: Cluster inflation factor
    """

    # matrix = matrix.todense()

    # Expansion
    matrix = expand(matrix, expansion)

    # Inflation
    matrix = inflate(matrix, inflation)

    return matrix


def get_clusters(matrix):
    """
    Retrieve the clusters from the matrix

    :param matrix: The matrix produced by the MCL algorithm
    :returns: A list of tuples where each tuple represents a cluster and
              contains the indices of the nodes belonging to the cluster
    """
    if not isspmatrix(matrix):
        # cast to sparse so that we don't need to handle different
        # matrix types
        matrix = csc_matrix(matrix)

    # get the attractors - non-zero elements of the matrix diagonal
    attractors = matrix.diagonal().nonzero()[0]

    # somewhere to put the clusters
    clusters = set()

    # the nodes in the same row as each attractor form a cluster
    for attractor in attractors:
        cluster = tuple(matrix.getrow(attractor).nonzero()[1].tolist())
        clusters.add(cluster)

    return sorted(list(clusters))


def run_mcl(
    matrix,
    expansion=2,
    inflation=2,
    loop_value=1,
    iterations=100,
    pruning_threshold=0.001,
    pruning_frequency=1,
    convergence_check_frequency=1,
    verbose=False,
):
    """
    Perform MCL on the given similarity matrix

    :param matrix: The similarity matrix to cluster
    :param expansion: The cluster expansion factor
    :param inflation: The cluster inflation factor
    :param loop_value: Initialization value for self-loops
    :param iterations: Maximum number of iterations
           (actual number of iterations will be less if convergence is reached)
    :param pruning_threshold: Threshold below which matrix elements will be set
           set to 0
    :param pruning_frequency: Perform pruning every 'pruning_frequency'
           iterations.
    :param convergence_check_frequency: Perform the check for convergence
           every convergence_check_frequency iterations
    :param verbose: Print extra information to the console
    :returns: The final matrix
    """
    assert expansion > 1, "Invalid expansion parameter"
    assert inflation > 1, "Invalid inflation parameter"
    assert loop_value >= 0, "Invalid loop_value"
    assert iterations > 0, "Invalid number of iterations"
    assert pruning_threshold >= 0, "Invalid pruning_threshold"
    assert pruning_frequency > 0, "Invalid pruning_frequency"
    assert convergence_check_frequency > 0, "Invalid convergence_check_frequency"

    # Initialize self-loops
    if loop_value > 0:
        matrix = add_self_loops(matrix, loop_value)

    # Normalize
    matrix = normalize(matrix)

    # iterations
    for i in range(iterations):

        # store current matrix for convergence checking
        last_mat = matrix.copy()

        # perform MCL expansion and inflation

        matrix = iterate(matrix, expansion, inflation)

        # prune
        if pruning_threshold > 0 and i % pruning_frequency == pruning_frequency - 1:
            matrix = prune(matrix, pruning_threshold)

        # Check for convergence
        if i % convergence_check_frequency == convergence_check_frequency - 1:
            if converged(matrix, last_mat):
                break

    return matrix
