import numpy as np


def __normalize(adjacency_matrix):
    adjacency_matrix = adjacency_matrix / np.sum(adjacency_matrix, axis=0)
    return adjacency_matrix


def __expand(a, m):
    return np.linalg.matrix_power(a, m)


def __inflate(a, r):
    return __normalize(np.power(a, r))


def __get_clusters(a):
    clusters = []
    for i, r in enumerate((a > 0).tolist()):
        if r[i]:
            clusters.append(a[i, :] > 0)
    clustering_map = {}
    for cn, c in enumerate(clusters):
        for x in [i for i, x in enumerate(c) if x]:
            clustering_map[cn] = clustering_map.get(cn, []) + [x]
    return clustering_map


def __create_adj_matrix(initial_matrix, first_column, second_column):
    for i in range(first_column.size):
        x = int(first_column[i])
        y = int(second_column[i])
        # connecting edges, as it is undirected graph edge is set both ways.
        initial_matrix[x][y] = 1.0
        initial_matrix[y][x] = 1.0
        # Adding self loops
        initial_matrix[x][x] = 1.0
        initial_matrix[y][y] = 1.0
    # removing the extra row and column
    initial_matrix = np.delete(initial_matrix, 0, 0)
    initial_matrix = np.delete(initial_matrix, 0, 1)
    return initial_matrix


def buildCluMap(clusters):
    custMap = {}
    for key in clusters:
        listele = clusters.get(key)
        for ele in listele:
            custMap[ele] = key
    return custMap


def markov(graph, max_loop=1000):
    """
    An Efficient Algorithm for Large-scale Detection of Protein Families (Nucleic Acids Research 2002)
    Anton Enright, Stijn Van Dongen, and Christos Ouzounis

    https://github.com/HarshHarwani/markov-clustering-for-graphs

    :param graph:
    :param max_loop:
    :return:
    """
    edges = np.array(list(map(np.float64, graph.edges())))

    # taking the max from the array to create new matrix of required dimension
    max_value = int(np.amax(edges))

    # extracting two columns from the original two dimensional array
    first_column = edges[:, 0]
    second_column = edges[:, 1]

    # creating the initial empty array, creating one extra dimension to start the index from 1.
    inital_matrix = np.ndarray(shape=(max_value + 1, max_value + 1))

    # creating the adjacencyMatrix
    adjacency_matrix = __create_adj_matrix(inital_matrix, first_column, second_column)

    cls = {}

    for m in range(2, 7):
        for r in np.arange(1.1, 2.2, 0.2):

            # normalizing the matrix
            matrix = __normalize(adjacency_matrix)

            # configuration parameters
            for i in range(max_loop):
                # maintaining a previous copy for checking convergence
                prev = matrix.copy()
                matrix = __expand(matrix, m)
                matrix = __inflate(matrix, r)

                # convergence condition
                if np.array_equal(matrix, prev):
                    break

            clusters = __get_clusters(matrix)
            cls = clusters

    communities = []

    for part in cls.values():
        com = []
        for eid in part:
            com.extend(list(map(int, edges[eid])))
        communities.append(list(set(com)))

    return communities
