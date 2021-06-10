import numpy as np
import collections as co

try:
    import igraph as ig
except ModuleNotFoundError:
    ig = None
import operator
import itertools
import argparse


#############################
# Fuzzy Modularity Measures #
#############################


def nepusz_modularity(G, cover):
    raise NotImplementedError("See the CONGA 2010 paper")


def zhang_modularity(G, cover):
    raise NotImplementedError(
        """See 'Identification of overlapping algorithms structure in
        complex networks using fuzzy C-means clustering'"""
    )


def nicosia_modularity(G, cover):
    raise NotImplementedError(
        """See 'Extending the definition of
        modularity to directed graphs with overlapping communities'"""
    )


#############################
# Crisp modularity measures #
#############################


def count_communities(G, cover):
    """
    Helper for lazar_modularity.
    Returns a dict {v:count} where v is a vertex id and
    count is the number of different communities it is
    assigned to.
    """
    counts = {i.index: 0 for i in G.vs}
    for community in cover:
        for v in community:
            counts[v] += 1
    return counts


def get_weights(G):
    """
    Given a graph G, returns a list of weights. If the graph is unweighted,
    returns a list of 1s the same length as the number of edges.
    """
    try:
        # asssumes weight as an attribute name means graph is weighted.
        weights = G.es["weight"]
    except KeyError:
        # unweighted means all weights are 1.
        weights = [1 for e in G.es]
    return weights


def get_single_lazar_modularity(G, community, weights, counts):
    """
    Returns the lazar modularity of a single algorithms.
    """
    totalInternalWeight = sum(weights[G.es[e].index] for e in community)  # m_c in paper
    numVerticesInCommunity = len(community)  # V_c in paper
    numPossibleInternalEdges = numVerticesInCommunity * (numVerticesInCommunity - 1) / 2
    if numPossibleInternalEdges == 0:
        return 0
    edgeDensity = (
        totalInternalWeight / numPossibleInternalEdges / numVerticesInCommunity
    )
    interVsIntra = 0
    comm = set(community)
    for v in community:
        interVsIntraInternal = 0
        neighbors = G.neighbors(v)
        degree = len(neighbors)  # k_i in paper
        numCommunitiesWithin = counts[v]  # s_i in paper
        for n in neighbors:
            weight = weights[G.get_eid(v, n)]
            if n in comm:
                interVsIntraInternal += weight
            else:
                interVsIntraInternal -= weight
        interVsIntraInternal /= degree * numCommunitiesWithin
        interVsIntra += interVsIntraInternal
    return edgeDensity * interVsIntra


def lazar_modularity(G, cover):
    """
    Returns the crisp modularity measure as defined by Lazar et al. 2009
    Defined as the average edge density times normalized difference
    between inter and intracommunity edges for each algorithms.
    See CONGA 2010 or Lazar's paper for a precise definition.
    """
    numCommunities = len(cover)  # |C| in the paper
    totalModularity = 0  # M_c in the paper
    weights = get_weights(G)
    counts = count_communities(G, cover)
    for c in cover:
        totalModularity += get_single_lazar_modularity(G, c, weights, counts)
    averageModularity = 1 / numCommunities * totalModularity  #  M in the paper
    return averageModularity


##################################
# Classes for overlapping covers #
##################################


class CrispOverlap(object):
    """
    TODO
    """

    def __init__(
        self,
        graph,
        covers,
        modularities=None,
        optimal_count=None,
        modularity_measure="lazar",
    ):
        """
        Initializes a CrispOverlap object with the given parameters.
            Graph: The graph to which the object refers
            covers: a dict of VertexCovers, also referring to this graph, of the form {k : v}
                where k is the number of clusters and v is the vertexCluste
            modularities (optional): a dict of modularities of the form {c:m} where c is
                the number of clusters and m is the modularity.
            optimal_count (optional): A hint for the number of clusters to use.
            modularity_measure (optional): The name of the modularity function to use.
                Right now, the only choice is "lazar."
        """
        # Possibly figure out a better data structure like a merge
        # list that contains all information needed?

        # So far only know of Lazar's measure for crisp overlapping.
        self._measureDict = {"lazar": lazar_modularity}
        self._covers = covers
        self._graph = graph
        self._optimal_count = optimal_count
        self._modularities = modularities
        if modularity_measure in self._measureDict:
            self._modularity_measure = modularity_measure
        else:
            raise KeyError("Modularity measure not found.")

    def __getitem__(self, numClusters):
        """
        Returns the cover with the given number of clusters.
        """
        if not numClusters:
            raise KeyError("Number of clusters must be a positive integer.")
        return self._covers[numClusters]

    def __iter__(self):
        """
        Iterates over the covers in the list.
        """
        return (v for v in list(self._covers.values()))

    def __len__(self):
        """
        Returns the number of covers in the list.
        """
        return len(self._covers)

    def __bool__(self):
        """
        Returns True when there is at least one cover in the list.
        """
        return bool(self._covers)

    def __str__(self):
        """
        Returns a string representation of the list of covers.
        """
        return "{0} vertices in {1} possible covers.".format(
            len(self._graph.vs), len(self._covers)
        )

    def as_cover(self):
        """
        Returns the optimal cover (by modularity) from the object.
        """
        return self._covers[self.optimal_count]

    def recalculate_modularities(self):
        """
        Recalculates the modularities and optimal count using the modularity_measure.
        """
        modDict = {}
        for cover in self._covers.values():
            modDict[len(cover)] = self._measureDict[self._modularity_measure](
                self._graph, cover
            )
        self._modularities = modDict
        self._optimal_count = max(
            iter(self._modularities.items()), key=operator.itemgetter(1)
        )[0]
        return self._modularities

    @property
    def modularities(self):
        """
        Returns the a dict {c : m} where c is the number of clusters
        in the cover and m is the modularity. If modularity has not
        been calculated, it recalculates it for all covers. Otherwise,
        it returns the stored dict.
        Note: Call recalculate_modularities to recalculate the modularity.
        """
        if self._modularities:
            return self._modularities
        self._modularities = self.recalculate_modularities()
        return self._modularities

    @property
    def optimal_count(self):
        """Returns the optimal number of clusters for this dendrogram.
        If an optimal count hint was given at construction time and
        recalculate_modularities has not been called, this property simply returns the
        hint. If such a count was not given, this method calculates the optimal cover
        by maximizing the modularity along all possible covers in the object.
        Note: Call recalculate_modularities to recalculate the optimal count.
        """
        if self._optimal_count is not None:
            return self._optimal_count
        else:
            modularities = self.modularities
            self._optimal_count = max(
                list(modularities.items()), key=operator.itemgetter(1)
            )[0]
            return self._optimal_count

    def pretty_print_cover(self, numClusters, label="CONGA_index"):
        """
        Takes a cover in vertex-id form and prints it nicely
        using label as each vertex's name.
        """
        cover = self._covers[numClusters]
        # if label == 'CONGA_index':
        pp = [self._graph.vs[num] for num in [cluster for cluster in cover]]
        # else:
        #    pp = [G.vs[num][label] for num in [cluster for cluster in cover]]
        for count, comm in enumerate(pp):
            print("Community {0}:".format(count))
            for v in comm:
                print("\t {0}".format(v.index if label == "CONGA_index" else v[label]))
            print()

    def make_fuzzy(self):
        """
        TODO. see CONGA 2010
        """
        pass


#################################################################################################################################################
# Possible optimizations and notes:
#   * Calculating the pair betweennesses is the large bottleneck.
#       * However, calculation of all-pairs-shortest-paths
#          and pair betweennesses are highly parallelizable.
#   * Keep a record of splits or merges?
#       * Right now, we store a lot of redundant information with a new
#           VertexCover item for every split.


def conga(OG, calculate_modularities=None, optimal_count=None):
    """
    Defines the CONGA algorithm outlined in the Gregory 2007 paper
    (An Algorithm to Find Overlapping Community Structure in Networks)
    Returns a CrispOverlap object of all of the covers.
    """

    G = OG.copy()

    comm = G.components()

    # Just in case the original graph is disconnected
    nClusters = len(comm)

    # Store the original ids of all vertices
    G.vs["CONGA_orig"] = [i.index for i in OG.vs]
    allCovers = {nClusters: ig.VertexCover(OG)}
    while G.es:
        split = remove_edge_or_split_vertex(G)
        if split:
            comm = G.components().membership
            cover = get_cover(G, OG, comm)
            nClusters += 1
            # short circuit stuff would go here.
            allCovers[nClusters] = cover
    if calculate_modularities is None:
        calculate_modularities = "lazar"
    return CrispOverlap(
        OG,
        allCovers,
        modularity_measure=calculate_modularities,
        optimal_count=optimal_count,
    )


def remove_edge_or_split_vertex(G):
    """
    The heart of the CONGA algorithm. Decides which edge should be
    removed or which vertex should be split. Returns True if the
    modification split the graph.
    """
    # has the graph split this iteration?
    split = False
    eb = G.edge_betweenness()

    maxIndex, maxEb = max(enumerate(eb), key=operator.itemgetter(1))
    # We might be able to calculate vertex betweenness and edge
    # betweenness at the same time. The current internal is slower
    # than the builtin, though.
    vb = G.betweenness()

    # Only consider vertices with vertex betweenness >= max
    # edge betweenness. From Gregory 2007 step 3
    vi = [i for i, b in enumerate(vb) if b > maxEb]

    edge = G.es[maxIndex].tuple

    if not vi:
        split = delete_edge(G, edge)
    else:
        pb = pair_betweenness(G, vi)
        maxSplit, vNum, splitInstructions = max_split_betweenness(G, pb)
        if maxSplit > maxEb:
            split = split_vertex(G, vNum, splitInstructions[0])
        else:
            split = delete_edge(G, edge)
    return split


def get_cover(G, OG, comm):
    """
    Given the graph, the original graph, and a algorithms
    membership list, returns a vertex cover of the communities
    referring back to the original algorithms.
    """
    coverDict = co.defaultdict(list)
    for i, community in enumerate(comm):
        coverDict[community].append(int(G.vs[i]["CONGA_orig"]))
    return ig.clustering.VertexCover(OG, clusters=list(coverDict.values()))


def delete_edge(G, edge):
    """
    Given a graph G and one of its edges in tuple form, checks if the deletion
    splits the graph.
    """
    G.delete_edges(edge)
    return check_for_split(G, edge)


def check_for_split(G, edge):
    """
    Given an edge in tuple form, check if it splits the
    graph into two disjoint clusters. If so, it returns
    True. Otherwise, False.
    """
    # Possibly keep a record of splits.
    if edge[0] == edge[1]:
        return False
    return not G.edge_disjoint_paths(source=edge[0], target=edge[1])


def split_vertex(G, v, splitInstructions):
    """
    Splits the vertex v into two new vertices, each with
    edges depending on s. Returns True if the split
    divided the graph, else False.
    """
    new_index = G.vcount()
    G.add_vertex()
    G.vs[new_index]["CONGA_orig"] = G.vs[v]["CONGA_orig"]

    # adding all relevant edges to new vertex, deleting from old one.
    for partner in splitInstructions:
        G.add_edge(new_index, partner)
        G.delete_edges((v, partner))

    # check if the two new vertices are disconnected.
    return check_for_split(G, (v, new_index))


def order_tuple(toOrder):
    """
    Given a tuple (a, b), returns (a, b) if a <= b,
    else (b, a).
    """
    if toOrder[0] <= toOrder[1]:
        return toOrder
    return (toOrder[1], toOrder[0])


def update_betweenness(G, path, pair, count, relevant):
    """
    Given a shortest path in G, along with a count of paths
    that length, to determine weight, updates the edge and
    pair betweenness dicts with the path's new information.
    """
    weight = 1.0 / count
    pos = 0
    while pos < len(path) - 2:
        if path[pos + 1] in relevant:
            pair[path[pos + 1]][order_tuple((path[pos], path[pos + 2]))] += weight
        pos += 1


def pair_betweenness(G, relevant):
    """
    Returns a dictionary of the pair betweenness of all vertices in relevant.
    The structure of the returned dictionary is dic[v][(u, w)] = c, where c
    is the number of shortest paths traverse through u, v, w.
    """
    pair_betweenness = {
        vertex: {uw: 0 for uw in itertools.combinations(G.neighbors(vertex), 2)}
        for vertex in relevant
    }

    for i in G.vs:
        pathCounts = co.Counter()
        # Only find the shortest paths that we haven't already seen
        shortest_paths_from_v = G.get_all_shortest_paths(
            i, to=G.vs[i.index + 1 :]
        )  # here too. need all shortest paths. too bad.
        for path in shortest_paths_from_v:  # reads twice. Can I get it down to once?
            pathCounts[path[-1]] += 1
        for path in shortest_paths_from_v:
            update_betweenness(
                G, path, pair_betweenness, pathCounts[path[-1]], set(relevant)
            )
    return pair_betweenness


def create_clique(G, v, pb):
    """
    Given a vertex and its pair betweennesses, returns a k-clique
    representing all of its neighbors, with edge weights determined by the pair
    betweenness scores. Algorithm discussed on page 5 of the CONGA paper.
    """
    neighbors = G.neighbors(v)

    # map each neighbor to its index in the adjacency matrix
    mapping = {neigh: i for i, neigh in enumerate(neighbors)}
    n = len(neighbors)

    # Can use ints instead: (dtype=int). Only works if we use matrix_min
    # instead of mat_min.
    clique = np.zeros((n, n))

    for uw, score in pb.items():

        clique[mapping[uw[0]], mapping[uw[1]]] = score
        clique[mapping[uw[1]], mapping[uw[0]]] = score

    # Ignore any self loops if they're there. If not, this line
    # does nothing and can be removed.
    np.fill_diagonal(clique, 0)
    return clique


def max_split_betweenness(G, dic):
    """
    Given a dictionary of vertices and their pair betweenness scores, uses the greedy
    algorithm discussed in the CONGA paper to find a (hopefully) near-optimal split.
    Returns a 3-tuple (vMax, vNum, vSpl) where vMax is the max split betweenness,
    vNum is the vertex with said split betweenness, and vSpl is a list of which
    vertices are on each side of the optimal split.
    """
    vMax = 0
    # for every vertex of interest, we want to figure out the maximum score achievable
    # by splitting the vertices in various ways, and return that optimal split
    for v in dic:
        clique = create_clique(G, v, dic[v])

        # initialize a list on how we will map the neighbors to the collapsing matrix
        vMap = [[ve] for ve in G.neighbors(v)]

        # we want to keep collapsing the matrix until we have a 2x2 matrix and its
        # score. Then we want to remove index j from our vMap list and concatenate
        # it with the vMap[i]. This begins building a way of keeping track of how
        # we are splitting the vertex and its neighbors
        while clique.size > 4:
            i, j, clique = reduce_matrix(clique)
            vMap[i] += vMap.pop(j)
        if clique[0, 1] >= vMax:
            vMax = clique[0, 1]
            vNum = v
            vSpl = vMap
    return vMax, vNum, vSpl


def matrix_min(mat):
    """
    Given a symmetric matrix, find an index of the minimum value
    in the upper triangle (not including the diagonal.)
    """
    # Currently, this function is unused, as its result is
    # the same as that of mat_min, and it is not always
    # faster. Left in for reference in case mat_min becomes
    # a bottleneck.

    # find the minimum from the upper triangular matrix
    # (not including the diagonal)
    upperTri = np.triu_indices(mat.shape[0], 1)
    minDex = mat[upperTri].argmin()

    # find the index in the big matrix. TODO: do so
    # with some algebra.
    triN = mat.shape[0] - 1
    row = 0
    while minDex >= triN:
        minDex -= triN
        triN -= 1
        row += 1
    col = mat.shape[0] - triN + minDex
    return row, col


def mat_min(M):
    """
    Given a matrix, find an index of the minimum value (not including the
    diagonal).
    """
    # take a matrix we pass in, and fill the diagonal with the matrix max. This is
    # so that we don't grab any values from the diag.
    np.fill_diagonal(M, float("inf"))

    # figure out the indices of the cell with the lowest value.
    i, j = np.unravel_index(M.argmin(), M.shape)
    np.fill_diagonal(M, 0)
    return i, j


def reduce_matrix(M):
    """
    Given a matrix M, collapses the row and column of the minimum value. This is just
    an adjacency matrix way of implementing the greedy "collapse" discussed in CONGA.
    Returns the new matrix and the collapsed indices.
    """
    i, j = mat_min(M)
    # i, j = matrix_min(M)
    # add the ith row to the jth row and overwrite the ith row with those values
    M[i, :] = M[j, :] + M[i, :]

    # delete the jth row
    M = np.delete(M, (j), axis=0)

    # similarly with the columns
    M[:, i] = M[:, j] + M[:, i]
    M = np.delete(M, (j), axis=1)
    np.fill_diagonal(M, 0)  # not sure necessary.
    return i, j, M


def pretty_print_cover(G, cover, label="CONGA_index"):
    """
    Takes a cover in vertex-id form and prints it nicely
    using label as each vertex's name.
    """
    pp = [G.vs[num] for num in [cluster for cluster in cover]]
    for count, comm in enumerate(pp):
        print("Community {0}:".format(count))
        for v in comm:
            print("\t", end=" ")
            if label == "CONGA_index":
                print(v.index)
            else:
                print(v[label])
        print()


def run_demo():
    """
    Finds the communities of the Zachary graph and gets the optimal one using
    Lazar's measure of modularity. Finally, pretty-prints the optimal cover.
    """
    G = ig.Graph().Famous("Zachary").as_undirected()
    result = conga(G, calculate_modularities="lazar")
    result.pretty_print_cover(result.optimal_count, label="CONGA_index")


def main():
    parser = argparse.ArgumentParser(
        description="""Run CONGA from the command line. Mostly meant as a demo -- only prints one cover."""
    )
    parser.add_argument(
        "-m",
        "--modularity_measure",
        choices=["lazar"],
        help="""Calculate the modularities using the specified
                            modularity measure. Currently only supports lazar.""",
    )
    parser.add_argument(
        "-n",
        "--num_clusters",
        type=int,
        help="""Specify the number of clusters to use.""",
    )
    parser.add_argument(
        "-d",
        "--demo",
        action="store_true",
        help="""Run a demo with the famous Zachary's Karate Club data set. Overrides all other options.""",
    )
    parser.add_argument(
        "-l",
        "--label",
        default="CONGA_index",
        nargs="?",
        const="label",
        help="""Choose which attribute of the graph to print.
                            When this option is present with no parameters, defaults to 'label'. When the option is not
                            present, defaults to the index.""",
    )
    parser.add_argument(
        "file", nargs="?", help="""The path to the file in igraph readable format."""
    )
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return
    if not args.file:
        print("conga.py: error: no file specified.\n")
        print(parser.parse_args("-h".split()))
        return

    # only works for undirected
    G = ig.read(args.file).as_undirected()
    result = conga(
        G,
        calculate_modularities=args.modularity_measure,
        optimal_count=args.num_clusters,
    )
    result.pretty_print_cover(result.optimal_count, label=args.label)


def Conga_(graph, number_communities=0):
    """

    :param graph:
    :param number_communities:
    :return:
    """

    result = conga(graph)

    if number_communities == 0:
        cover = result._covers[result.optimal_count]
        number_communities = result.optimal_count
    else:
        cover = result._covers[number_communities]

    list_communities = []
    for i in range(0, number_communities):
        list_communities.append(cover._clusters[i])

    return list_communities
