"""
Baumes, Jeffrey, Mark Goldberg, and Malik Magdon-Ismail.
"Efficient identification of overlapping communities."
International Conference on Intelligence and Security Informatics.
Springer, Berlin, Heidelberg, 2005.

Reference internal
https://github.com/kritishrivastava/CommunityDetection-Project2GDM

"""

import networkx as nx


def __weight(community):
    """

    :param community: a subgraph/algorithms in the graph
    :return: weight of the algorithms (using the formula mentioned in the paper)
    """
    if nx.number_of_nodes(community) == 0:
        return 0
    else:
        return float(2 * nx.number_of_edges(community) / nx.number_of_nodes(community))


def __order_nodes(graph):
    """

    :param graph:
    :return: list of nodes sorted in the decreasing order of their page rank
    """
    dict_of_nodes = nx.pagerank(graph)
    ordered_nodes = dict_of_nodes.items()
    ordered_nodes = sorted(ordered_nodes, reverse=True, key=__get_key)
    return ordered_nodes


def __get_key(node):
    """

    :param node: list containing node name and its page rank
    :return: return rank of the node
    """
    return node[1]


def __list_aggregate(graph):
    """

    :param graph:
    :return: a group of clusters (initial guesses to be fed into the second algorithm)
    order the vertices using page rank
    """
    ordered_nodes = __order_nodes(graph)
    coms = []
    for i in ordered_nodes:
        added = False

        for c in coms:
            # Add the node and see if the weight of the cluster increases
            temp1 = graph.subgraph(c)
            cc = list(c)
            cc.append(i[0])
            temp2 = graph.subgraph(cc)

            # If weight increases, add the node to the cluster
            if __weight(temp2) > __weight(temp1):
                added = True
                c.append(i[0])
        if not added:
            coms.append([i[0]])
    return coms


def __is2(cluster, graph):
    """

    :param cluster:cluster to be improved
    :param graph:
    :return: improved cluster
    """
    c = graph.subgraph(cluster)
    initial_weight = __weight(c)
    increased = True
    while increased:
        list_of_nodes = cluster

        for vertex in c.nodes():
            # Get adjacent nodes
            adjacent_nodes = graph.neighbors(vertex)
            # Adding all adjacent nodes to the cluster
            list_of_nodes = list(set(list_of_nodes).union(set(adjacent_nodes)))

        for vertex in list_of_nodes:
            list_of_nodes = list(c.nodes())

            # If the vertex was a part of inital cluster
            if vertex in c.nodes():
                # Remove vertex from the cluster
                list_of_nodes.remove(vertex)

            # If the vertex is one of the recently added neighbours
            else:
                # Add vertex to the cluster
                list_of_nodes.append(vertex)
            c_dash = graph.subgraph(list_of_nodes)

            if __weight(c_dash) > __weight(c):
                c = c_dash.copy()

        new_weight = __weight(c)

        if new_weight == initial_weight:
            increased = False
        else:
            initial_weight = new_weight
    return c


def LAIS2(graph):
    """

    :param graph:
    :return:
    """

    # Generate initial "guesses" for clusters using Link Aggregate algorithm
    initial_clusters = __list_aggregate(graph)

    # Get final clusters using Improved Iterative Scan Algorithm
    final_clusters = []
    initial_clusters_without_duplicates = []
    for cluster in initial_clusters:
        cluster = sorted(cluster)
        if cluster not in initial_clusters_without_duplicates:
            initial_clusters_without_duplicates.append(cluster)
            updated_cluster = __is2(cluster, graph)
            final_clusters.append(updated_cluster.nodes())

    # Removing duplicate clusters and printing output to a file
    final_clusters_without_duplicates = []
    for cluster in final_clusters:
        cluster = sorted(cluster)
        if cluster not in final_clusters_without_duplicates:
            final_clusters_without_duplicates.append(cluster)

    return final_clusters_without_duplicates
