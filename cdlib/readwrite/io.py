from cdlib import NodeClustering

__all__ = ["write_community_csv", "read_community_csv"]


def write_community_csv(communities,  path, delimiter=","):
    """
    Save community structure to comma separated value (csv) file.

    :param communities: a NodeClustering object
    :param path: output filename
    :param delimiter: column delimiter

    :Example:

    >>> import networkx as nx
    >>> from cdlib import algorithms, readwrite
    >>> g = nx.karate_club_graph()
    >>> coms = algorithms.louvain(g)
    >>> readwrite.write_community_csv(coms, "communities.csv", ",")

    """
    with open(path, "w") as f:
        for cid, community in enumerate(communities.communities):
            res = delimiter.join(list(map(str, community)))
            f.write("%s\n" % res)


def read_community_csv(path, delimiter=",", nodetype=str):
    """
    Read community list from comma separated value (csv) file.

    :param path: input filename
    :param delimiter: column delimiter
    :param nodetype: specify the type of node labels, default str
    :return: NodeClustering object

    :Example:

    >>> import networkx as nx
    >>> from cdlib import algorithms, readwrite
    >>> g = nx.karate_club_graph()
    >>> coms = algorithms.louvain(g)
    >>> readwrite.write_community_csv(coms, "communities.csv", ",")
    >>> coms = readwrite.read_community_csv(coms, "communities.csv", ",", str)

    """
    communities = []
    with open(path) as f:
        for row in f:
            community = list(map(nodetype, row.rstrip().split(delimiter)))
            communities.append(tuple(community))

    return NodeClustering(communities, None, "")
