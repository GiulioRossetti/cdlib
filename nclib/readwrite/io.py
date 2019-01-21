from nclib import NodeClustering

def write_community_csv(communities,  path, delimiter=",", community_id=False):
    """
    Save community structure to comma separated value (csv) file.

    :param communities: a NodeClustering object
    :param path: output filename
    :param delimiter: column delimiter
    :param community_id: boolean flag. If True an incremental id is assigned to each community

    :Example:

    >>> import networkx as nx
    >>> from nclib import community, readwrite
    >>> g = nx.karate_club_graph()
    >>> coms = community.louvain(g)
    >>> readwrite.write_community_csv(coms, "communities.csv", ",", False)

    """
    with open(path, "w") as f:
        for cid, community in enumerate(communities.communities):
            res = delimiter.join(list(map(str, community)))
            if community_id:
                res = "%s\t%s\n" % (cid, res)
            f.write("%s\n" % res)


def read_community_csv(path, delimiter=",", community_id=False, nodetype=str):
    """
    Read community list from comma separated value (csv) file.

    :param path: input filename
    :param delimiter: column delimiter
    :param community_id: boolean flag. If True the first value for each column is considered a community identifier
    :param nodetype: specify the type of node labels, default str
    :return: NodeClustering object

    :Example:

    >>> import networkx as nx
    >>> from nclib import community, readwrite
    >>> g = nx.karate_club_graph()
    >>> coms = community.louvain(g)
    >>> readwrite.write_community_csv(coms, "communities.csv", ",", False)
    >>> coms = readwrite.read_community_csv(coms, "communities.csv", ",", False, str)

    """
    communities = []
    with open(path) as f:
        for row in f:
            if community_id:
                cid, row = row.split("\t")
            community = list(map(nodetype, row.rstrip().split(delimiter)))
            communities.append(tuple(community))

    return NodeClustering(communities, None, "")
