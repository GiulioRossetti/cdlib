from cdlib import NodeClustering, FuzzyNodeClustering, EdgeClustering
import json
import gzip

__all__ = [
    "write_community_csv",
    "read_community_csv",
    "write_community_json",
    "read_community_json",
    "read_community_from_json_string",
]


def write_community_csv(
    communities: object, path: str, delimiter: str = ",", compress: bool = False
):
    """
    Save community structure to comma separated value (csv) file.

    :param communities: a NodeClustering object
    :param path: output filename
    :param delimiter: column delimiter
    :param compress: wheter to copress the csv, default False

    :Example:

    >>> import networkx as nx
    >>> from cdlib import algorithms, readwrite
    >>> g = nx.karate_club_graph()
    >>> coms = algorithms.louvain(g)
    >>> readwrite.write_community_csv(coms, "communities.csv", ",")

    """
    if compress:
        op = gzip.open
    else:
        op = open

    with op(path, "wt") as f:
        for cid, community in enumerate(communities.communities):
            res = delimiter.join(list(map(str, community)))
            f.write("%s\n" % res)


def read_community_csv(
    path: str, delimiter: str = ",", nodetype: type = str, compress: bool = False
) -> object:
    """
    Read community list from comma separated value (csv) file.

    :param path: input filename
    :param delimiter: column delimiter
    :param nodetype: specify the type of node labels, default str
    :param compress: wheter the file is compressed or not, default False
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

    if compress:
        op = gzip.open
    else:
        op = open

    with op(path, "rt") as f:
        for row in f:
            community = list(map(nodetype, row.rstrip().split(delimiter)))
            communities.append(list(community))

    return NodeClustering(communities, None, "")


def write_community_json(communities: object, path: str, compress: bool = False):
    """
    Generate a JSON representation of the clustering object

    :param communities: a cdlib clustering object
    :param path: output filename
    :param compress: wheter to copress the JSON, default False
    :return: a JSON formatted string representing the object

    :Example:

    >>> import networkx as nx
    >>> from cdlib import algorithms, readwrite
    >>> g = nx.karate_club_graph()
    >>> coms = algorithms.louvain(g)
    >>> readwrite.write_community_json(coms, "communities.json")
    """

    partition = {
        "communities": communities.communities,
        "algorithm": communities.method_name,
        "params": communities.method_parameters,
        "overlap": communities.overlap,
        "coverage": communities.node_coverage,
    }

    try:
        partition["allocation_matrix"] = communities.allocation_matrix
    except AttributeError:
        pass

    js_dmp = json.dumps(partition)

    if compress:
        op = gzip.open
    else:
        op = open

    with op(path, "wt") as f:
        f.write(js_dmp)


def read_community_json(path: str, compress: bool = False) -> object:
    """
    Read community list from JSON file.

    :param path: input filename
    :param compress: wheter the file is in a copress format, default False
    :return: a Clustering object

    :Example:

    >>> import networkx as nx
    >>> from cdlib import algorithms, readwrite
    >>> g = nx.karate_club_graph()
    >>> coms = algorithms.louvain(g)
    >>> readwrite.write_community_json(coms, "communities.json")
    >>> readwrite.read_community_json(coms, "communities.json")
    """

    if compress:
        op = gzip.open
    else:
        op = open

    with op(path, "rt") as f:
        coms = json.load(f)

    nc = NodeClustering(
        [list(c) for c in coms["communities"]],
        None,
        coms["algorithm"],
        coms["params"],
        coms["overlap"],
    )
    nc.node_coverage = coms["coverage"]

    if "allocation_matrix" in coms:
        nc.__class__ = FuzzyNodeClustering
        nc.allocation_matrix = coms["allocation_matrix"]

    if type(nc.communities[0][0]) is list:
        cms = []
        for c in nc.communities:
            cm = []
            for e in c:
                cm.append(tuple(e))
            cms.append(list(cm))
        nc.communities = cms
        nc.__class__ = EdgeClustering

    return nc


def read_community_from_json_string(json_repr: str) -> object:
    """
    Read community list from JSON file.

    :param json_repr: json community representation
    :return: a Clustering object

    :Example:

    >>> import networkx as nx
    >>> from cdlib import algorithms, readwrite
    >>> g = nx.karate_club_graph()
    >>> coms = algorithms.louvain(g)
    >>> readwrite.write_community_json(coms, "communities.json")
    >>> with open("community.json") as f:
    >>>     cr = f.read()
    >>>     readwrite.write_community_from_json_string(cr)
    """

    coms = json.loads(json_repr)

    nc = NodeClustering(
        [list(c) for c in coms["communities"]],
        None,
        coms["algorithm"],
        coms["params"],
        coms["overlap"],
    )
    nc.node_coverage = coms["coverage"]

    if "allocation_matrix" in coms:
        nc.__class__ = FuzzyNodeClustering
        nc.allocation_matrix = coms["allocation_matrix"]

    if type(nc.communities[0][0]) is list:
        cms = []
        for c in nc.communities:
            cm = []
            for e in c:
                cm.append(tuple(e))
            cms.append(tuple(cm))
        nc.communities = cms
        nc.__class__ = EdgeClustering

    return nc
