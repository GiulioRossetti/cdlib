from contextlib import contextmanager
import igraph as ig
import networkx as nx
import sys
import os


@contextmanager
def suppress_stdout():
    """
    Suppress the standard out messages.
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def __from_nx_to_igraph(g, directed=False):
    """

    :param g:
    :param directed:
    :return:
    """
    gi = ig.Graph(directed=directed)
    gi.add_vertices(list(g.nodes()))
    gi.add_edges(list(g.edges()))
    return gi


def __from_igraph_to_nx(ig, directed=False):
    """

    :param ig:
    :param directed:
    :return:
    """
    if directed:
        tp = nx.DiGraph()
    else:
        tp = nx.Graph()

    for names in [ig.vs['name']]:
        for x in ig.get_edgelist():
            tp.add_edge(names[x[0]], names[x[1]])

    print(type(tp))
    return tp


def convert_graph_formats(graph, desired_format, directed=False):
    """Converts from/to networkx/igraph


    :param graph: original graph object
    :param desired_format: desired final type. Either nx.Graph or ig.Graph
    :param directed: boolean, default **False**
    :return: the converted graph
    :raises TypeError: if input graph is neither an instance of nx.Graph nor ig.Graph
    """
    if isinstance(graph, desired_format):
        return graph
    elif desired_format is nx.Graph:
        return __from_igraph_to_nx(graph, directed)
    elif desired_format is ig.Graph:
        return __from_nx_to_igraph(graph, directed)
    else:
        raise TypeError("The graph object should be either a networkx or an igraph one.")


def nx_node_integer_mapping(graph):
    """Maps node labels from strings to integers.

    :param graph: networkx graph
    :return: if the node labels are string: networkx graph, dictionary <numeric_id, original_node_label>, false otherwise
    """

    convert = False
    for nid in graph.nodes():
        if isinstance(nid, str):
            convert = True
            break

    if convert:
        node_map = {}
        label_map = {}
        if isinstance(graph, nx.Graph):
            for nid, name in enumerate(graph.nodes()):
                node_map[nid] = name
                label_map[name] = nid

            nx.relabel_nodes(graph, label_map, copy=False)
            return graph, node_map
        else:
            raise ValueError("graph must be a networkx Graph object")

    return graph, None


def remap_node_communities(communities, node_map):
    """Apply a map to the obtained communities to retreive the original node labels

    :param communities: NodeClustering object
    :param node_map: dictionary <numeric_id, node_label>
    :return: remapped communities
    """

    cms = []
    for community in communities:
        community = [node_map[n] for n in community]
        cms.append(community)
    communities = cms
    return communities
