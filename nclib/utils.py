from contextlib import contextmanager
import igraph as ig
import networkx as nx
import sys
import os


@contextmanager
def suppress_stdout():
    """

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

    g = nx.from_edgelist([(names[x[0]], names[x[1]])
                          for names in [ig.vs['name']] for x in ig.get_edgelist()], tp)
    return g


def convert_graph_formats(graph, desired_format, directed=False):
    if isinstance(graph, desired_format):
        return graph
    elif desired_format is nx.Graph:
        return __from_igraph_to_nx(graph, directed)
    elif desired_format is ig.Graph:
        return __from_nx_to_igraph(graph, directed)
    else:
        raise TypeError("The graph object should be either a networkx or an igraph one.")


def nx_node_integer_mapping(graph):
    """

    :param graph:
    :return:
    """

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


def remap_node_communities(communities, node_map):
    """

    :param communities:
    :param node_map:
    :return:
    """

    cms = []
    for community in communities:
        community = [node_map[n] for n in community]
        cms.append(community)
    communities = cms
    return communities
