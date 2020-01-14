from contextlib import contextmanager
try:
    import igraph as ig
except ModuleNotFoundError:
    ig = None

try:
    import graph_tool as gt
except ModuleNotFoundError:
    gt = None

import networkx as nx
import sys
import os
import numpy as np


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


def __from_nx_to_graph_tool(g, directed=False):
    """

    :param g:
    :param directed:
    :return:
    """

    if gt is None:
        raise ModuleNotFoundError(
            "Optional dependency not satisfied: install igraph to use the selected feature.")

    gt_g = gt.Graph(directed=directed)

    node_map = {v: i for i, v in enumerate(g.nodes())}

    gt_g.add_vertex(len(node_map))
    gt_g.add_edge_list([(node_map[u], node_map[v]) for u, v in g.edges()])

    return gt_g, {v: k for k, v in node_map.items()}


def __from_graph_tool_to_nx(graph, node_map=None, directed=False):
    if directed:
        tp = nx.DiGraph()
    else:
        tp = nx.Graph()

    tp.add_nodes_from([int(v) for v in graph.vertices()])
    tp.add_edges_from([(int(e.source()), int(e.target()))
                       for e in graph.edges()])
    if node_map:
        nx.relabel_nodes(tp, node_map, copy=False)

    return tp


def __from_nx_to_igraph(g, directed=False):
    """

    :param g:
    :param directed:
    :return:
    """

    if ig is None:
        raise ModuleNotFoundError(
            "Optional dependency not satisfied: install igraph to use the selected feature.")

    gi = ig.Graph(directed=directed)
    gi.add_vertices(list(g.nodes()))
    gi.add_edges(list(g.edges()))
    edgelist = nx.to_pandas_edgelist(g)
    for attr in edgelist.columns[2:]:
        gi.es[attr] = edgelist[attr]

    return gi


def __from_igraph_to_nx(ig, directed=False):
    """

    :param ig:
    :param directed:
    :return:
    """

    if ig is None:
        raise ModuleNotFoundError(
            "Optional dependency not satisfied: install igraph to use the selected feature.")

    if directed:
        tp = nx.DiGraph()
    else:
        tp = nx.Graph()

    for e in ig.es:
        tp.add_edge(e.source, e.target, **e.attributes())

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
    elif ig is not None and desired_format is ig.Graph:
        return __from_nx_to_igraph(graph, directed)
    else:
        raise TypeError(
            "The graph object should be either a networkx or an igraph one.")


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


def affiliations2nodesets(affiliations):
    """
    Transform community format to nodesets

    Representation expected in input: dictionary, key=node, value=list/set of snapshot_affiliations ID
    Representation in output: bidict, key=community ID , value=set of nodes

    :param affiliations:bidict, key=community ID , value=set of nodes
    :return: dict, key=community ID , value=set of nodes
    """

    if affiliations is None:
        return None

    asNodeSets = dict()

    if len(affiliations) == 0:
        return asNodeSets

    for n, coms in affiliations.items():
        if isinstance(coms, str) or isinstance(coms, int) or isinstance(coms, np.int32):
            coms = [coms]
        for c in coms:
            asNodeSets.setdefault(c, set())
            asNodeSets[c].add(n)

    return asNodeSets
