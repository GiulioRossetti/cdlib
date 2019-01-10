import infomap as map
import networkx as nx
import igraph as ig
from collections import defaultdict
from wurlitzer import pipes
from nclib.utils import convert_graph_formats


def infomap(g):
    """

    :param g:
    :return:
    """
    g = convert_graph_formats(g, nx.Graph)

    g1 = nx.convert_node_labels_to_integers(g, label_attribute="name")
    name_map = nx.get_node_attributes(g1, 'name')
    coms_to_node = defaultdict(list)

    with pipes():
        im = map.Infomap()
        network = im.network()
        for e in g1.edges():
           network.addLink(e[0],e[1])
        im.run()

        for node in im.iterTree():
           if node.isLeaf():
               nid = node.physicalId
               module = node.moduleIndex()
               nm = name_map[nid]
               coms_to_node[module].append(nm)

    coms_infomap = [tuple(c) for c in coms_to_node.values()]
    return coms_infomap


def walktrap(g):
    """

    :param g:
    :return:
    """
    g = convert_graph_formats(g, ig.Graph)
    coms = g.community_walktrap().as_clustering()
    communities = []

    for c in coms:
        communities.append(c)

    return communities
