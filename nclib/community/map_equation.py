import infomap as map
import networkx as nx
import igraph as ig
from collections import defaultdict
from wurlitzer import pipes
from nclib import NodeClustering
from nclib.utils import convert_graph_formats


def infomap(g):
    """
    Infomap is based on ideas of information theory.
    The algorithm uses the probability flow of random walks on a network as a proxy for information flows in the real system and it decomposes the network into modules by compressing a description of the probability flow.

    :param g: a networkx/igraph object
    :return: a list of communities

    :Example:

    >>> from nclib import community
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = community.infomap(G)

    :References:

    Rosvall M, Bergstrom CT (2008) **Maps of random walks on complex networks reveal community structure.** Proc Natl Acad SciUSA 105(4):1118–1123
    """
    g = convert_graph_formats(g, nx.Graph)

    g1 = nx.convert_node_labels_to_integers(g, label_attribute="name")
    name_map = nx.get_node_attributes(g1, 'name')
    coms_to_node = defaultdict(list)

    with pipes():
        im = map.Infomap()
        network = im.network()
        for e in g1.edges():
           network.addLink(e[0], e[1])
        im.run()

        for node in im.iterTree():
           if node.isLeaf():
               nid = node.physicalId
               module = node.moduleIndex()
               nm = name_map[nid]
               coms_to_node[module].append(nm)

    coms_infomap = [tuple(c) for c in coms_to_node.values()]
    return NodeClustering(coms_infomap, g, "Infomap")


def walktrap(g):
    """
    walktrap is an approach based on random walks.
    The general idea is that if you perform random walks on the graph, then the walks are more likely to stay within the same community because there are only a few edges that lead outside a given community. Walktrap runs short random walks and uses the results of these random walks to merge separate communities in a bottom-up manner.

    :param g: a networkx/igraph object
    :return: a list of communities

    :Example:

    >>> from nclib import community
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = community.walktrap(G)

    :References:

    Pons, Pascal, and Matthieu Latapy. **Computing communities in large networks using random walks.** J. Graph Algorithms Appl. 10.2 (2006): 191-218.
    """
    g = convert_graph_formats(g, ig.Graph)
    coms = g.community_walktrap().as_clustering()
    communities = []

    for c in coms:
        communities.append([g.vs[x]['name'] for x in c])

    return NodeClustering(communities, g, "Walktrap")
