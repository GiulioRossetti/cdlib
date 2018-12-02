import infomap as map
import networkx as nx
from collections import defaultdict
from wurlitzer import pipes


def infomap(g):
    """

    :param g:
    :return:
    """

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
