import unittest
import networkx as nx

try:
    import igraph as ig
except ModuleNotFoundError:
    ig = None
from cdlib import utils
from cdlib import algorithms


def get_string_graph():
    g = nx.karate_club_graph()
    node_map = {}
    for n in g.nodes():
        node_map[n] = "$%s$" % n
    nx.relabel_nodes(g, node_map, False)
    return g


# networkx nodes just need to be Hashable:
# this class is used in a test below that verifies
# that such graphs can be converted into igraph
class Node:
    def __init__(self, id, type):
        self.id = id
        self.type = type

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


class UtilsTests(unittest.TestCase):
    def test_convert(self):
        g = nx.karate_club_graph()
        if ig is not None:
            ign = utils.convert_graph_formats(g, ig.Graph)
            self.assertEqual(g.number_of_nodes(), ign.vcount())
            self.assertEqual(g.number_of_edges(), ign.ecount())

            g2 = utils.convert_graph_formats(ign, nx.Graph)
            self.assertEqual(g.number_of_nodes(), g2.number_of_nodes())
            self.assertEqual(g.number_of_edges(), g2.number_of_edges())

            g3 = utils.convert_graph_formats(g, nx.Graph)
            self.assertEqual(isinstance(g, nx.Graph), isinstance(g3, nx.Graph))

            g3 = utils.convert_graph_formats(ign, nx.Graph, directed=True)
            self.assertIsInstance(g3, nx.DiGraph)

    def test_convert_bipartite(self):
        john = Node('John Travolta', 'actor')
        nick = Node('Nick Cage', 'actor')
        face_off = Node('Face Off', 'movie')

        g = nx.Graph()
        g.add_node(john)
        g.add_node(nick)
        g.add_edge(john, face_off, label='ACTED_IN')
        g.add_edge(nick, face_off, label='ACTED_IN')

        if ig is not None:
            ign = utils.convert_graph_formats(g, ig.Graph)
            self.assertEqual(g.number_of_nodes(), ign.vcount())
            self.assertEqual(g.number_of_edges(), ign.ecount())

            g2 = utils.convert_graph_formats(ign, nx.Graph)
            self.assertEqual(g.number_of_nodes(), g2.number_of_nodes())
            self.assertEqual(g.number_of_edges(), g2.number_of_edges())

            g3 = utils.convert_graph_formats(g, nx.Graph)
            self.assertEqual(isinstance(g, nx.Graph), isinstance(g3, nx.Graph))

            g3 = utils.convert_graph_formats(ign, nx.Graph, directed=True)
            self.assertIsInstance(g3, nx.DiGraph)

    def test_nx_integer_mapping(self):
        g = nx.karate_club_graph()
        g, node_map = utils.nx_node_integer_mapping(g)
        self.assertIsNone(node_map)

        g = get_string_graph()
        nodes = list(g.nodes())
        g, node_map = utils.nx_node_integer_mapping(g)
        self.assertListEqual(sorted(nodes), sorted(list(node_map.values())))

    def test_remap_node_community(self):
        g = get_string_graph()
        nodes = list(g.nodes())
        g, node_map = utils.nx_node_integer_mapping(g)

        coms = algorithms.louvain(g)
        coms_remap = utils.remap_node_communities(coms.communities, node_map)

        flat_list = [item for sublist in coms_remap for item in sublist]
        self.assertListEqual(sorted(nodes), sorted(flat_list))
