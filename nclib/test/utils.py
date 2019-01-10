import unittest
import networkx as nx
import igraph as ig
from nclib import utils
from nclib import community


class UtilsTests(unittest.TestCase):

    def test_nx_to_ig(self):
        g = nx.karate_club_graph()
        ign = utils.from_nx_to_igraph(g)
        self.assertEqual(g.number_of_nodes(), ign.vcount())
        self.assertEqual(g.number_of_edges(), ign.ecount())

    def test_ig_to_nx(self):
        g = nx.karate_club_graph()
        ign = utils.from_nx_to_igraph(g)
        g2 = utils.from_igraph_to_nx(ign)

        self.assertEqual(g.number_of_nodes(), g2.number_of_nodes())
        self.assertEqual(g.number_of_edges(), g2.number_of_edges())

    def test_convert(self):
        g = nx.karate_club_graph()
        ign = utils.convert_graph_formats(g, ig.Graph)
        self.assertEqual(g.number_of_nodes(), ign.vcount())
        self.assertEqual(g.number_of_edges(), ign.ecount())

        g2 = utils.convert_graph_formats(ign, nx.Graph)
        self.assertEqual(g.number_of_nodes(), g2.number_of_nodes())
        self.assertEqual(g.number_of_edges(), g2.number_of_edges())

        g3 = utils.convert_graph_formats(g, nx.Graph)
        self.assertEqual(isinstance(g, nx.Graph), isinstance(g3, nx.Graph))

    def test_nx_integer_mapping(self):
        g = nx.karate_club_graph()
        nodes = list(g.nodes())
        g, node_map = utils.nx_node_integer_mapping(g)
        self.assertListEqual(sorted(nodes), sorted(list(node_map.values())))

    def test_remap_node_community(self):
        g = nx.karate_club_graph()
        g, node_map = utils.nx_node_integer_mapping(g)
        nodes = list(g.nodes())
        coms = community.louvain(g)
        coms_remap = utils.remap_node_communities(coms, node_map)

        flat_list = [item for sublist in coms_remap for item in sublist]
        self.assertListEqual(sorted(nodes), sorted(flat_list))