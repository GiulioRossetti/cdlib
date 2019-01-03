import unittest
import networkx as nx
import igraph as ig
from nclib.utils import from_nx_to_igraph, from_igraph_to_nx, convert_graph_formats


class UtilsTests(unittest.TestCase):

    def test_nx_to_ig(self):
        g = nx.karate_club_graph()
        ign = from_nx_to_igraph(g)
        self.assertEqual(g.number_of_nodes(), ign.vcount())
        self.assertEqual(g.number_of_edges(), ign.ecount())

    def test_ig_to_nx(self):
        g = nx.karate_club_graph()
        ign = from_nx_to_igraph(g)
        g2 = from_igraph_to_nx(ign)

        self.assertEqual(g.number_of_nodes(), g2.number_of_nodes())
        self.assertEqual(g.number_of_edges(), g2.number_of_edges())

    def test_convert(self):
        g = nx.karate_club_graph()
        ign = convert_graph_formats(g, ig.Graph)
        self.assertEqual(g.number_of_nodes(), ign.vcount())
        self.assertEqual(g.number_of_edges(), ign.ecount())

        g2 = convert_graph_formats(ign, nx.Graph)
        self.assertEqual(g.number_of_nodes(), g2.number_of_nodes())
        self.assertEqual(g.number_of_edges(), g2.number_of_edges())

        g3 = convert_graph_formats(g, nx.Graph)
        self.assertEqual(isinstance(g, nx.Graph), isinstance(g3, nx.Graph))