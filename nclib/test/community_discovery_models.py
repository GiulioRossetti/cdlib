import unittest
from nclib.models.node_centric import demon, angel, ego_networks, node_perception
from nclib.models.modularity import louvain, leiden, greedy_modularity, significance_communities, \
    surprise_communities, cpm, rb_pots, rber_pots
from nclib.models.map_equation import infomap
from nclib.models.label_propagation import label_propagation, async_fluid, SLPA, multicom
from nclib.models.structural import kclique, girvan_newman, EM, LFM, SCAN, HierarchicalLinkCommunity
import networkx as nx
import os


class NodeCentricTests(unittest.TestCase):

    def test_ego(self):
        g = nx.karate_club_graph()
        coms = ego_networks(g)
        self.assertEqual(len(coms), g.number_of_nodes())
        self.assertEqual(type(coms), list)

    def test_demon(self):
        g = nx.karate_club_graph()
        coms = demon(g, epsilon=0.25)
        self.assertEqual(type(coms), list)

    def test_nodeperception(self):
        g = nx.karate_club_graph()
        coms = node_perception(g, threshold=0.25, overlap_threshold=0.25)
        self.assertEqual(type(coms), list)

    def test_angel(self):
        g = nx.karate_club_graph()
        coms = angel(g, threshold=0.25)
        self.assertEqual(type(coms), list)

    def test_louvain(self):
        g = nx.karate_club_graph()
        coms = louvain(g)
        self.assertEqual(type(coms), list)

    def test_leiden(self):
        g = nx.karate_club_graph()
        coms = leiden(g)
        self.assertEqual(type(coms), list)

    def test_significance(self):
        g = nx.karate_club_graph()
        coms = significance_communities(g)
        self.assertEqual(type(coms), list)

    def test_surprise(self):
        g = nx.karate_club_graph()
        coms = surprise_communities(g)
        self.assertEqual(type(coms), list)

    def test_cpm(self):
        g = nx.karate_club_graph()
        coms = cpm(g)
        self.assertEqual(type(coms), list)

    def test_rbpots(self):
        g = nx.karate_club_graph()
        coms = rb_pots(g)
        self.assertEqual(type(coms), list)

    def test_rberpots(self):
        g = nx.karate_club_graph()
        coms = rber_pots(g)
        self.assertEqual(type(coms), list)

    def test_greedy_modularity(self):
        g = nx.karate_club_graph()
        coms = greedy_modularity(g)
        self.assertEqual(type(coms), list)

    def test_infomap(self):
        g = nx.karate_club_graph()
        coms = infomap(g)
        self.assertEqual(type(coms), list)
        os.remove(".tree")

    def test_lp(self):
        g = nx.karate_club_graph()
        coms = label_propagation(g)
        self.assertEqual(type(coms), list)

    def test_slpa(self):
        g = nx.karate_club_graph()
        coms = SLPA(g)
        self.assertEqual(type(coms), list)

    def test_fluid(self):
        g = nx.karate_club_graph()
        coms = async_fluid(g, 3)
        self.assertEqual(type(coms), list)

    def test_kclique(self):
        g = nx.karate_club_graph()
        coms = kclique(g, 3)
        self.assertEqual(type(coms), list)

    def test_gn(self):
        g = nx.karate_club_graph()
        coms = girvan_newman(g, 3)
        self.assertEqual(type(coms), list)

    def test_multicom(self):
        g = nx.karate_club_graph()
        coms = multicom(g, seed_node=0)
        self.assertEqual(type(coms), list)

    def test_em(self):
        g = nx.karate_club_graph()
        coms = EM(g, k=3)
        self.assertEqual(type(coms), list)

    def test_LFM(self):
        g = nx.karate_club_graph()
        coms = LFM(g, alpha=0.8)
        self.assertEqual(type(coms), list)

    def test_SCAN(self):
        g = nx.karate_club_graph()
        coms = SCAN(g, 0.7, 3)
        self.assertEqual(type(coms), list)

    def test_HLC(self):
        g = nx.karate_club_graph()
        coms = HierarchicalLinkCommunity(g)
        self.assertEqual(type(coms), list)


if __name__ == '__main__':
    unittest.main()