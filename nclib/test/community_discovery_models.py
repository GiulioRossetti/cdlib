import unittest
from nclib import community
import networkx as nx
import os


class NodeCentricTests(unittest.TestCase):

    def test_ego(self):
        g = nx.karate_club_graph()
        coms = community.ego_networks(g)
        self.assertEqual(len(coms), g.number_of_nodes())
        self.assertEqual(type(coms), list)

    def test_demon(self):
        g = nx.karate_club_graph()
        coms = community.demon(g, epsilon=0.25)
        self.assertEqual(type(coms), list)

    def test_nodeperception(self):
        g = nx.karate_club_graph()
        coms = community.node_perception(g, threshold=0.25, overlap_threshold=0.25)
        self.assertEqual(type(coms), list)

    def test_angel(self):
        g = nx.karate_club_graph()
        coms = community.angel(g, threshold=0.25)
        self.assertEqual(type(coms), list)

    def test_louvain(self):
        g = nx.karate_club_graph()
        coms = community.louvain(g)
        self.assertEqual(type(coms), list)

    def test_leiden(self):
        g = nx.karate_club_graph()
        coms = community.leiden(g)
        self.assertEqual(type(coms), list)

    def test_significance(self):
        g = nx.karate_club_graph()
        coms = community.significance_communities(g)
        self.assertEqual(type(coms), list)

    def test_surprise(self):
        g = nx.karate_club_graph()
        coms = community.surprise_communities(g)
        self.assertEqual(type(coms), list)

    def test_cpm(self):
        g = nx.karate_club_graph()
        coms = community.cpm(g)
        self.assertEqual(type(coms), list)

    def test_rbpots(self):
        g = nx.karate_club_graph()
        coms = community.rb_pots(g)
        self.assertEqual(type(coms), list)

    def test_rberpots(self):
        g = nx.karate_club_graph()
        coms = community.rber_pots(g)
        self.assertEqual(type(coms), list)

    def test_greedy_modularity(self):
        g = nx.karate_club_graph()
        coms = community.greedy_modularity(g)
        self.assertEqual(type(coms), list)

    def test_infomap(self):
        g = nx.karate_club_graph()
        coms = community.infomap(g)
        self.assertEqual(type(coms), list)
        os.remove(".tree")

    def test_lp(self):
        g = nx.karate_club_graph()
        coms = community.label_propagation(g)
        self.assertEqual(type(coms), list)

    def test_slpa(self):
        g = nx.karate_club_graph()
        coms = community.SLPA(g)
        self.assertEqual(type(coms), list)

    def test_fluid(self):
        g = nx.karate_club_graph()
        coms = community.async_fluid(g, 3)
        self.assertEqual(type(coms), list)

    def test_kclique(self):
        g = nx.karate_club_graph()
        coms = community.kclique(g, 3)
        self.assertEqual(type(coms), list)

    def test_gn(self):
        g = nx.karate_club_graph()
        coms = community.girvan_newman(g, 3)
        self.assertEqual(type(coms), list)

    def test_multicom(self):
        g = nx.karate_club_graph()
        coms = community.multicom(g, seed_node=0)
        self.assertEqual(type(coms), list)

    def test_em(self):
        g = nx.karate_club_graph()
        coms = community.EM(g, k=3)
        self.assertEqual(type(coms), list)

    def test_LFM(self):
        g = nx.karate_club_graph()
        coms = community.LFM(g, alpha=0.8)
        self.assertEqual(type(coms), list)

    def test_SCAN(self):
        g = nx.karate_club_graph()
        coms = community.SCAN(g, 0.7, 3)
        self.assertEqual(type(coms), list)

    def test_HLC(self):
        g = nx.karate_club_graph()
        coms = community.HierarchicalLinkCommunity(g)
        self.assertEqual(type(coms), list)

    def test_DER(self):
        g = nx.karate_club_graph()
        coms = community.der(g)
        self.assertEqual(type(coms), list)


if __name__ == '__main__':
    unittest.main()