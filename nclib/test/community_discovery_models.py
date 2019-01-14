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

    def test_node_perception(self):
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
        if os.path.exists(".tree"):
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

    def test_osse(self):
        g = nx.karate_club_graph()
        seeds = [0, 2, 5]
        communities = community.overlapping_seed_set_expansion(g, seeds)
        self.assertEqual(type(communities), list)

    def test_markov_clustering(self):
        g = nx.watts_strogatz_graph(200, 10, 0.1)
        communities = community.markov_clustering(g)
        self.assertEqual(type(communities), list)

    def test_bigClam(self):
        g = nx.karate_club_graph()
        coms = community.bigClam(g)
        self.assertEqual(type(coms), list)

    def test_lemon(self):
        g = nx.karate_club_graph()
        seeds = [0, 2, 5]
        com = community.Lemon(g, seeds, min_com_size=2, max_com_size=5)
        self.assertEqual(type(com), list)

    def test_lais2(self):
        g = nx.karate_club_graph()
        com = community.lais2(g)
        self.assertEqual(type(com), list)

    def test_gdmp2(self):
        g = nx.karate_club_graph()
        com = community.gdmp2(g, min_threshold=.75)
        self.assertEqual(type(com), list)

    def test_spinglass(self):
        g = nx.karate_club_graph()
        com = community.spinglass(g)
        self.assertEqual(type(com), list)

    def test_walktrap(self):
        g = nx.karate_club_graph()
        com = community.walktrap(g)
        self.assertEqual(type(com), list)

    def test_eigenvector(self):
        g = nx.karate_club_graph()
        com = community.eigenvector(g)
        self.assertEqual(type(com), list)

    def test_Congo(self):
        g = nx.karate_club_graph()
        coms = community.Congo(g)
        self.assertEqual(type(coms), list)

    def test_Conga(self):
        g = nx.karate_club_graph()
        coms = community.Conga(g)
        self.assertEqual(type(coms), list)

    def test_Fluid(self):
        g = nx.karate_club_graph()
        coms = community.Conga(g)
        self.assertEqual(type(coms), list)


if __name__ == '__main__':
    unittest.main()
