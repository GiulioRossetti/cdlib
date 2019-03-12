import unittest
from cdlib import algorithms
import networkx as nx
import os


def get_string_graph():
    g = nx.karate_club_graph()
    node_map = {}
    for n in g.nodes():
        node_map[n] = "$%s$" % n
    nx.relabel_nodes(g, node_map, False)
    return g


class CommunityDiscoveryTests(unittest.TestCase):
    
    def test_ego(self):
        g = get_string_graph()
        coms = algorithms.ego_networks(g)
        self.assertEqual(len(coms.communities), g.number_of_nodes())
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_demon(self):
        g = get_string_graph()
        coms = algorithms.demon(g, epsilon=0.25)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_node_perception(self):
        g = get_string_graph()

        coms = algorithms.node_perception(g, threshold=0.25, overlap_threshold=0.25)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)

        g = nx.karate_club_graph()

        coms = algorithms.node_perception(g, threshold=0.25, overlap_threshold=0.25)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_angel(self):
        g = get_string_graph()
        coms = algorithms.angel(g, threshold=0.25)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_louvain(self):
        g = get_string_graph()
        coms = algorithms.louvain(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_leiden(self):
        g = get_string_graph()
        coms = algorithms.leiden(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_significance(self):
        g = get_string_graph()
        coms = algorithms.significance_communities(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_surprise(self):
        g = get_string_graph()
        coms = algorithms.surprise_communities(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_cpm(self):
        g = get_string_graph()
        coms = algorithms.cpm(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_rbpots(self):
        g = get_string_graph()
        coms = algorithms.rb_pots(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_rberpots(self):
        g = get_string_graph()
        coms = algorithms.rber_pots(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_greedy_modularity(self):
        g = get_string_graph()
        coms = algorithms.greedy_modularity(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_infomap(self):
        g = get_string_graph()
        coms = algorithms.infomap(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)
        if os.path.exists(".tree"):
            os.remove(".tree")

    def test_lp(self):
        g = get_string_graph()
        coms = algorithms.label_propagation(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_slpa(self):
        g = get_string_graph()
        coms = algorithms.slpa(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_fluid(self):
        g = get_string_graph()
        coms = algorithms.async_fluid(g, 3)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_kclique(self):
        g = get_string_graph()
        coms = algorithms.kclique(g, 3)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_gn(self):
        g = get_string_graph()
        coms = algorithms.girvan_newman(g, 3)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_multicom(self):
        g = get_string_graph()
        coms = algorithms.multicom(g, seed_node=0)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_em(self):
        g = get_string_graph()
        coms = algorithms.em(g, k=3)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_LFM(self):
        g = get_string_graph()
        coms = algorithms.lfm(g, alpha=0.8)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_SCAN(self):
        g = get_string_graph()
        coms = algorithms.scan(g, 0.7, 3)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_HLC(self):
        g = get_string_graph()
        coms = algorithms.hierarchical_link_community(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), tuple)

    def test_DER(self):
        g = get_string_graph()
        coms = algorithms.der(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_osse(self):
        g = get_string_graph()
        seeds = ["$0$", "$2$", "$5$"]
        communities = algorithms.overlapping_seed_set_expansion(g, seeds)
        self.assertEqual(type(communities.communities), list)
        if len(communities.communities) > 0:
            self.assertEqual(type(communities.communities[0]), list)
            self.assertEqual(type(communities.communities[0][0]), str)

    def test_markov_clustering(self):
        g = get_string_graph()

        communities = algorithms.markov_clustering(g)
        self.assertEqual(type(communities.communities), list)
        if len(communities.communities) > 0:
            self.assertEqual(type(communities.communities[0]), list)
            if len(communities.communities[0]) > 0:
                self.assertEqual(type(communities.communities[0][0]), tuple)

        g = nx.karate_club_graph()

        communities = algorithms.markov_clustering(g)
        self.assertEqual(type(communities.communities), list)
        if len(communities.communities) > 0:
            self.assertEqual(type(communities.communities[0]), list)
            if len(communities.communities[0]) > 0:
                self.assertEqual(type(communities.communities[0][0]), tuple)

    def test_bigClam(self):
        g = get_string_graph()
        coms = algorithms.big_clam(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            if len(coms.communities[0]) > 0:
                self.assertEqual(type(coms.communities[0][0]), str)

    def test_lemon(self):
        g = get_string_graph()
        seeds = ["$0$", "$2$", "$3$"]
        com = algorithms.lemon(g, seeds, min_com_size=10, max_com_size=50)
        self.assertEqual(type(com.communities), list)
        if len(com.communities) > 0:
            self.assertEqual(type(com.communities[0]), list)
            self.assertEqual(type(com.communities[0][0]), str)

    def test_lais2(self):
        g = get_string_graph()
        com = algorithms.lais2(g)
        self.assertEqual(type(com.communities), list)
        if len(com.communities) > 0:
            self.assertEqual(type(com.communities[0]), list)
            self.assertEqual(type(com.communities[0][0]), str)

    def test_gdmp2(self):
        g = get_string_graph()
        com = algorithms.gdmp2(g, min_threshold=.75)
        self.assertEqual(type(com.communities), list)
        if len(com.communities) > 0:
            self.assertEqual(type(com.communities[0]), list)
            self.assertEqual(type(com.communities[0][0]), str)

    def test_spinglass(self):
        g = get_string_graph()
        com = algorithms.spinglass(g)
        self.assertEqual(type(com.communities), list)
        if len(com.communities) > 0:
            self.assertEqual(type(com.communities[0]), list)
            self.assertEqual(type(com.communities[0][0]), str)

    def test_walktrap(self):
        g = get_string_graph()
        com = algorithms.walktrap(g)
        self.assertEqual(type(com.communities), list)
        if len(com.communities) > 0:
            self.assertEqual(type(com.communities[0]), list)
            self.assertEqual(type(com.communities[0][0]), str)

    def test_eigenvector(self):
        g = get_string_graph()
        com = algorithms.eigenvector(g)
        self.assertEqual(type(com.communities), list)
        if len(com.communities) > 0:
            self.assertEqual(type(com.communities[0]), list)
            self.assertEqual(type(com.communities[0][0]), str)

    def test_Congo(self):
        g = get_string_graph()
        coms = algorithms.congo(g, number_communities=3, height=2)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_Conga(self):
        g = get_string_graph()
        coms = algorithms.conga(g, number_communities=3)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_agdl(self):
        g = get_string_graph()
        coms = algorithms.agdl(g, 3, 2, 2, 0.5)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_frc_fgsn(self):
        g = get_string_graph()
        coms = algorithms.frc_fgsn(g, 1, 0.5, 3)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), tuple)
            self.assertIsInstance(coms.allocation_matrix, dict)
            self.assertEqual(len(coms.allocation_matrix), g.number_of_nodes())
