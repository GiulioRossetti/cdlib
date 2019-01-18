import unittest
from nclib import community
import networkx as nx
import os


def get_string_graph():
    g = nx.karate_club_graph()
    node_map = {}
    for n in g.nodes():
        node_map[n] = "$%s$" % n
    nx.relabel_nodes(g, node_map, False)
    return g


class NodeCentricTests(unittest.TestCase):
    
    def test_ego(self):
        g = get_string_graph()
        coms = community.ego_networks(g)
        self.assertEqual(len(coms.communities), g.number_of_nodes())
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_demon(self):
        g = get_string_graph()
        coms = community.demon(g, epsilon=0.25)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_node_perception(self):
        g = get_string_graph()

        coms = community.node_perception(g, threshold=0.25, overlap_threshold=0.25)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_angel(self):
        g = get_string_graph()
        coms = community.angel(g, threshold=0.25)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_louvain(self):
        g = get_string_graph()
        coms = community.louvain(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_leiden(self):
        g = get_string_graph()
        coms = community.leiden(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_significance(self):
        g = get_string_graph()
        coms = community.significance_communities(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_surprise(self):
        g = get_string_graph()
        coms = community.surprise_communities(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_cpm(self):
        g = get_string_graph()
        coms = community.cpm(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_rbpots(self):
        g = get_string_graph()
        coms = community.rb_pots(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_rberpots(self):
        g = get_string_graph()
        coms = community.rber_pots(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_greedy_modularity(self):
        g = get_string_graph()
        coms = community.greedy_modularity(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_infomap(self):
        g = get_string_graph()
        coms = community.infomap(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0][0]), str)
        if os.path.exists(".tree"):
            os.remove(".tree")

    def test_lp(self):
        g = get_string_graph()
        coms = community.label_propagation(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_slpa(self):
        g = get_string_graph()
        coms = community.slpa(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_fluid(self):
        g = get_string_graph()
        coms = community.async_fluid(g, 3)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_kclique(self):
        g = get_string_graph()
        coms = community.kclique(g, 3)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_gn(self):
        g = get_string_graph()
        coms = community.girvan_newman(g, 3)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_multicom(self):
        g = get_string_graph()
        coms = community.multicom(g, seed_node=0)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_em(self):
        g = get_string_graph()
        coms = community.em(g, k=3)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_LFM(self):
        g = get_string_graph()
        coms = community.lfm(g, alpha=0.8)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_SCAN(self):
        g = get_string_graph()
        coms = community.scan(g, 0.7, 3)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_HLC(self):
        g = get_string_graph()
        coms = community.hierarchical_link_community(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0][0]), tuple)

    def test_DER(self):
        g = get_string_graph()
        coms = community.der(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_osse(self):
        g = get_string_graph()
        seeds = ["$0$", "$2$", "$5$"]
        communities = community.overlapping_seed_set_expansion(g, seeds)
        self.assertEqual(type(communities.communities), list)
        if len(communities.communities) > 0:
            self.assertEqual(type(communities.communities[0][0]), str)

    def test_markov_clustering(self):
        g = nx.watts_strogatz_graph(200, 10, 0.1)
        communities = community.markov_clustering(g)
        self.assertEqual(type(communities.communities), list)
        if len(communities.communities) > 0:
            self.assertEqual(type(communities.communities[0][0]), str)

    def test_bigClam(self):
        g = get_string_graph()
        coms = community.big_clam(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_lemon(self):
        g = get_string_graph()
        seeds = ["$0$", "$2$", "$3$"]
        com = community.lemon(g, seeds, min_com_size=2, max_com_size=5)
        print(com.communities)
        self.assertEqual(type(com.communities), list)
        if len(com.communities) > 0:
            self.assertEqual(type(com.communities[0][0]), str)

    def test_lais2(self):
        g = get_string_graph()
        com = community.lais2(g)
        self.assertEqual(type(com.communities), list)
        if len(com.communities) > 0:
            self.assertEqual(type(com.communities[0][0]), str)

    def test_gdmp2(self):
        g = get_string_graph()
        com = community.gdmp2(g, min_threshold=.75)
        self.assertEqual(type(com.communities), list)
        if len(com.communities) > 0:
            self.assertEqual(type(com.communities[0][0]), str)

    def test_spinglass(self):
        g = get_string_graph()
        com = community.spinglass(g)
        self.assertEqual(type(com.communities), list)
        if len(com.communities) > 0:
            self.assertEqual(type(com.communities[0][0]), str)

    def test_walktrap(self):
        g = get_string_graph()
        com = community.walktrap(g)
        self.assertEqual(type(com.communities), list)
        if len(com.communities) > 0:
            self.assertEqual(type(com.communities[0][0]), str)

    def test_eigenvector(self):
        g = get_string_graph()
        com = community.eigenvector(g)
        self.assertEqual(type(com.communities), list)
        if len(com.communities) > 0:
            self.assertEqual(type(com.communities[0][0]), str)

    def test_Congo(self):
        g = get_string_graph()
        coms = community.congo(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_Conga(self):
        g = get_string_graph()
        coms = community.conga(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_agdl(self):
        g = get_string_graph()
        coms = community.agdl(g, 3, 2, 2, 0.5)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0][0]), str)


if __name__ == '__main__':
    unittest.main()
