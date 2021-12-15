import unittest
from cdlib import algorithms
import networkx as nx
import itertools
import random
import os

try:
    import pycombo as pycombo_part
except ModuleNotFoundError:
    pycombo_part = None

try:
    import igraph as ig
except ModuleNotFoundError:
    ig = None

try:
    import leidenalg
except ModuleNotFoundError:
    leidenalg = None

try:
    import infomap
except ModuleNotFoundError:
    infomap = None

try:
    import graph_tool.all as gt
except ModuleNotFoundError:
    gt = None

try:
    import ASLPAw_package as ASLPAw
except ModuleNotFoundError:
    ASLPAw = None

try:
    import GraphRicciCurvature as grc
except ModuleNotFoundError:
    grc = None


def get_string_graph():
    g = nx.karate_club_graph()
    node_map = {}
    for n in g.nodes():
        node_map[n] = "$%s$" % n
    nx.relabel_nodes(g, node_map, False)
    return g


def random_dag(N, P):
    nodes = [n for n in range(1, N + 1)]
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for n1, n2 in itertools.combinations(nodes, 2):
        p = random.random()
        if p <= P:
            if n1 > n2:
                G.add_edge(n2, n1)
            else:
                G.add_edge(n1, n2)
    return G


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
        if ig is not None:
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
        if leidenalg is not None:
            g = get_string_graph()
            coms = algorithms.leiden(g)
            self.assertEqual(type(coms.communities), list)
            if len(coms.communities) > 0:
                self.assertEqual(type(coms.communities[0]), list)
                self.assertEqual(type(coms.communities[0][0]), str)

    def test_significance(self):
        if leidenalg is not None:
            g = get_string_graph()
            coms = algorithms.significance_communities(g)
            self.assertEqual(type(coms.communities), list)
            if len(coms.communities) > 0:
                self.assertEqual(type(coms.communities[0]), list)
                self.assertEqual(type(coms.communities[0][0]), str)

    def test_surprise(self):
        if leidenalg is not None:
            g = get_string_graph()
            coms = algorithms.surprise_communities(g)
            self.assertEqual(type(coms.communities), list)
            if len(coms.communities) > 0:
                self.assertEqual(type(coms.communities[0]), list)
                self.assertEqual(type(coms.communities[0][0]), str)

    def test_cpm(self):
        if leidenalg is not None:
            g = get_string_graph()
            coms = algorithms.cpm(g)
            self.assertEqual(type(coms.communities), list)
            if len(coms.communities) > 0:
                self.assertEqual(type(coms.communities[0]), list)
                self.assertEqual(type(coms.communities[0][0]), str)

    def test_rbpots(self):
        if leidenalg is not None:
            g = get_string_graph()
            coms = algorithms.rb_pots(g)
            self.assertEqual(type(coms.communities), list)
            if len(coms.communities) > 0:
                self.assertEqual(type(coms.communities[0]), list)
                self.assertEqual(type(coms.communities[0][0]), str)

    def test_rberpots(self):
        if leidenalg is not None:
            g = get_string_graph()
            coms = algorithms.rber_pots(g)
            self.assertEqual(type(coms.communities), list)
            if len(coms.communities) > 0:
                self.assertEqual(type(coms.communities[0]), list)
                self.assertEqual(type(coms.communities[0][0]), str)

    def test_greedy_modularity(self):
        if leidenalg is not None:
            g = get_string_graph()
            try:
                coms = algorithms.greedy_modularity(g)
                self.assertEqual(type(coms.communities), list)
                if len(coms.communities) > 0:
                    self.assertEqual(type(coms.communities[0]), list)
                    self.assertEqual(type(coms.communities[0][0]), str)
            except:
                pass

    def test_infomap(self):
        if infomap is not None:
            g = get_string_graph()
            coms = algorithms.infomap(g)
            self.assertEqual(type(coms.communities), list)
            if len(coms.communities) > 0:
                self.assertEqual(type(coms.communities[0]), list)
                self.assertEqual(type(coms.communities[0][0]), str)
            if os.path.exists(".tree"):
                os.remove(".tree")

            h = nx.DiGraph()
            for e in g.edges():
                h.add_edge(e[0], e[1], weight=3)

            coms = algorithms.infomap(h)
            self.assertEqual(type(coms.communities), list)
            if len(coms.communities) > 0:
                self.assertEqual(type(coms.communities[0]), list)
                self.assertEqual(type(coms.communities[0][0]), str)
            if os.path.exists(".tree"):
                os.remove(".tree")

            gg = ig.Graph(directed=True)
            gg.add_vertices([v for v in h.nodes()])
            gg.add_edges([(u, v) for u, v in h.edges()])

            coms = algorithms.infomap(gg)
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
        if ig is not None:
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

        g = nx.karate_club_graph()
        coms = algorithms.multicom(g, seed_node=0)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_em(self):
        g = get_string_graph()
        coms = algorithms.em(g, k=3)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)

        g = nx.karate_club_graph()
        coms = algorithms.em(g, k=3)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

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
                self.assertEqual(type(communities.communities[0][0]), str)

        g = nx.karate_club_graph()

        communities = algorithms.markov_clustering(g)
        self.assertEqual(type(communities.communities), list)
        if len(communities.communities) > 0:
            self.assertEqual(type(communities.communities[0]), list)
            if len(communities.communities[0]) > 0:
                self.assertEqual(type(communities.communities[0][0]), int)

    def test_bigClam(self):
        g = nx.karate_club_graph()
        coms = algorithms.big_clam(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_lemon(self):
        g = get_string_graph()
        seeds = ["$0$", "$2$", "$3$"]
        com = algorithms.lemon(g, seeds, min_com_size=10, max_com_size=50)
        self.assertEqual(type(com.communities), list)
        if len(com.communities) > 0:
            self.assertEqual(type(com.communities[0]), list)
            self.assertEqual(type(com.communities[0][0]), str)

        g = nx.karate_club_graph()
        seeds = [0, 2, 3]
        com = algorithms.lemon(g, seeds, min_com_size=10, max_com_size=50)
        self.assertEqual(type(com.communities), list)
        if len(com.communities) > 0:
            self.assertEqual(type(com.communities[0]), list)
            self.assertEqual(type(com.communities[0][0]), int)

    def test_lais2(self):
        g = get_string_graph()
        com = algorithms.lais2(g)
        self.assertEqual(type(com.communities), list)
        if len(com.communities) > 0:
            self.assertEqual(type(com.communities[0]), list)
            self.assertEqual(type(com.communities[0][0]), str)

    def test_gdmp2(self):
        g = get_string_graph()
        com = algorithms.gdmp2(g, min_threshold=0.75)
        self.assertEqual(type(com.communities), list)
        if len(com.communities) > 0:
            self.assertEqual(type(com.communities[0]), list)
            self.assertEqual(type(com.communities[0][0]), str)

    def test_spinglass(self):
        if ig is not None:
            g = get_string_graph()
            com = algorithms.spinglass(g)
            self.assertEqual(type(com.communities), list)
            if len(com.communities) > 0:
                self.assertEqual(type(com.communities[0]), list)
                self.assertEqual(type(com.communities[0][0]), str)

    def test_walktrap(self):
        if ig is not None:
            g = get_string_graph()
            com = algorithms.walktrap(g)
            self.assertEqual(type(com.communities), list)
            if len(com.communities) > 0:
                self.assertEqual(type(com.communities[0]), list)
                self.assertEqual(type(com.communities[0][0]), str)

    def test_eigenvector(self):
        if ig is not None:
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
        coms = algorithms.agdl(g, 3, 2)
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
            self.assertEqual(type(coms.communities[0][0]), str)
            self.assertIsInstance(coms.allocation_matrix, dict)
            self.assertEqual(len(coms.allocation_matrix), g.number_of_nodes())

    def test_principled(self):
        g = get_string_graph()
        coms = algorithms.principled_clustering(g, 3)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)
            self.assertIsInstance(coms.allocation_matrix, dict)
            self.assertEqual(len(coms.allocation_matrix), g.number_of_nodes())

    def test_sbm_dl(self):
        if gt is not None:
            g = get_string_graph()
            coms = algorithms.sbm_dl(g)
            self.assertEqual(type(coms.communities), list)
            if len(coms.communities) > 0:
                self.assertEqual(type(coms.communities[0]), list)
                self.assertEqual(type(coms.communities[0][0]), str)

    def test_sbm_nested_dl(self):
        if gt is not None:
            g = get_string_graph()
            coms = algorithms.sbm_dl_nested(g)
            self.assertEqual(type(coms.communities), list)
            if len(coms.communities) > 0:
                self.assertEqual(type(coms.communities[0]), list)
                self.assertEqual(type(coms.communities[0][0]), str)

    def test_danmf(self):
        g = get_string_graph()
        coms = algorithms.danmf(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_egonet_splitter(self):
        g = get_string_graph()
        coms = algorithms.egonet_splitter(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_nnsed(self):
        g = nx.karate_club_graph()
        coms = algorithms.nnsed(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_mnmf(self):
        g = nx.karate_club_graph()
        coms = algorithms.mnmf(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_edmot(self):
        g = nx.karate_club_graph()
        coms = algorithms.edmot(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_bimlpa(self):
        g = nx.algorithms.bipartite.random_graph(50, 50, 0.25)
        coms = algorithms.bimlpa(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_aslpaw(self):
        if ASLPAw is not None:
            g = nx.karate_club_graph()
            coms = algorithms.aslpaw(g)

            self.assertEqual(type(coms.communities), list)
            if len(coms.communities) > 0:
                self.assertEqual(type(coms.communities[0]), list)
                self.assertEqual(type(coms.communities[0][0]), int)

    def test_percomvc(self):
        g = nx.karate_club_graph()
        coms = algorithms.percomvc(g)

        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_chinese_whispers(self):
        g = get_string_graph()

        communities = algorithms.chinesewhispers(g)
        self.assertEqual(type(communities.communities), list)
        if len(communities.communities) > 0:
            self.assertEqual(type(communities.communities[0]), list)
            if len(communities.communities[0]) > 0:
                self.assertEqual(type(communities.communities[0][0]), str)

        g = nx.karate_club_graph()

        communities = algorithms.chinesewhispers(g)
        self.assertEqual(type(communities.communities), list)
        if len(communities.communities) > 0:
            self.assertEqual(type(communities.communities[0]), list)
            if len(communities.communities[0]) > 0:
                self.assertEqual(type(communities.communities[0][0]), int)

    def test_wCommunities(self):

        g = get_string_graph()
        nx.set_edge_attributes(g, values=1, name="weight")

        communities = algorithms.wCommunity(
            g, min_bel_degree=0.6, threshold_bel_degree=0.6
        )
        self.assertEqual(type(communities.communities), list)
        if len(communities.communities) > 0:
            self.assertEqual(type(communities.communities[0]), list)
            if len(communities.communities[0]) > 0:
                self.assertEqual(type(communities.communities[0][0]), str)

        g = nx.karate_club_graph()
        nx.set_edge_attributes(g, values=1, name="weight")

        communities = algorithms.wCommunity(
            g, min_bel_degree=0.6, threshold_bel_degree=0.6
        )
        self.assertEqual(type(communities.communities), list)
        if len(communities.communities) > 0:
            self.assertEqual(type(communities.communities[0]), list)
            if len(communities.communities[0]) > 0:
                self.assertEqual(type(communities.communities[0][0]), int)

    def test_siblinarity_antichain(self):

        g = random_dag(100, 0.1)
        communities = algorithms.siblinarity_antichain(g, Lambda=1)
        self.assertEqual(type(communities.communities), list)
        if len(communities.communities) > 0:
            self.assertEqual(type(communities.communities[0]), list)
            if len(communities.communities[0]) > 0:
                self.assertEqual(type(communities.communities[0][0]), int)

    def test_ga(self):

        g = nx.karate_club_graph()

        communities = algorithms.ga(g)
        self.assertEqual(type(communities.communities), list)
        if len(communities.communities) > 0:
            self.assertEqual(type(communities.communities[0]), list)
            if len(communities.communities[0]) > 0:
                self.assertEqual(type(communities.communities[0][0]), int)

    def test_belief(self):

        g = nx.karate_club_graph()

        communities = algorithms.belief(g)
        self.assertEqual(type(communities.communities), list)
        if len(communities.communities) > 0:
            self.assertEqual(type(communities.communities[0]), list)
            if len(communities.communities[0]) > 0:
                self.assertEqual(type(communities.communities[0][0]), int)

    def test_CPM_Bipartite(self):
        g = ig.Graph.Erdos_Renyi(n=80, m=600)
        g.vs["type"] = 0
        g.vs[15:]["type"] = 1

        coms = algorithms.CPM_Bipartite(g, 0.3)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

        g = nx.algorithms.bipartite.random_graph(50, 50, 0.25)
        coms = algorithms.CPM_Bipartite(g, 0.3)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_infomap_Bipartite(self):
        g = nx.algorithms.bipartite.random_graph(300, 100, 0.2)
        coms = algorithms.infomap_bipartite(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_condor(self):

        g = nx.algorithms.bipartite.random_graph(300, 100, 0.2)

        communities = algorithms.condor(g)
        self.assertEqual(type(communities.communities), list)
        if len(communities.communities) > 0:
            self.assertEqual(type(communities.communities[0]), list)
            if len(communities.communities[0]) > 0:
                self.assertEqual(type(communities.communities[0][0]), int)

    def test_threshold_clustering(self):
        g = get_string_graph()

        for _, _, d in g.edges(data=True):
            d["weight"] = 3

        coms = algorithms.threshold_clustering(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)

    def test_lswl(self):

        G = nx.karate_club_graph()

        coms = algorithms.lswl(G, 1, online=True)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

        coms = algorithms.lswl(G, 1, online=False)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

        coms = algorithms.lswl_plus(G, merge_outliers=False, detect_overlap=True)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

        coms = algorithms.lswl_plus(G, merge_outliers=True, detect_overlap=False)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_mod_m(self):

        G = nx.karate_club_graph()

        coms = algorithms.mod_m(G, 1)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_mod_r(self):

        G = nx.karate_club_graph()

        coms = algorithms.mod_r(G, 1)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_head_tail(self):

        G = nx.karate_club_graph()

        coms = algorithms.head_tail(G, 0.8)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_core_expansion(self):

        G = nx.karate_club_graph()

        coms = algorithms.core_expansion(G)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_lpanni(self):

        G = nx.karate_club_graph()

        coms = algorithms.lpanni(G)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_lpam(self):
        G = nx.karate_club_graph()

        coms = algorithms.lpam(G, k=2, threshold=0.4, distance="amp")
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_dcs(self):
        G = nx.karate_club_graph()

        coms = algorithms.dcs(G)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_umstmo(self):
        G = nx.karate_club_graph()

        coms = algorithms.umstmo(G)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_kcut(self):
        G = get_string_graph()

        try:
            coms = algorithms.kcut(G)
            self.assertEqual(type(coms.communities), list)
            if len(coms.communities) > 0:
                self.assertEqual(type(coms.communities[0]), list)
                self.assertEqual(type(coms.communities[0][0]), str)
        except ValueError:
            print("Kcut error to be checked (conda packaging)")

    def test_symmnmf(self):
        G = nx.karate_club_graph()

        coms = algorithms.symmnmf(G)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_scd(self):
        G = nx.karate_club_graph()

        coms = algorithms.scd(G)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_gemsec(self):
        G = nx.karate_club_graph()

        coms = algorithms.gemsec(G)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_pycombo(self):

        if pycombo_part is not None:
            G = nx.karate_club_graph()

            coms = algorithms.pycombo(G)
            self.assertEqual(type(coms.communities), list)
            if len(coms.communities) > 0:
                self.assertEqual(type(coms.communities[0]), list)
                self.assertEqual(type(coms.communities[0][0]), int)

    def test_walkscan(self):
        G = nx.karate_club_graph()

        coms = algorithms.walkscan(G)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_paris(self):
        G = nx.karate_club_graph()

        coms = algorithms.paris(G)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_ricci(self):
        if grc is not None:
            G = nx.karate_club_graph()

            coms = algorithms.ricci_community(G)
            self.assertEqual(type(coms.communities), list)
            if len(coms.communities) > 0:
                self.assertEqual(type(coms.communities[0]), list)
                self.assertEqual(type(coms.communities[0][0]), int)

    def test_endntm(self):
        G = nx.karate_club_graph()

        coms_l = [
            algorithms.louvain(G),
            algorithms.label_propagation(G),
            algorithms.walktrap(G),
        ]
        coms = algorithms.endntm(G, coms_l)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_scd(self):
        G = nx.karate_club_graph()

        coms = algorithms.spectral(G, kmax=4)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_mcode(self):
        G = nx.karate_club_graph()

        coms = algorithms.mcode(G)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

        nx.set_edge_attributes(G, values=2, name="weight")
        coms = algorithms.mcode(G, weights="weight")

        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_ipca(self):
        G = nx.karate_club_graph()

        coms = algorithms.ipca(G)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

        nx.set_edge_attributes(G, values=2, name="weight")
        coms = algorithms.ipca(G, weights="weight")

        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_dpclus(self):
        G = nx.karate_club_graph()

        coms = algorithms.dpclus(G)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

        coms = algorithms.dpclus(G, overlap=False)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

        nx.set_edge_attributes(G, values=2, name="weight")
        coms = algorithms.dpclus(G, weights="weight")

        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_coach(self):
        G = nx.karate_club_graph()

        coms = algorithms.coach(G)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_entropy(self):
        G = nx.karate_club_graph()

        coms = algorithms.graph_entropy(G)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

        nx.set_edge_attributes(G, values=2, name="weight")
        coms = algorithms.graph_entropy(G, weights="weight")

        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_entropy_2(self):
        G = nx.karate_club_graph()

        coms = algorithms.ebgc(G)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_rsc(self):
        G = nx.karate_club_graph()

        coms = algorithms.r_spectral_clustering(
            G, n_clusters=2, method="percentile", percentile=20
        )
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

        coms = algorithms.r_spectral_clustering(G, n_clusters=2, method="vanilla")
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

        coms = algorithms.r_spectral_clustering(
            G, n_clusters=2, method="regularized", percentile=20
        )
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

        coms = algorithms.r_spectral_clustering(
            G, n_clusters=2, method="regularized_with_kmeans"
        )
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

        coms = algorithms.r_spectral_clustering(
            G, n_clusters=2, method="sklearn_spectral_embedding"
        )
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

        coms = algorithms.r_spectral_clustering(
            G, n_clusters=2, method="sklearn_kmeans"
        )
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)
