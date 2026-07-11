import unittest
from cdlib import algorithms
import networkx as nx
import numpy as np
import itertools
import random
import os
from cdlib.prompt_utils import prompt_import_failure

# try:
#     import karateclub
# except ModuleNotFoundError:
#     karateclub = None

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
except Exception as exception:
    prompt_import_failure("infomap", exception)

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

try:
    import hidef
except ModuleNotFoundError:
    hidef = None

try:
    import sknetwork
except ModuleNotFoundError:
    sknetwork = None


try:
    import bayanpy as by
except ModuleNotFoundError:
    by = None


try:
    from cdlib.algorithms.internal.LPAM import LPAM
except ModuleNotFoundError:
    LPAM = None


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

        coms2 = algorithms.louvain(g, partition=coms)
        self.assertEqual(type(coms2.communities), list)
        if len(coms2.communities) > 0:
            self.assertEqual(type(coms2.communities[0]), list)
            self.assertEqual(type(coms2.communities[0][0]), str)

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
        if True or infomap is not None:
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
        coms = algorithms.label_propagation_cordasco_gargano(g)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), str)

        g = nx.karate_club_graph()
        self.assertIs(algorithms.label_propagation, algorithms.label_propagation_raghavan)
        legacy = algorithms.label_propagation(g)
        self.assertEqual(type(legacy.communities), list)
        self.assertEqual(legacy.method_name, "Label Propagation Raghavan")

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

        g = nx.karate_club_graph()

        communities = algorithms.markov_clustering(g)
        self.assertEqual(type(communities.communities), list)
        if len(communities.communities) > 0:
            self.assertEqual(type(communities.communities[0]), list)
            if len(communities.communities[0]) > 0:
                self.assertEqual(type(communities.communities[0][0]), int)

    def test_big_clam(self):
        g = nx.karate_club_graph()
        coms = algorithms.big_clam(g)
        self.assertEqual(type(coms.communities), list)
        self.assertFalse(coms.overlap)
        if len(coms.communities) > 0:
            for com in coms.communities:
                self.assertEqual(type(com), list)
                if len(com) > 0:
                    self.assertEqual(type(com[0]), int)

        coms = algorithms.big_clam(g, affiliation_method="threshold")
        self.assertEqual(type(coms.communities), list)
        self.assertTrue(coms.overlap)
        if len(coms.communities) > 0:
            for com in coms.communities:
                self.assertEqual(type(com), list)
                if len(com) > 0:
                    self.assertEqual(type(com[0]), int)

        with self.assertRaises(ValueError):
            algorithms.big_clam(g, affiliation_method="invalid_method")

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

    # def test_danmf(self):
    #    if karateclub is None:
    #        return
    #    g = get_string_graph()
    #    coms = algorithms.danmf(g)
    #    self.assertEqual(type(coms.communities), list)
    #    if len(coms.communities) > 0:
    #        self.assertEqual(type(coms.communities[0]), list)
    #        self.assertEqual(type(coms.communities[0][0]), str)

    # def test_egonet_splitter(self):
    #     if karateclub is None:
    #         return
    #     g = get_string_graph()
    #     coms = algorithms.egonet_splitter(g)
    #     self.assertEqual(type(coms.communities), list)
    #     if len(coms.communities) > 0:
    #         self.assertEqual(type(coms.communities[0]), list)
    #         self.assertEqual(type(coms.communities[0][0]), str)

    # def test_nnsed(self):
    #     if karateclub is None:
    #         return
    #     g = nx.karate_club_graph()
    #     coms = algorithms.nnsed(g)
    #     self.assertEqual(type(coms.communities), list)
    #     if len(coms.communities) > 0:
    #         self.assertEqual(type(coms.communities[0]), list)
    #         self.assertEqual(type(coms.communities[0][0]), int)

    # def test_mnmf(self):
    #     if karateclub is None:
    #         return
    #     g = nx.karate_club_graph()
    #     coms = algorithms.mnmf(g)
    #     self.assertEqual(type(coms.communities), list)
    #     if len(coms.communities) > 0:
    #         self.assertEqual(type(coms.communities[0]), list)
    #         self.assertEqual(type(coms.communities[0][0]), int)

    # def test_edmot(self):
    #     if karateclub is None:
    #         return
    #     g = nx.karate_club_graph()
    #     coms = algorithms.edmot(g)
    #     self.assertEqual(type(coms.communities), list)
    #     if len(coms.communities) > 0:
    #         self.assertEqual(type(coms.communities[0]), list)
    #         self.assertEqual(type(coms.communities[0][0]), int)

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

    def test_highway(self):
        g = nx.Graph()
        g.add_edges_from(
            [
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3),
                (2, 3),
                (4, 5),
                (4, 6),
                (4, 7),
                (5, 6),
                (5, 7),
                (6, 7),
                (3, 4),
            ]
        )

        coms = algorithms.highway(g)

        self.assertEqual(type(coms.communities), list)
        self.assertTrue(coms.overlap)
        self.assertEqual(coms.method_name, "Highway")
        self.assertEqual(
            {frozenset(comm) for comm in coms.communities},
            {frozenset({0, 1, 2, 3}), frozenset({4, 5, 6, 7})},
        )

        string_graph = nx.relabel_nodes(
            g, {n: f"${n}$" for n in g.nodes()}, copy=True
        )
        string_coms = algorithms.highway(string_graph)

        self.assertEqual(type(string_coms.communities), list)
        self.assertTrue(string_coms.overlap)
        self.assertEqual(string_coms.method_name, "Highway")
        self.assertEqual(
            {frozenset(comm) for comm in string_coms.communities},
                {
                    frozenset({"$0$", "$1$", "$2$", "$3$"}),
                    frozenset({"$4$", "$5$", "$6$", "$7$"}),
                },
        )

    def test_splitter(self):
        g = nx.Graph()
        g.add_edges_from(
            [
                (0, 1),
                (1, 2),
                (2, 0),
                (2, 3),
                (3, 4),
                (4, 2),
            ]
        )

        coms = algorithms.splitter(g, resolution=1.0, min_community_size=2)

        self.assertEqual(type(coms.communities), list)
        self.assertTrue(coms.overlap)
        self.assertEqual(coms.method_name, "Splitter")
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)

    def test_apal(self):
        g = nx.Graph()
        g.add_edges_from(
            [
                (0, 1),
                (1, 2),
                (2, 0),
                (2, 3),
                (3, 4),
                (4, 2),
            ]
        )

        coms = algorithms.apal(g, threshold=0.5)

        self.assertEqual(type(coms.communities), list)
        self.assertTrue(coms.overlap)
        self.assertEqual(coms.method_name, "APAL")
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)

    @unittest.skip("Skipped by default; requires optional torch dependency for NOCD.")
    def test_nocd(self):
        g = nx.Graph()
        g.add_edges_from(
            [
                (0, 1),
                (1, 2),
                (2, 0),
                (2, 3),
                (3, 4),
                (4, 2),
                (4, 5),
                (5, 0),
            ]
        )

        coms = algorithms.nocd(
            g,
            dimensions=2,
            hidden_sizes=(8,),
            epochs=5,
            display_step=5,
            batch_size=8,
            threshold=0.0,
            feature_mode="identity",
            seed=42,
        )

        self.assertEqual(type(coms.communities), list)
        self.assertTrue(coms.overlap)
        self.assertEqual(coms.method_name, "NOCD")
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)

    def test_lazyfox(self):
        g = nx.Graph()
        g.add_edges_from(
            [
                (0, 1),
                (1, 2),
                (2, 0),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 3),
            ]
        )

        coms = algorithms.lazyfox(g, threshold=0.01)

        self.assertEqual(type(coms.communities), list)
        self.assertTrue(coms.overlap)
        self.assertEqual(coms.method_name, "LazyFox")
        self.assertGreaterEqual(len(coms.communities), 2)
        self.assertTrue(any({0, 1, 2}.issubset(set(c)) for c in coms.communities))
        self.assertTrue(any({3, 4, 5}.issubset(set(c)) for c in coms.communities))

    def test_wghac(self):
        g = nx.Graph()
        g.add_edges_from(
            [
                (0, 1),
                (1, 2),
                (2, 0),
                (2, 3),
                (3, 4),
                (4, 2),
            ]
        )

        ct_distance_matrix = np.array(
            [
                [0.0, 1.0, 1.0, 10.0, 10.0],
                [1.0, 0.0, 1.0, 10.0, 10.0],
                [1.0, 1.0, 0.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 0.0, 1.0],
                [10.0, 10.0, 10.0, 1.0, 0.0],
            ]
        )

        coms = algorithms.wghac(
            g,
            min_base_size=2,
            linkage_method="single",
            ct_distance_matrix=ct_distance_matrix,
        )

        self.assertEqual(type(coms.communities), list)
        self.assertTrue(coms.overlap)
        self.assertEqual(coms.method_name, "wGHAC")
        self.assertGreaterEqual(len(coms.communities), 2)
        self.assertTrue(any({0, 1, 2}.issubset(set(c)) for c in coms.communities))
        self.assertTrue(any({2, 3, 4}.issubset(set(c)) for c in coms.communities))

    def test_seed_node_cd(self):
        g = nx.Graph()
        g.add_edges_from(
            (u, v) for u in range(5) for v in range(u + 1, 5)
        )
        g.add_edges_from(
            (u, v) for u in range(5, 10) for v in range(u + 1, 10)
        )
        g.add_edge(4, 5)

        coms = algorithms.seed_node_cd(g)

        self.assertEqual(type(coms.communities), list)
        self.assertFalse(coms.overlap)
        self.assertEqual(coms.method_name, "Seed-Node CD")
        self.assertEqual(len(coms.communities), 2)
        self.assertTrue(any({0, 1, 2, 3, 4} == set(c) for c in coms.communities))
        self.assertTrue(any({5, 6, 7, 8, 9} == set(c) for c in coms.communities))

    @unittest.skipUnless(hidef is not None, "Skipped by default; requires optional HiDeF dependency.")
    def test_hidef(self):
        g = nx.Graph()
        g.add_edges_from(
            (u, v) for u in range(5) for v in range(u + 1, 5)
        )
        g.add_edges_from(
            (u, v) for u in range(5, 10) for v in range(u + 1, 10)
        )
        g.add_edge(4, 5)

        coms = algorithms.hidef(
            g,
            minres=0.1,
            maxres=0.2,
            sample=1.0,
            jaccard=0.75,
            alg="leiden",
            density=1.0,
            neighbors=1,
            k=2,
            f=1.0,
            p=0,
            numthreads=1,
        )

        self.assertEqual(type(coms.communities), list)
        self.assertTrue(coms.overlap)
        self.assertEqual(coms.method_name, "HiDeF")
        self.assertEqual(len(coms.communities), 2)
        self.assertTrue(any({0, 1, 2, 3, 4} == set(c) for c in coms.communities))
        self.assertTrue(any({5, 6, 7, 8, 9} == set(c) for c in coms.communities))

    def test_l1_ppr(self):
        g = get_string_graph()
        seeds = ["$0$", "$2$", "$3$"]
        coms = algorithms.l1_ppr(g, seeds, min_comm_size=3, max_comm_size=10)
        self.assertEqual(type(coms.communities), list)
        self.assertTrue(coms.overlap)
        self.assertEqual(coms.method_name, "L1 PPR")
        self.assertEqual(len(coms.communities), 1)
        self.assertTrue(set(seeds).issubset(set(coms.communities[0])))
        self.assertEqual(type(coms.communities[0][0]), str)

        g = nx.karate_club_graph()
        seeds = [0, 2, 3]
        coms = algorithms.l1_ppr(g, seeds, min_comm_size=3, max_comm_size=10)
        self.assertEqual(type(coms.communities), list)
        self.assertTrue(coms.overlap)
        self.assertEqual(len(coms.communities), 1)
        self.assertTrue(set(seeds).issubset(set(coms.communities[0])))
        self.assertEqual(type(coms.communities[0][0]), int)

    def test_ppr_sweep(self):
        g = get_string_graph()
        seeds = ["$0$", "$2$", "$3$"]
        coms = algorithms.ppr_sweep(g, seeds, min_comm_size=3, max_comm_size=10)
        self.assertEqual(type(coms.communities), list)
        self.assertTrue(coms.overlap)
        self.assertEqual(coms.method_name, "PPR Sweep")
        self.assertEqual(len(coms.communities), 1)
        self.assertTrue(set(seeds).issubset(set(coms.communities[0])))
        self.assertEqual(type(coms.communities[0][0]), str)

        g = nx.karate_club_graph()
        seeds = [0, 2, 3]
        coms = algorithms.ppr_sweep(g, seeds, min_comm_size=3, max_comm_size=10)
        self.assertEqual(type(coms.communities), list)
        self.assertTrue(coms.overlap)
        self.assertEqual(len(coms.communities), 1)
        self.assertTrue(set(seeds).issubset(set(coms.communities[0])))
        self.assertEqual(type(coms.communities[0][0]), int)

    def test_hk_sweep(self):
        g = get_string_graph()
        seeds = ["$0$", "$2$", "$3$"]
        coms = algorithms.hk_sweep(g, seeds, min_comm_size=3, max_comm_size=10)
        self.assertEqual(type(coms.communities), list)
        self.assertTrue(coms.overlap)
        self.assertEqual(coms.method_name, "Heat Kernel Sweep")
        self.assertEqual(len(coms.communities), 1)
        self.assertTrue(set(seeds).issubset(set(coms.communities[0])))
        self.assertEqual(type(coms.communities[0][0]), str)

        g = nx.karate_club_graph()
        seeds = [0, 2, 3]
        coms = algorithms.hk_sweep(g, seeds, min_comm_size=3, max_comm_size=10)
        self.assertEqual(type(coms.communities), list)
        self.assertTrue(coms.overlap)
        self.assertEqual(len(coms.communities), 1)
        self.assertTrue(set(seeds).issubset(set(coms.communities[0])))
        self.assertEqual(type(coms.communities[0][0]), int)

    def test_clauset(self):
        g = get_string_graph()
        seeds = ["$0$", "$2$", "$3$"]
        coms = algorithms.clauset(g, seeds, min_comm_size=3, max_comm_size=10)
        self.assertEqual(type(coms.communities), list)
        self.assertTrue(coms.overlap)
        self.assertEqual(coms.method_name, "Clauset")
        self.assertEqual(len(coms.communities), 1)
        self.assertTrue(set(seeds).issubset(set(coms.communities[0])))
        self.assertEqual(type(coms.communities[0][0]), str)

        g = nx.karate_club_graph()
        seeds = [0, 2, 3]
        coms = algorithms.clauset(g, seeds, min_comm_size=3, max_comm_size=10)
        self.assertEqual(type(coms.communities), list)
        self.assertTrue(coms.overlap)
        self.assertEqual(len(coms.communities), 1)
        self.assertTrue(set(seeds).issubset(set(coms.communities[0])))
        self.assertEqual(type(coms.communities[0][0]), int)

    # def test_chinese_whispers(self):
    #    g = get_string_graph()
    #
    #    communities = algorithms.chinesewhispers(g)
    #    self.assertEqual(type(communities.communities), list)
    #    if len(communities.communities) > 0:
    #        self.assertEqual(type(communities.communities[0]), list)
    #        if len(communities.communities[0]) > 0:
    #            self.assertEqual(type(communities.communities[0][0]), str)
    #
    #    g = nx.karate_club_graph()
    #
    #    communities = algorithms.chinesewhispers(g)
    #    self.assertEqual(type(communities.communities), list)
    #    if len(communities.communities) > 0:
    #        self.assertEqual(type(communities.communities[0]), list)
    #        if len(communities.communities[0]) > 0:
    #            self.assertEqual(type(communities.communities[0][0]), int)

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

        g = nx.algorithms.bipartite.random_graph(50, 50, 0.25)

        if leidenalg is None:
            return
        coms = algorithms.CPM_Bipartite(g, 0.5)
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
        if infomap is None:
            return
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

    @unittest.skipUnless(
        sknetwork is not None, "Skipped by default; requires optional scikit-network dependency."
    )
    def test_bi_louvain(self):
        g = nx.Graph()
        g.add_nodes_from([0, 1, 2, 3], bipartite=0)
        g.add_nodes_from([4, 5, 6, 7], bipartite=1)
        g.add_edges_from([(0, 4), (0, 5), (1, 4), (1, 5), (2, 6), (2, 7), (3, 6), (3, 7)])

        coms = algorithms.bi_louvain(g, resolution=1.0)
        self.assertEqual(type(coms.communities), list)
        self.assertEqual(coms.method_name, "Bi-Louvain")
        self.assertEqual(type(coms.communities[0]), list)
        self.assertGreaterEqual(len(coms.communities), 2)

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
        if LPAM is not None:
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

    # def test_symmnmf(self):
    #     if karateclub is None:
    #         return
    #     G = nx.karate_club_graph()
    #
    #     coms = algorithms.symmnmf(G)
    #     self.assertEqual(type(coms.communities), list)
    #     if len(coms.communities) > 0:
    #         self.assertEqual(type(coms.communities[0]), list)
    #         self.assertEqual(type(coms.communities[0][0]), int)

    # def test_scd(self):
    #     G = nx.karate_club_graph()
    #
    #     coms = algorithms.scd(G)
    #     self.assertEqual(type(coms.communities), list)
    #     if len(coms.communities) > 0:
    #         self.assertEqual(type(coms.communities[0]), list)
    #         self.assertEqual(type(coms.communities[0][0]), int)

    # def test_gemsec(self):
    #     if karateclub is None:
    #         return
    #     G = nx.karate_club_graph()
    #
    #     coms = algorithms.gemsec(G)
    #     self.assertEqual(type(coms.communities), list)
    #     if len(coms.communities) > 0:
    #         self.assertEqual(type(coms.communities[0]), list)
    #         self.assertEqual(type(coms.communities[0][0]), int)

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
            algorithms.label_propagation_raghavan(G),
            algorithms.walktrap(G),
        ]
        coms = algorithms.endntm(G, coms_l)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_scd(self):
        G = nx.karate_club_graph()

        coms = algorithms.spectral(G, kmax=2)
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

    @unittest.skip("Skipped by default; requires optional BayanPy/Gurobi dependencies.")
    def test_bayan(self):

        if by is not None:

            try:
                import gurobipy as gp
            except ModuleNotFoundError:
                return

            G = nx.florentine_families_graph()

            coms = algorithms.bayan(G)
            self.assertEqual(type(coms.communities), list)
            if len(coms.communities) > 0:
                self.assertEqual(type(coms.communities[0]), list)
                self.assertEqual(type(coms.communities[0][0]), str)
