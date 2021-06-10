import unittest
from cdlib import algorithms
from cdlib import TemporalClustering
import dynetx as dn
import networkx as nx
import random


def get_temporal_network_clustering():

    tc = TemporalClustering()
    for t in range(10):
        g = nx.erdos_renyi_graph(100, 0.05)
        coms = algorithms.louvain(g)
        tc.add_clustering(coms, t)

    return tc


class TemporalDCDTests(unittest.TestCase):
    def test_eTiles(self):
        dg = dn.DynGraph()
        for x in range(10):
            g = nx.erdos_renyi_graph(200, 0.05)
            dg.add_interactions_from(list(g.edges()), t=x)

        coms = algorithms.tiles(dg, 2)
        self.assertIsInstance(coms, TemporalClustering)
