import unittest
from cdlib import algorithms
from cdlib import TemporalClustering, NamedClustering
from cdlib import evaluation
import networkx as nx
import random
import json


def get_temporal_network_clustering():

    tc = TemporalClustering()
    for t in range(10):
        g = nx.erdos_renyi_graph(100, 0.05)
        coms = algorithms.label_propagation(g)
        # simulating named clustering
        nc = NamedClustering(
            {i: c for i, c in enumerate(coms.communities)}, g, coms.method_name
        )

        tc.add_clustering(nc, t)

    return tc


class TemporalClusteringTests(unittest.TestCase):
    def test_TC(self):
        tc = get_temporal_network_clustering()
        self.assertIsInstance(tc, TemporalClustering)

        tids = tc.get_observation_ids()
        self.assertIsInstance(tids, list)

        for tid in tids:
            coms = tc.get_clustering_at(tid)
            self.assertIsInstance(coms, NamedClustering)

    def test_stability(self):
        tc = get_temporal_network_clustering()
        trend = tc.clustering_stability_trend(evaluation.normalized_mutual_information)
        self.assertEqual(len(trend), len(tc.get_observation_ids()) - 1)

    def test_community_access(self):
        tc = get_temporal_network_clustering()

        com = tc.get_community("0_0")
        self.assertIsInstance(com, list)

    def test_to_json(self):
        tc = get_temporal_network_clustering()
        js = tc.to_json()
        self.assertIsInstance(js, str)
        res = json.loads(js)
        self.assertIsNone(res["matchings"])
