import unittest
import cdlib
from cdlib import benchmark
import dynetx as dn


class DNBenchTest(unittest.TestCase):
    def test_RDyn(self):

        G, coms = benchmark.RDyn()
        self.assertIsInstance(G, dn.DynGraph)
        self.assertIsInstance(coms, cdlib.TemporalClustering)
        obs = coms.get_observation_ids()

        for t in obs:
            cm = coms.get_clustering_at(t)
            self.assertIsInstance(cm, cdlib.NamedClustering)
