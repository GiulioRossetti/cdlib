import unittest
from nclib import community
from nclib import ensemble
from nclib import evaluation
import networkx as nx


class BunchExecTests(unittest.TestCase):

    def test_grid(self):
        g = nx.karate_club_graph()
        resolution = ensemble.Parameter(name="resolution", start=0.1, end=1, step=0.1)

        for params, communities in ensemble.grid_execution(graph=g, method=community.louvain, parameters=[resolution]):
            self.assertIsInstance(params, dict)
            self.assertIsInstance(communities, list)

    def test_grid_search(self):
        g = nx.karate_club_graph()
        resolution = ensemble.Parameter(name="resolution", start=0.1, end=1, step=0.1)
        randomize = ensemble.BoolParameter(name="randomize")

        params, communities = ensemble.grid_search(graph=g, method=community.louvain,
                                                   parameters=[resolution, randomize],
                                                   quality_score=evaluation.erdos_renyi_modularity,
                                                   aggregate=max)
        self.assertIsInstance(params, tuple)
        self.assertIsInstance(communities, dict)


if __name__ == '__main__':
    unittest.main()
