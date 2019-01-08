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
            self.assertIsInstance(params, tuple)
            self.assertIsInstance(communities, list)

    def test_grid_search(self):
        g = nx.karate_club_graph()
        resolution = ensemble.Parameter(name="resolution", start=0.1, end=1, step=0.1)
        randomize = ensemble.BoolParameter(name="randomize")

        params, communities, scoring = ensemble.grid_search(graph=g, method=community.louvain,
                                                            parameters=[resolution, randomize],
                                                            quality_score=evaluation.erdos_renyi_modularity,
                                                            aggregate=max)
        self.assertIsInstance(params, tuple)
        self.assertIsInstance(communities, list)
        self.assertIsInstance(scoring, float)

    def test_pool(self):
        g = nx.karate_club_graph()

        # Louvain
        resolution = ensemble.Parameter(name="resolution", start=0.1, end=1, step=0.1)
        randomize = ensemble.BoolParameter(name="randomize")
        louvain_conf = [resolution, randomize]

        # Angel
        threshold = ensemble.Parameter(name="threshold", start=0.1, end=1, step=0.1)
        angel_conf = [threshold]

        methods = [community.louvain, community.angel]

        for method, parameters, communities in ensemble.pool(g, methods, [louvain_conf, angel_conf]):
            self.assertIsInstance(method, str)
            self.assertIsInstance(parameters, tuple)
            self.assertIsInstance(communities, list)

    def test_pool_filtered(self):
        g = nx.karate_club_graph()

        # Louvain
        resolution = ensemble.Parameter(name="resolution", start=0.1, end=1, step=0.1)
        randomize = ensemble.BoolParameter(name="randomize")
        louvain_conf = [resolution, randomize]

        # Angel
        threshold = ensemble.Parameter(name="threshold", start=0.1, end=1, step=0.1)
        angel_conf = [threshold]

        methods = [community.louvain, community.angel]

        for method, parameters, communities, scoring in \
                ensemble.pool_grid_filter(g, methods, [louvain_conf, angel_conf],
                                          quality_score=evaluation.erdos_renyi_modularity,
                                          aggregate=max):

            self.assertIsInstance(method, str)
            self.assertIsInstance(parameters, tuple)
            self.assertIsInstance(communities, list)
            self.assertIsInstance(scoring, float)


if __name__ == '__main__':
    unittest.main()
