import unittest
from nclib import community
from nclib import ensemble
from nclib import evaluation
import networkx as nx


class BunchExecTests(unittest.TestCase):

    def test_grid(self):
        g = nx.karate_club_graph()
        resolution = ensemble.Parameter(name="resolution", start=0.1, end=1, step=0.1)
        for res in ensemble.grid_execution(graph=g, method=community.louvain, parameters=[resolution]):
            print(res)

    def test_grid_search(self):
        g = nx.karate_club_graph()
        resolution = ensemble.Parameter(name="resolution", start=0.1, end=1, step=0.1)
        ensemble.grid_search(graph=g, method=community.louvain, parameters=[resolution],
                             quality_score=evaluation.erdos_renyi_modularity, aggregate=max)


if __name__ == '__main__':
    unittest.main()
