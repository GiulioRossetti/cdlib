import unittest
from cdlib import algorithms
from cdlib import evaluation
import networkx as nx


class RankingTests(unittest.TestCase):

    def test_ranking(self):
        g = nx.karate_club_graph()
        coms = algorithms.louvain(g)
        coms2 = algorithms.demon(g, 0.25)
        rk = evaluation.Ranking(g, [coms, coms2])
        r=rk.rank(evaluation.fraction_over_median_degree)
        print(r)