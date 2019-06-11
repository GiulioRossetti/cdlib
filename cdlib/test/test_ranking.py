import unittest
from cdlib import algorithms
from cdlib import evaluation
import networkx as nx


class RankingTests(unittest.TestCase):

    def test_ranking(self):
        g = nx.karate_club_graph()
        coms = algorithms.louvain(g)
        coms2 = algorithms.demon(g, 0.25)
        coms3 = algorithms.label_propagation(g)
        coms4 = algorithms.angel(g, 0.6)
        coms5 = algorithms.angel(g, 0.9)

        rk = evaluation.Ranking(g, [coms2, coms,coms3 ,coms4, coms5])

        rk.rank(evaluation.fraction_over_median_degree)
        rk.rank(evaluation.edges_inside)
        rk.rank(evaluation.cut_ratio)
        rk.rank(evaluation.erdos_renyi_modularity)
        rk.rank(evaluation.newman_girvan_modularity)
        rk.rank(evaluation.modularity_density)

        rnk = rk.topsis()
        self.assertEquals(len(rnk), 5)

