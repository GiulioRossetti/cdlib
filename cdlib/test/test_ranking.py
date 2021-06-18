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

        rk = evaluation.FitnessRanking(g, [coms2, coms, coms3, coms4])

        rk.rank(evaluation.fraction_over_median_degree)
        rk.rank(evaluation.edges_inside)
        rk.rank(evaluation.cut_ratio)
        rk.rank(evaluation.erdos_renyi_modularity)
        rk.rank(evaluation.newman_girvan_modularity)
        rk.rank(evaluation.modularity_density)

        rnk, _ = rk.topsis()
        self.assertEqual(len(rnk), 4)

        pc = rk.bonferroni_post_hoc()
        self.assertLessEqual(len(pc), 4)

    def test_ranking_significance(self):
        g = nx.karate_club_graph()
        coms = algorithms.louvain(g)
        coms2 = algorithms.demon(g, 0.25)
        coms3 = algorithms.label_propagation(g)
        coms4 = algorithms.angel(g, 0.6)

        rk = evaluation.FitnessRanking(g, [coms2, coms, coms3, coms4])

        rk.rank(evaluation.fraction_over_median_degree)
        rk.rank(evaluation.edges_inside)
        rk.rank(evaluation.cut_ratio)
        rk.rank(evaluation.erdos_renyi_modularity)
        rk.rank(evaluation.newman_girvan_modularity)
        rk.rank(evaluation.modularity_density)

        rnk, p_value = rk.friedman_ranking()
        self.assertEqual(len(rnk), 4)
        self.assertLessEqual(p_value, 1)

        pc = rk.bonferroni_post_hoc()
        self.assertLessEqual(len(pc), 4)

    def test_ranking_comp(self):
        g = nx.karate_club_graph()
        coms = algorithms.louvain(g)
        coms2 = algorithms.kclique(g, 2)
        coms3 = algorithms.label_propagation(g)

        rk = evaluation.ComparisonRanking([coms, coms2, coms3])

        rk.rank(evaluation.overlapping_normalized_mutual_information_LFK)
        rk.rank(evaluation.overlapping_normalized_mutual_information_MGH)
        rk.rank(evaluation.omega)

        rnk, _ = rk.topsis()
        self.assertEqual(len(rnk), 3)

        pc = rk.bonferroni_post_hoc()
        self.assertLessEqual(len(pc), 4)

    def test_ranking_significance_comp(self):
        g = nx.karate_club_graph()
        coms = algorithms.louvain(g)
        coms2 = algorithms.kclique(g, 2)
        coms3 = algorithms.label_propagation(g)

        rk = evaluation.ComparisonRanking([coms, coms2, coms3])

        rk.rank(evaluation.overlapping_normalized_mutual_information_LFK)
        rk.rank(evaluation.overlapping_normalized_mutual_information_MGH)
        rk.rank(evaluation.omega)

        rnk, p_value = rk.friedman_ranking()
        self.assertEqual(len(rnk), 3)
        self.assertLessEqual(p_value, 1)

        pc = rk.bonferroni_post_hoc()
        self.assertLessEqual(len(pc), 4)
