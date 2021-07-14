from collections import namedtuple

import networkx as nx
from typing import Callable
import cdlib
from cdlib.evaluation.internal.TOPSIS import topsis
from cdlib.evaluation.internal.statistical_ranking import (
    bonferroni_dunn_test,
    friedman_test,
)
import numpy as np

__all__ = ["elem", "FitnessRanking"]

elem = namedtuple("elem", "rk alg param score")
elem.__new__.__defaults__ = (None,) * len(elem._fields)

post_hoc = namedtuple("post_hoc", "comparison z_value p_value adj_p_value")


class FitnessRanking(object):
    def __init__(self, graph: nx.Graph, partitions: list):
        """
        Class that provides functionalities for the generation of Clustering objects w.r.t. a given set of fitness scores.

        :param graph: a Networkx/iGraph object
        :param partitions: a list of Clustering objects
        """
        self.partitions = partitions
        self.graph = graph
        self.rankings = {}
        self.ranks = []
        self.rnk = {}

    def rank(
        self, scoring_function: Callable[[nx.Graph, object], object]
    ) -> [str, dict]:
        """
        Computes the specified scoring function to all the Clustering objects for which a ranking is required.

        :param scoring_function: a fitness function from cdlib.evaluation
        :return: a tuple whose first element is the scoring_function name, while the second is a dictionary having as key the clustering name and as value the computed score.

        :Example:

        >>> import networkx as nx
        >>> from cdlib import evaluation
        >>> from cdlib import algorithms
        >>> g = nx.karate_club_graph()
        >>> coms = algorithms.louvain(g)
        >>> coms2 = algorithms.demon(g, 0.25)
        >>> coms3 = algorithms.label_propagation(g)
        >>> coms4 = algorithms.angel(g, 0.6)
        >>> rk = evaluation.FitnessRanking(g, [coms2, coms, coms3, coms4])
        >>> rk.rank(evaluation.fraction_over_median_degree)
        """

        ranks = {}
        for partition in self.partitions:
            s = scoring_function(self.graph, partition, summary=True)
            ranks[f"{partition.method_name}_{partition.method_parameters}"] = s.score
        s_ranks = sorted(ranks.items(), key=lambda x: -x[1])
        s_ranks = [
            elem(rk=x + 1, alg=c[0].split("_")[0], param=c[0].split("_")[1], score=c[1])
            for x, c in enumerate(s_ranks)
        ]

        res = {f"{i.alg}_{i.param}": (i.rk, i.score) for i in s_ranks}

        self.rankings[scoring_function.__name__] = res

        return scoring_function.__name__, ranks

    def topsis(self) -> [list, None]:
        """
        The Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS) is a multi-criteria decision analysis method.
        TOPSIS is based on the concept that the chosen alternative should have the shortest geometric distance from the positive ideal solution (PIS) and the longest geometric distance from the negative ideal solution (NIS).

        :return: a tuple whose first element is the ranking dictionary assigning a TOPSIS score to each Clustering object, while the second is None (to maintain coherence with friedman_ranking).

        :Example:

        >>> import networkx as nx
        >>> from cdlib import evaluation
        >>> from cdlib import algorithms
        >>> g = nx.karate_club_graph()
        >>> coms = algorithms.louvain(g)
        >>> coms2 = algorithms.demon(g, 0.25)
        >>> coms3 = algorithms.label_propagation(g)
        >>> coms4 = algorithms.angel(g, 0.6)
        >>> rk = evaluation.FitnessRanking(g, [coms2, coms, coms3, coms4])
        >>> rk.rank(evaluation.fraction_over_median_degree)
        >>> rk.rank(evaluation.edges_inside)
        >>> rk.rank(evaluation.cut_ratio)
        >>> rk.rank(evaluation.erdos_renyi_modularity)
        >>> rk.rank(evaluation.newman_girvan_modularity)
        >>> rk.rank(evaluation.modularity_density)
        >>> rnk, _ = rk.topsis()

        :References:

        1. Hwang, C.L.; Yoon, K. (1981). Multiple Attribute Decision Making: Methods and Applications. New York: Springer-Verlag.
        2. Yoon, K. (1987). "A reconciliation among discrete compromise situations". Journal of the Operational Research Society. 38 (3): 277–286. doi:10.1057/jors.1987.44.
        """
        rp = {
            a: [] for score, value in self.rankings.items() for a, vals in value.items()
        }
        for score, value in self.rankings.items():
            for a, vals in value.items():
                rp[a].append(vals[1])
        ranks = np.array([np.array(x) for x in rp.values()])

        agg_rank = topsis(ranks, [1] * len(self.rankings), "v", "m")
        for ids, alg in enumerate(rp):
            rp[alg] = agg_rank[ids]

        s_ranks = sorted(rp.items(), key=lambda x: -x[1])
        self.ranks = [
            elem(rk=x + 1, alg=c[0].split("_")[0], param=c[0].split("_")[1], score=c[1])
            for x, c in enumerate(s_ranks)
        ]

        self.rnk = {f"{i.alg}_{i.param}": i.score for i in self.ranks}

        return self.ranks, None

    def friedman_ranking(self) -> [list, float]:
        """
        Performs a Friedman ranking test.
        Tests the hypothesis that in a set of k dependent samples groups (where k >= 2) at least two of the groups represent populations with different median values.


        :return: a tuple whose first element is a dictionary assigning a rank to each Clustering object, while the second is the p-value associated to the ranking.

        :Example:

        >>> import networkx as nx
        >>> from cdlib import evaluation
        >>> from cdlib import algorithms
        >>> g = nx.karate_club_graph()
        >>> coms = algorithms.louvain(g)
        >>> coms2 = algorithms.demon(g, 0.25)
        >>> coms3 = algorithms.label_propagation(g)
        >>> coms4 = algorithms.angel(g, 0.6)
        >>> rk = evaluation.FitnessRanking(g, [coms2, coms, coms3, coms4])
        >>> rk.rank(evaluation.fraction_over_median_degree)
        >>> rk.rank(evaluation.edges_inside)
        >>> rk.rank(evaluation.cut_ratio)
        >>> rk.rank(evaluation.erdos_renyi_modularity)
        >>> rk.rank(evaluation.newman_girvan_modularity)
        >>> rk.rank(evaluation.modularity_density)
        >>> rnk, p_value = rk.friedman_ranking()

        :References:

        1. M. Friedman, The use of ranks to avoid the assumption of normality implicit in the analysis of variance, Journal of the American Statistical Association 32 (1937) 674–701.
        2. D.J. Sheskin, Handbook of parametric and nonparametric statistical procedures. crc Press, 2003, Test 25: The Friedman Two-Way Analysis of Variance by Ranks
        """

        rp = {
            a: [] for score, value in self.rankings.items() for a, vals in value.items()
        }
        for score, value in self.rankings.items():
            for a, vals in value.items():
                rp[a].append(vals[1])
        ranks = [x for x in rp.values()]

        for ids, alg in enumerate(rp):
            rp[alg] = ranks[ids]

        statistic, p_value, ranking, rank_cmp = friedman_test(*rp.values())

        self.rnk = {key: rank_cmp[i] for i, key in enumerate(rp.keys())}

        s_ranks = sorted(self.rnk.items(), key=lambda x: -x[1])
        self.ranks = [
            elem(rk=x + 1, alg=c[0].split("_")[0], param=c[0].split("_")[1], score=c[1])
            for x, c in enumerate(s_ranks)
        ]

        return self.ranks, p_value

    def bonferroni_post_hoc(self) -> list:
        """
        Performs a Bonferroni-Dunn post-hoc test using the pivot quantities obtained by a ranking test.
        Tests the hypothesis that the ranking of the control method (best ranked clustering) is different to each of the other methods.

        :return: a list of named tuples reporting the pairwise statistical significant comparisons among the best ranked clustering and the others (in terms of z-value, p-value, adjusted-p-value)

        :Example:

        >>> import networkx as nx
        >>> from cdlib import evaluation
        >>> from cdlib import algorithms
        >>> g = nx.karate_club_graph()
        >>> coms = algorithms.louvain(g)
        >>> coms2 = algorithms.demon(g, 0.25)
        >>> coms3 = algorithms.label_propagation(g)
        >>> coms4 = algorithms.angel(g, 0.6)
        >>> rk = evaluation.FitnessRanking(g, [coms2, coms, coms3, coms4])
        >>> rk.rank(evaluation.fraction_over_median_degree)
        >>> rk.rank(evaluation.edges_inside)
        >>> rk.rank(evaluation.cut_ratio)
        >>> rk.rank(evaluation.erdos_renyi_modularity)
        >>> rk.rank(evaluation.newman_girvan_modularity)
        >>> rk.rank(evaluation.modularity_density)
        >>> rnk, p_value = rk.friedman_ranking()
        >>> pc = rk.bonferroni_post_hoc()

        :References:

        O.J. Dunn, Multiple comparisons among means, Journal of the American Statistical Association 56 (1961) 52–64.
        """

        res = []
        comparisons, z_values, p_values, adj_p_values = bonferroni_dunn_test(self.rnk)
        for i in range(0, len(comparisons)):
            res.append(
                post_hoc(
                    comparison=comparisons[i],
                    z_value=z_values[i],
                    p_value=p_values[i],
                    adj_p_value=adj_p_values[i],
                )
            )
        return res
