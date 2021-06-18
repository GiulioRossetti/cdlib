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

        :param graph:
        :param partitions:
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

        :param scoring_function:
        :return:
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

        :return:
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

        :return:
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

        :return:
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
