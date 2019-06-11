from collections import namedtuple
from cdlib.evaluation.internal.TOPSIS import topsis
import numpy as np

# elem = namedtuple('elem', ['rk', 'alg', 'param', 'score'])

elem = namedtuple('elem', 'rk alg param score')
elem.__new__.__defaults__ = (None,) * len(elem._fields)


class Ranking(object):

    def __init__(self, graph, partitions):
        self.partitions = partitions
        self.graph = graph
        self.rankings = {}

    def rank(self, scoring_function):

        ranks = {}
        for partition in self.partitions:
            s = scoring_function(self.graph, partition, summary=True)
            ranks[f"{partition.method_name}_{partition.method_parameters}"] = s.score
        s_ranks = sorted(ranks.items(), key=lambda x: -x[1])
        s_ranks = [elem(rk=x+1, alg=c[0].split("_")[0], param=c[0].split("_")[1], score=c[1]) for x, c in enumerate(s_ranks)]

        res = {f"{i.alg}_{i.param}": (i.rk, i.score) for i in s_ranks}

        self.rankings[scoring_function.__name__] = res

        return scoring_function.__name__, ranks

    def topsis(self):
        rp = {a: [] for score, value in self.rankings.items() for a, vals in value.items()}
        for score, value in self.rankings.items():
            for a, vals in value.items():
                rp[a].append(vals[1])
        ranks = np.array([np.array(x) for x in rp.values()])

        agg_rank = topsis(ranks, [1]*len(self.rankings), 'v', 'm')
        for ids, alg in enumerate(rp):
            rp[alg] = agg_rank[ids]
        return rp
