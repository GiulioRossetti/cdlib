from collections import namedtuple

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
            ranks[f"{partition.method_name}_{partition.method_parameters}"] = s.mean
        s_ranks = sorted(ranks.items(), key=lambda x: -x[1])
        s_ranks = [elem(rk=x+1, alg=c[0].split("_")[0], param=c[0].split("_")[1], score=c[1]) for x, c in enumerate(s_ranks)]
        return scoring_function.__name__, s_ranks
