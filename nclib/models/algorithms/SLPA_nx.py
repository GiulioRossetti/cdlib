import numpy as np
from collections import defaultdict
import time
import sys
import networkx as nx

def timeit(method):
    """
    Decorator: Compute the execution time of a function
    :param method: the function
    :return: the method runtime
    """

    def timed(*arguments, **kw):
        ts = time.time()
        result = method(*arguments, **kw)
        te = time.time()

        sys.stdout.write('Time:  %r %2.2f sec\n' % (method.__name__.strip("_"), te - ts))
        sys.stdout.write('------------------------------------\n')
        sys.stdout.flush()
        return result

    return timed


class SLPA(object):

    def __init__(self, network_filename, T=21, r=0.1, out_file=""):
        self.network_filename = network_filename
        self.__read_graph(network_filename)
        self.T = T
        self.r = r
        self.outfile_name = out_file

    @timeit
    def __read_graph(self, network_filename):
        """
        Read .ncol network file

        :param network_filename: complete path for the .ncol file
        :return: an undirected igraph network
        """
        self.G = nx.read_edgelist(network_filename, nodetype=int)

    @timeit
    def execute(self):
        """
        Speaker-Listener Label Propagation Algorithm (SLPA)
        see http://arxiv.org/abs/1109.5720
        """

        # Stage 1: Initialization
        memory = {i: {i: 1} for i in self.G.nodes()}

        # Stage 2: Evolution
        for t in range(self.T):

            listenersOrder = list(self.G.nodes())
            np.random.shuffle(listenersOrder)

            for listener in listenersOrder:
                speakers = self.G[listener].keys()
                if len(speakers) == 0:
                    continue

                labels = defaultdict(int)

                for j, speaker in enumerate(speakers):
                    # Speaker Rule
                    total = float(sum(memory[speaker].values()))
                    labels[memory[speaker].keys()[
                        np.random.multinomial(1, [freq / total for freq in memory[speaker].values()]).argmax()]] += 1

                # Listener Rule
                acceptedLabel = max(labels, key=labels.get)

                # Update listener memory
                if acceptedLabel in memory[listener]:
                    memory[listener][acceptedLabel] += 1
                else:
                    memory[listener][acceptedLabel] = 1

        #  Stage 3:
        for node, mem in memory.items():
            for label, freq in mem.items():
                if freq / float(self.T + 1) < self.r:
                    del mem[label]

        # Find nodes membership
        communities = {}
        for node, mem in memory.items():
            for label in mem.keys():
                if label in communities:
                    communities[label].add(node)
                else:
                    communities[label] = set([node])

        # Remove nested communities
        nestedCommunities = set()
        keys = communities.keys()
        for i, label0 in enumerate(keys[:-1]):
            comm0 = communities[label0]
            for label1 in keys[i + 1:]:
                comm1 = communities[label1]
                if comm0.issubset(comm1):
                    nestedCommunities.add(label0)
                elif comm0.issuperset(comm1):
                    nestedCommunities.add(label1)

        for comm in nestedCommunities:
            del communities[comm]

        #return communities

        if self.outfile_name is not False:
            out_file_com = open(self.outfile_name, "w")
            idc = 0
            for c in communities.values():
                out_file_com.write("%d\t%s\n" % (idc, str(sorted(c))))
                idc += 1
            out_file_com.flush()
            out_file_com.close()
        else:
            return communities


if __name__ == "__main__":
    import argparse

    print "------------------------------------"
    print "               SLPA                 "
    print "------------------------------------"

    parser = argparse.ArgumentParser()

    parser.add_argument('network_file', type=str, help='network file (edge list format)')
    parser.add_argument('T', type=int, help='merging threshold')
    parser.add_argument('r', type=float, help='minimum community size', default=0.1)
    parser.add_argument('-o', '--out_file', type=str, help='output file', default="slpa_coms.txt")

    args = parser.parse_args()

    d = SLPA(args.network_file, T=args.T, r=args.r, out_file=args.out_file)
    d.execute()
    sys.stdout.flush()
