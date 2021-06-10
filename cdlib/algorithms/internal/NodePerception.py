import os
import sys
import time
import networkx as nx
import community
import shutil

__author1__ = "Sucheta Soundarajan"
__author2__ = "John Hopcroft"
__contact__ = "susounda@syr.edu"


def timeit(method):
    """
    Decorator: Compute the execution time of a function
    :param method: the function
    :return: the method runtime in seconds
    """

    def timed(*arguments, **kw):
        ts = time.time()
        result = method(*arguments, **kw)
        te = time.time()
        sys.stdout.write("------------------------------------\n")
        sys.stdout.write(
            "Time:  %r %2.2f sec\n" % (method.__name__.strip("_"), te - ts)
        )
        sys.stdout.write("------------------------------------\n")
        sys.stdout.flush()
        return result

    return timed


class NodePerception(object):
    """
    Node Perception

    Sucheta Soundarajan and John Hopcroft.
    Use of Local Group Information to Identify Communities in Networks.
    ACM Transactions on Knowledge Discovery from Data (TKDD). 2015.

    http://www.soundarajan.org/papers/TKDDCommunities.pdf

    """

    def __init__(
        self, g, sim_threshold=0.2, overlap_threshold=1, min_comm_size=2, out_file=""
    ):
        """
        Constructor

        :param network_filename: the .ncol network file
        :param sim_threshold: the tolerance required in order to merge communities
        :param overlap_threshold: the overlap tolerance
        :param min_comm_size: minimum algorithms size
        :param out_file: desired output file name
        """
        self.g = g
        self.out_file = out_file
        self.sim_threshold = sim_threshold
        self.min_comm_size = min_comm_size
        self.overlap_threshold = overlap_threshold

        if not os.path.exists("tmp"):
            os.makedirs("tmp")

    def __GetEdgesIntoDict(self):
        edges = {}
        for t in self.g.edges():
            if len(t) > 0:
                if t[0] != t[1]:
                    if t[0] not in edges:
                        edges[t[0]] = {t[1]}
                    else:
                        edges[t[0]].add(t[1])
                    if t[1] not in edges:
                        edges[t[1]] = {t[0]}
                    else:
                        edges[t[1]].add(t[0])
        return edges

    def __FirstPartition(self, edges, first_part_file):
        OUT = open(first_part_file, "w")
        node_count = 0
        for n in edges:
            node_count = node_count + 1
            if node_count > 0:
                # the nodes
                index = {}
                reverse_index = {}
                count = 0
                to_add_edges = []
                adj = set([])
                for neighbor in edges[n]:
                    index[count] = neighbor
                    reverse_index[neighbor] = count
                    adj.add(neighbor)
                    count = count + 1
                for m in reverse_index:
                    for k in edges[m]:
                        if k in reverse_index and reverse_index[k] < reverse_index[m]:
                            to_add_edges.append((reverse_index[m], reverse_index[k]))
                G = nx.Graph()
                G.add_nodes_from([i for i in index])
                G.add_edges_from(to_add_edges)
                if len(to_add_edges) > 0:
                    dict_H = community.best_partition(G)
                    H = {}
                    for node in dict_H:
                        if dict_H[node] not in H:
                            H[dict_H[node]] = set([])
                        H[dict_H[node]].add(node)

                    for i in H:
                        comm = H[i]
                        if len(comm) > 0:
                            for c in comm:
                                OUT.write(str(index[int(c)]) + " ")
                            OUT.write(str(n))
                            OUT.write("\n")
                        elif len(comm) > 0:
                            for c in comm:
                                if index[int(c)] in edges[n]:
                                    OUT.write(str(index[int(c)]) + " ")
                            OUT.write(str(n))
                            OUT.write("\n")
        OUT.close()

    def __GetMembership(self, first_part_file, membership_file):
        node_membership = {}

        IN = open(first_part_file, "rb")
        read_line = IN.readline()
        count = 0
        while read_line:
            t = read_line.rstrip().split()
            if len(t) >= self.min_comm_size:
                for mem in t:
                    if mem not in node_membership:
                        node_membership[mem] = set([])
                    node_membership[mem].add(count)
            count = count + 1
            read_line = IN.readline()

        IN.close()

        OUT = open(membership_file, "w")
        for n in node_membership:
            in_comms = node_membership[n]
            OUT.write(str(n) + " ")
            for c in in_comms:
                OUT.write(str(c) + " ")
            OUT.write("\n")
        OUT.close()

    def __Jaccard(self, set1, set2):
        set1 = set(set1)
        set2 = set(set2)
        return float(len(set1.intersection(set2))) / float(len(set1.union(set2)))

    def __GetCommSimilarities(self, first_part_file, membership_file, sim_file):

        IN = open(first_part_file, "rb")
        line_offset = {}
        offset = 0
        count = 0
        for line in IN:
            line_offset[count] = offset
            count = count + 1
            offset += len(line)
        IN.close()
        num_lines = count - 1

        IN = open(membership_file, "r")
        node_membership = {}
        read_line = IN.readline()
        while read_line:
            t = read_line.rstrip().split()

            if len(t) > 0:
                node = t[0]
                t.remove(node)
                if node not in node_membership:
                    node_membership[node] = set([])
                for c in t:
                    node_membership[node].add(c)
            read_line = IN.readline()

        IN.close()
        IN = open(first_part_file, "rb")
        OUT = open(sim_file, "w")
        curr_count = 0
        total_lines = 0

        while curr_count < num_lines:
            IN.seek(line_offset[curr_count])
            read_line = IN.readline()
            t = read_line.rstrip().split()
            if len(t) >= self.min_comm_size:
                adj = set([])
                for mem in t:
                    adj = adj.union(set(node_membership[str(mem)]))
                for adj_comm in adj:
                    if int(adj_comm) > int(curr_count):
                        IN.seek(line_offset[int(adj_comm)])
                        read_line = IN.readline()
                        r = read_line.rstrip().split()
                        if len(r) >= self.min_comm_size:
                            sim = self.__Jaccard(t, r)
                            if r[len(r) - 1] == t[len(t) - 1]:
                                sim = 0
                            if sim > self.sim_threshold:
                                total_lines = total_lines + 1
                                OUT.write(
                                    str(curr_count)
                                    + " "
                                    + str(adj_comm)
                                    + " "
                                    + str(sim)
                                    + " "
                                    + str(len(set(t).intersection(set(r))))
                                    + " "
                                    + str(len(t))
                                    + " "
                                    + str(len(r))
                                )
                                OUT.write("\n")
            curr_count = curr_count + 1
        IN.close()
        OUT.close()

    def __SecondPartition(self, overlap_threshold, sim_file, first_part_file):
        return_vals = self.__ModClusteringSingleBig(
            sim_file, first_part_file, overlap_threshold
        )
        return return_vals

    def __ModClusteringSingleBig(self, sim_file, first_part_file, overlap_threshold=0):
        IN = open(sim_file, "r")
        read_line = IN.readline()
        num_lines = 0
        comm_edges = {}
        to_add_edges = []
        while read_line:
            t = read_line.rstrip().split()
            num_lines += 1
            if len(t) > 0:
                node1 = int(t[0])
                node2 = int(t[1])
                sim = t[2]
                overlap = t[3]
                if (
                    float(sim) >= self.sim_threshold
                    and float(overlap) > overlap_threshold
                ):
                    if node1 not in comm_edges:
                        comm_edges[node1] = set([])
                    if node2 not in comm_edges:
                        comm_edges[node2] = set([])
                    if node2 not in comm_edges[node1]:
                        comm_edges[node1].add(node2)
                        comm_edges[node2].add(node1)
                        weight = sim
                        to_add_edges.append((node1, node2, {"weight": float(weight)}))

            read_line = IN.readline()
        IN.close()
        G = nx.Graph()
        G.add_nodes_from(range(len(comm_edges)))
        G.add_edges_from(to_add_edges)
        dict_H = community.best_partition(G)
        H1 = {}
        for e in dict_H:
            if dict_H[e] not in H1:
                H1[dict_H[e]] = set([])
            H1[dict_H[e]].add(e)
        H = []
        for e in H1:
            H.append(H1[e])

        IN = open(first_part_file, "rb")
        line_offset = {}
        offset = 0
        count = 0
        for line in IN:
            line_offset[count] = offset
            count = count + 1
            offset += len(line)
        IN.close()

        IN = open(first_part_file, "rb")
        all_comms = {}
        i = 0
        for big_comm in H:
            comm_members = {}
            for comm in big_comm:
                IN.seek(line_offset[int(comm)])
                read_line = IN.readline()
                t = read_line.rstrip().split()
                if len(t) > 0:
                    for t1 in t:
                        if t1 not in comm_members:
                            comm_members[t1] = 0
                        comm_members[t1] += 1
            all_comms[i] = set([])
            for t1 in comm_members:
                if comm_members[t1] >= 0:
                    all_comms[i].add(t1)

            i += 1

        return all_comms

    def __GetModComms(self, G):
        dict_H = community.best_partition(G)
        H1 = {}
        for e in dict_H:
            if dict_H[e] not in H1:
                H1[dict_H[e]] = set([])
            H1[dict_H[e]].add(e)
        H = []
        for e in H1:
            H.append(H1[e])
        return H

    def __CleanComms(self, to_clean):
        comms = {}

        count = 0
        idx = {}
        for t in to_clean.values():
            if len(t) > 0:
                comms[count] = set(t)
                for i in t:
                    if i not in idx:
                        idx[i] = set([])

                    idx[i].add(count)
                count += 1
            elif len(t) > 0:
                comms[count] = set(t)
                count += 1

        coms = []
        for i in range(count):
            C = comms[i]
            if len(C) > 0:
                poss = set([])
                found = 0
                for n in C:
                    poss = poss.union(idx[n])
                for j in poss:
                    if j < i:
                        if (
                            len(comms[j]) == len(comms[i])
                            and len(comms[j].difference(comms[i])) == 0
                        ):
                            found = 1
                if found != 1:
                    coms.append([t.decode("utf-8") for t in C])
            else:
                coms.append([t.decode("utf-8") for t in C])
        return coms

    def execute(self):

        first_part_file = "tmp/part1_"
        membership_file = "tmp/membership_"
        sim_file = "tmp/simfile_"

        # Part 1
        # read the edge file
        edges = self.__GetEdgesIntoDict()

        sys.stdout.write("First Partition\n")
        # get the small communities and write them to a file
        self.__FirstPartition(edges, first_part_file)

        self.__GetMembership(first_part_file, membership_file)
        # calculate similarity between each pair of small communities; minimum threshold is set to jaccard of 0.2
        self.__GetCommSimilarities(first_part_file, membership_file, sim_file)

        # Part 2
        sys.stdout.write("Second Partition\n")
        return_vals = self.__SecondPartition(0, sim_file, first_part_file)
        coms = self.__CleanComms(return_vals)

        shutil.rmtree("tmp")
        return coms
