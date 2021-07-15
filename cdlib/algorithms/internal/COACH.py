# Author: True Price <jtprice@cs.unc.edu>

# A core-attachment based method to detect protein complexes in PPI networks
# Wu, Li, Kwoh, Ng (2009)
# http://www.biomedcentral.com/1471-2105/10/169

from collections import defaultdict
from itertools import combinations
import functools


# return average degree and density for a graph
def __graph_stats(graph):
    avg_deg = sum(len(n) for n in graph.values()) / float(len(graph))
    density = avg_deg / (len(graph) - 1)
    return avg_deg, density


# return core nodes, given a graph and its average degree
__get_core_nodes = lambda g, avg: set(v for v, n in g.items() if len(n) >= avg)

# return NA score
__NA_score = lambda a, b: float(len(a & b) ** 2) / (len(a) * len(b))


def __core_removal(graph, density_threshold):
    if len(graph) == 1:  # need at least two nodes in the graph...
        return [graph]

    avg_deg, density = __graph_stats(graph)

    if density >= density_threshold:
        return [graph]
    else:
        # find and remove core nodes; create connected subcomponents
        core_nodes = __get_core_nodes(graph, avg_deg)
        result = []
        subgraphs = []
        for v, n in graph.items():
            if v in core_nodes:
                continue
            n = n - core_nodes  # note that we're reassigning n
            for s in subgraphs:
                if not n.isdisjoint(s):
                    s |= n
                    break
            else:
                subgraphs.append(n | {v})
        # connected subcomponent joining
        i = 0
        while i < len(subgraphs) - 1:
            j = i + 1
            while j < len(subgraphs):
                if not subgraphs[i].isdisjoint(subgraphs[j]):
                    subgraphs[i] |= subgraphs[j]
                    subgraphs.pop(j)
                else:
                    j += 1
            i += 1
        # recursive core removal
        for s in subgraphs:
            tresults = __core_removal(
                dict((v, graph[v] & s) for v in s), density_threshold
            )
            for tc in tresults:
                nodes = set()
                for v, n in tc.items():
                    nodes.add(v)
                    n |= graph[v] & core_nodes
                for c in core_nodes:
                    tc[c] = graph[c] & (nodes | core_nodes)
            result += tresults
        return result


def co_ach(g, density_threshold=0.7, affinity_threshold=0.225, closeness_threshold=0.5):
    # read protein-protein pairs
    data = defaultdict(set)
    for a, b in g.edges():
        data[a].add(b)
        data[b].add(a)

    # step 1: find preliminary cores
    SC = []  # currently-detected preliminary cores
    count = 0
    for vertex, neighbors in data.items():
        # build neighborhood graph
        vertices = {vertex} | neighbors
        size1_neighbors = set()
        graph = {}
        for v in vertices:
            n = data[v] & vertices
            if len(n) > 1:  # ignore size-1 vertices
                graph[v] = n
            else:
                size1_neighbors.add(v)
        if len(graph) < 2:  # not enough connections in this graph
            continue
        graph[vertex] -= size1_neighbors

        # get core graph
        avg_deg, density = __graph_stats(graph)
        core_nodes = __get_core_nodes(graph, avg_deg)
        vertices = set(graph.keys())
        for v in vertices - core_nodes:
            del graph[v]
        for n in graph.values():
            n &= core_nodes
        if len(graph) < 2:  # not enough connections in this graph
            continue
        graph_nodes = set(graph)

        # inner loop
        for sg in __core_removal(graph, density_threshold):
            while True:
                _, density = __graph_stats(sg)
                # if density threshold met, stop; else, remove min degree node
                if density >= density_threshold:
                    break
                w = min(sg.items(), key=lambda k: len(k[1]))[0]
                del sg[w]
                for n in sg.values():
                    n.discard(w)

            sg_nodes = set(sg)
            while graph_nodes - sg_nodes:
                w = max(graph_nodes - sg_nodes, key=lambda v: len(graph[v] & sg_nodes))
                new_sg = sg.copy()
                for v, n in new_sg.items():
                    if w in graph[v]:
                        n.add(w)
                new_sg[w] = graph[w] & sg_nodes
                _, density = __graph_stats(new_sg)
                if density < density_threshold:
                    break
                sg = new_sg
                sg_nodes.add(w)

            # redundancy filtering
            max_sim = -1
            for i in range(len(SC)):
                sim = __NA_score(set(SC[i]), sg_nodes)
                if sim > max_sim:
                    max_sim = sim
                    index = i
            if max_sim < affinity_threshold:
                SC.append(sg)
            else:
                _, density_i = __graph_stats(SC[index])
                if density * len(sg) > density_i * len(SC[index]):
                    SC[index] = sg

    # step 2: adding peripheral proteins
    clusters = set()
    for core in SC:
        nodes = frozenset(core)
        neighbors = (
            functools.reduce(lambda x, y: x | y, (data[v] for v in nodes)) - nodes
        )
        neighbors -= set(
            v
            for v in neighbors
            if float(len(data[v] & nodes)) / len(nodes) <= closeness_threshold
        )
        clusters.add(nodes | neighbors)

    return [list(c) for c in clusters]
