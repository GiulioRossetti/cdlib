import networkx as nx
import timeit
import numpy as np


def __HeadTailCommunityDetection(G, finaledgelist, head_tail_ratio=0.6):

    H = nx.connected_components(G)

    for s in H:
        subgraph = nx.subgraph(G, s)
        result = nx.edge_betweenness(subgraph, normalized=False)
        edges = list(result.keys())
        values = list(result.values())
        mean = np.mean(values)
        edgelist = []
        edgetemp = subgraph.edges()
        if len(edgetemp) <= 2:
            for edge in edgetemp:
                finaledgelist.append(edge)
        else:
            for index in range(len(values)):
                if values[index] <= mean:
                    edgelist.append(edges[index])

            if (
                float(len(edgelist)) / float(len(edges)) <= head_tail_ratio
            ):  # change the head/tail division rule here, here is for tail percentage,
                # so if the rule is 40/60, the value should be assigned 0.6 as in the code.
                for edge in edgelist:
                    finaledgelist.append(edge)
            else:
                Gsub = nx.Graph()
                for edge in edgelist:
                    Gsub.add_edge(edge[0], edge[1])
                try:
                    __HeadTailCommunityDetection(Gsub, finaledgelist, head_tail_ratio)
                except:
                    pass
    return finaledgelist


def HeadTail(G):
    finaledgelist = __HeadTailCommunityDetection(G, [], head_tail_ratio=0.4)
    g1 = nx.Graph()
    g1.add_edges_from(finaledgelist)
    coms = [list(c) for c in nx.connected_components(g1)]
    return coms
