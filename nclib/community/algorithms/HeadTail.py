import networkx as nx


"""
Jiang B. and Ding M. (2015), 
Defining least community as a homogeneous group in complex networks, 
Physica A, 428, 154-160.


https://github.com/dingmartin/HeadTailCommunityDetection
"""



def getMean(values):
    mean=0
    numsum=0

    for index in range(len(values)):
        numsum =numsum + list(values)[index]
    mean=float(numsum)/float(len(values))
    return mean


def HeadTailCommunityDetection(graph, finaledgelist):
    """

    :param graph:
    :param finaledgelist:
    """
    H = nx.connected_component_subgraphs(graph)

    for subgraph in H:
        result=nx.edge_betweenness(subgraph, False, None)
        edges=result.keys()
        values=result.values()
        mean = getMean(values)
        edgelist=[]
        edgetemp=subgraph.edges()

        if len(edgetemp) <= 2:
            for edge in edgetemp:
                finaledgelist.append(edge)
        else:
            for index in range(len(values)):
                    if list(values)[index] <= mean:
                        edgelist.append(list(edges)[index])
            if (float(len(edgelist))/float(len(edges))) <= 0.6:  # change the head/tail division rule here, here is for tail percentage, so if the rule is 40/60, the value should be assigned 0.6 as in the code.
                for edge in edgelist:
                    finaledgelist.append(edge)
            else:
                Gsub = nx.Graph()
                for edge in edgelist:
                    Gsub.add_edge(edge[0], edge[1])
                HeadTailCommunityDetection(Gsub, finaledgelist)

    #return finaledgelist # I assume...


def HeadTailInitiator(graph):
    finaledgelist=[]
    HeadTailCommunityDetection(graph, finaledgelist)
    print(finaledgelist)

HeadTailInitiator(nx.karate_club_graph())

