import networkx as nx
import numpy as np
import scipy


def newman_girvan_modularity(graph, communities):

    m = graph.number_of_edges()
    q = 0

    for community in communities:
        c = nx.subgraph(graph, community)
        mc = c.number_of_edges()

        lc = 0
        for node in c:
            kin = c.degree(node)
            kout = kin - graph.degree(node)
            lc += kout

        q += mc - ((2*mc + lc)**2)/(4*m)

    return (1/m) * q


def erdos_renyi_modularity(graph, communities):

    m = graph.number_of_edges()
    n = graph.number_of_nodes()
    q = 0

    for community in communities:
        c = nx.subgraph(graph, community)
        mc = c.number_of_edges()
        nc = c.number_of_nodes()
        q += mc - (m*nc*(nc - 1)) / (n*(n-1))

    return (1 / m) * q


def modularity_density(graph, communities):

    q = 0

    for community in communities:
        c = nx.subgraph(graph, community)

        nc = c.number_of_nodes()
        dint = []
        dext = []
        for node in c:
            dint.append(c.degree(node))
            dext.append(c.degree(node) - graph.degree(node))

        q += (1 / nc) * (np.mean(dint) - np.mean(dext))

    return q


def z_modularity(graph, communities):

    m = graph.number_of_edges()

    mmc = 0
    dc2m = 0

    for community in communities:
        c = nx.subgraph(graph, community)
        mc = c.number_of_edges()
        dc = 0

        for node in c:
            dc += graph.degree(node)

        mmc += (mc/m)
        dc2m += (dc/2*m)**2

    return (mmc - dc2m) * (dc2m * (1 - dc2m))**(-1/2)


def surprise(graph, communities):

    m = graph.number_of_edges()

    M = scipy.special.comb(m, 2, exact=True)
    mint = 0
    Mint = 0

    for community in communities:
        c = nx.subgraph(graph, community)
        kinc = 0
        for node in c:
            kinc += c.degree(node)
            nc = c.number_of_nodes()
            Mint += scipy.special.comb(nc, 2, exact=True)

        mint += (kinc/2)

    q = mint/m
    qa = Mint/M

    sp = m * (q*np.log((q/qa)) + (1-q)*np.log((1-q)/(1-qa)))
    return sp


def significance(graph, communities):

    m = graph.number_of_edges()

    M = scipy.special.comb(m, 2, exact=True)
    p = m/M

    z = 0

    for community in communities:
        c = nx.subgraph(graph, community)
        nc = c.number_of_nodes()
        mc = c.number_of_edges()

        Min = scipy.special.comb(nc, 2, exact=True)
        pc = mc / Min

        z += (pc*np.log((pc/p)) + (1-pc)*np.log((1-pc)/(1-p)))

    return z

