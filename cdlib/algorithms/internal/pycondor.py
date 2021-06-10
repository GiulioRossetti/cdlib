import numpy as np
import pandas as pd
from igraph import *


def condor_object(net):
    """Initialization of the condor object. The function gets a network in edgelist format encoded in a pandas
    dataframe. Returns a dictionary with an igraph network, names of the targets and regulators, list of edges,
    modularity, and vertex memberships.
    """

    # Error flags.
    assert (
        len(set(net.iloc[:, 0]).intersection(net.iloc[:, 1])) == 0
    ), "The network must be bipartite."
    assert not net.isnull().any().any(), "NaN values detected."
    assert not (
        "" in list(net.iloc[:, 0]) or "" in list(net.iloc[:, 1])
    ), "Empty strings detected."

    # Builds graph object.
    if net.shape[1] == 3:
        edges = list(zip(net.iloc[:, 0], net.iloc[:, 1], net.iloc[:, 2]))
        Gr = Graph.TupleList(edges, weights=True)
    else:
        edges = list(zip(net.iloc[:, 0], net.iloc[:, 1], [1 for _ in net.iloc[:, 1]]))
        Gr = Graph.TupleList(edges, weights=True)

    # Assigns color names (bipartite sets).
    reg_names = sorted(set(net.iloc[:, 1]))
    tar_names = sorted(set(net.iloc[:, 0]))
    Gr.vs["type"] = 0  # Tar
    for j in [i.index for i in Gr.vs if i["name"] in reg_names]:
        Gr.vs[j]["type"] = 1  # Reg

    index_dict = {k.index: k["name"] for k in Gr.vs}

    return {
        "G": Gr,
        "tar_names": tar_names,
        "reg_names": reg_names,
        "index_dict": index_dict,
        "edges": edges,
        "modularity": None,
        "reg_memb": None,
        "Qcoms": None,
    }


def bipartite_modularity(B, m, R, T, CO):
    """Computation of the bipartite modularity as described in ""Modularity and community detection in bipartite
    networks" by Michael J. Barber." """
    RtBT = R.transpose().dot(B.dot(T))
    Qcoms = (1 / m) * (np.diagonal(RtBT))
    Q = sum(Qcoms)
    Qcoms = Qcoms[Qcoms > 0]
    CO["Qcoms"] = Qcoms
    return Q, CO


def initial_community(CO, method="LCS"):
    """Computation of the initial community structure based on unipartite methods.
    The implementation using bipartite projection is not yet available, but project=False
    performs better modularity-wise (at least with the networks I worked on).
    """

    # Computes initial community assignement based on different methods. Default method is Louvain clustering
    # which is really fast. The others take several times more.
    vc = None
    if method == "LCS":
        vc = Graph.community_multilevel(CO["G"], weights="weight")
    if method == "LEC":
        vc = Graph.community_leading_eigenvector(CO["G"], weights="weight")
    if method == "FG":
        vc = Graph.community_fastgreedy(CO["G"], weights="weight").as_clustering()

    CO["modularity"] = vc.modularity

    # Stores initial community in the condor object.
    reg_index = [i.index for i in CO["G"].vs.select(type_in=[1])]
    reg_memb = [vc.membership[i] for i in reg_index]
    T0 = pd.DataFrame(zip(reg_index, reg_memb))
    T0.columns = ["index", "community"]
    CO["reg_memb"] = T0

    return CO


def brim(CO, deltaQmin="def", c=25):
    """Implementation of the BRIM algorithm to iteratively maximize bipartite modularity.
    Note that c is the maximum number of communities. Dynamic choice of c is not yet implemented.
    """

    # Gets modularity matrix, initial community matrix and index dictionary.
    B, m, T0, R0, gn, rg = matrices(CO, c)

    # Default deltaQmin.
    if deltaQmin == "def":
        deltaQmin = min(1 / len(CO["edges"]), 1e-5)

    # BRIM iterative process
    Qnow = 0
    deltaQ = 1
    p, q = B.shape
    R, T = None, None
    while deltaQ > deltaQmin:
        # Right sweep
        Tp = B.dot(T0)
        R = np.zeros((p, c))
        am = np.array(np.argmax(Tp, axis=1))
        for i in range(0, len(am)):
            R[i, am[i][0]] = 1
        # Left sweep
        Rp = B.transpose().dot(R)
        T = np.zeros((q, c))
        am = np.array(np.argmax(Rp, axis=1))
        for i in range(0, len(am)):
            T[i, am[i][0]] = 1
        T0 = T

        Qthen = Qnow
        Qnow, CO = bipartite_modularity(B, m, R, T, CO)
        deltaQ = Qnow - Qthen
        # print(Qnow)

    # Update modularity attribute
    CO["modularity"] = Qnow

    # Update membership dataframes.
    CO["tar_memb"] = pd.DataFrame(
        list(zip(list(gn), [R[i, :].argmax() for i in range(0, len(gn))]))
    )
    CO["reg_memb"] = pd.DataFrame(
        list(zip(list(rg), [T[i, :].argmax() for i in range(0, len(rg))]))
    )
    CO["tar_memb"].columns = ["tar", "com"]
    CO["reg_memb"].columns = ["reg", "com"]

    return CO


def matrices(CO, c):
    """Computation of modularity matrix and initial community matrix."""

    # Dimensions of the matrix
    p = len(CO["tar_names"])
    q = len(CO["reg_names"])

    # Index dictionaries for the matrix. Note that this set of indices is different of that in
    # the CO object (that one is for the igraph network.)
    rg = {CO["reg_names"][i]: i for i in range(0, q)}
    gn = {CO["tar_names"][i]: i for i in range(0, p)}

    # Computes bipartite adjacency matrix.
    A = np.matrix(np.zeros((p, q)))
    for edge in CO["edges"]:
        A[gn[edge[0]], rg[edge[1]]] = edge[2]

    # Computes bipartite modularity matrix.
    ki = A.sum(1)
    dj = A.sum(0)
    m = float(sum(ki))
    B = A - ((ki @ dj) / m)

    # Creates initial community T0 matrix.
    d = CO["index_dict"]
    if "index" in CO["reg_memb"].columns:
        ed = zip(
            [rg[j] for j in [d[i] for i in CO["reg_memb"].iloc[:, 0]]],
            CO["reg_memb"].iloc[:, 1],
        )
    else:
        ed = zip([rg[j] for j in CO["reg_memb"].iloc[:, 0]], CO["reg_memb"].iloc[:, 1])
    T0 = np.zeros((q, c))
    for edge in ed:
        T0[edge] = 1

    if "tar_memb" not in CO:
        # print("Matrices computed in", time.time() - t)
        return B, m, T0, 0, gn, rg
    ed = zip([gn[j] for j in CO["tar_memb"].iloc[:, 0]], CO["tar_memb"].iloc[:, 1])
    R0 = np.zeros((p, c))
    for edge in ed:
        R0[edge] = 1
    # print("Matrices computed in", time.time() - t)
    return B, m, T0, R0, gn, rg


def qscores(CO):
    """Computes the qscores (contribution of a vertex to its community modularity)
    for each vertex in the network."""

    B, m, T, R, gn, rg = matrices(CO, 6)
    CO["Qscores"] = {"reg_qscores": None, "tar_qscores": None}

    # Qscores for the regulators:
    Rq = R.transpose().dot(B) / (2 * m)
    Qj = list()
    for j, r in CO["reg_memb"].iterrows():
        Qjh = Rq[r["com"], j] / CO["Qcoms"][r["com"]]
        Qj.append(Qjh)
    CO["Qscores"]["reg_qscores"] = CO["reg_memb"].copy()
    CO["Qscores"]["reg_qscores"]["qscore"] = Qj

    # Qscores for the targets:
    Tq = B.dot(T) / (2 * m)
    Qi = list()
    for i, r in CO["tar_memb"].iterrows():
        Qih = Tq[i, r["com"]] / CO["Qcoms"][r["com"]]
        Qi.append(Qih)
    CO["Qscores"]["tar_qscores"] = CO["tar_memb"].copy()
    CO["Qscores"]["tar_qscores"]["qscore"] = Qi

    return CO
