from cdlib import BiNodeClustering

import networkx as nx

try:
    import infomap as imp
except ModuleNotFoundError:
    imp = None

try:
    from wurlitzer import pipes
except ModuleNotFoundError:
    pipes = None

try:
    import igraph as ig
except ModuleNotFoundError:
    ig = None

try:
    import leidenalg
except ModuleNotFoundError:
    leidenalg = None

from cdlib.utils import convert_graph_formats
from collections import defaultdict
from cdlib.algorithms.internal.pycondor import condor_object, initial_community, brim

__all__ = ["bimlpa", "CPM_Bipartite", "infomap_bipartite", "condor"]


def bimlpa(g_original: object, theta: float = 0.3, lambd: int = 7) -> BiNodeClustering:
    """
    BiMLPA is designed to detect the many-to-many correspondence community in bipartite networks using multi-label propagation algorithm.


    **Supported Graph Types**

    ========== ======== ======== =========
    Undirected Directed Weighted Bipartite
    ========== ======== ======== =========
    Yes        No       No       Yes
    ========== ======== ======== =========

    :param g_original: a networkx/igraph object
    :param theta: Label weights threshold. Default 0.3.
    :param lambd: The max number of labels. Default 7.
    :return: BiNodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.algorithms.bipartite.generators.random_graph(100, 20, 0.1)
    >>> coms = algorithms.bimlpa(G)

    :References:

    Taguchi, Hibiki, Tsuyoshi Murata, and Xin Liu. "BiMLPA: Community Detection in Bipartite Networks by Multi-Label Propagation." International Conference on Network Science. Springer, Cham, 2020.

    .. note:: Reference implementation: https://github.com/hbkt/BiMLPA
    """
    from BiMLPA import BiMLPA_SqrtDeg, relabeling, output_community

    g = convert_graph_formats(g_original, nx.Graph)

    if not nx.algorithms.bipartite.is_bipartite(g):
        raise ValueError("The graph is not bipartite")

    bimlpa = BiMLPA_SqrtDeg(g, theta, lambd)
    bimlpa.start()
    relabeling(g)
    top_coms, bottom_coms = output_community(g)

    return BiNodeClustering(
        top_coms,
        bottom_coms,
        g_original,
        "BiMLPA",
        method_parameters={"theta": theta, "lambd": lambd},
    )


def CPM_Bipartite(
    g_original: object,
    resolution_parameter_01: float,
    resolution_parameter_0: float = 0,
    resolution_parameter_1: float = 0,
    degree_as_node_size: bool = False,
    seed: int = 0,
) -> BiNodeClustering:
    """
    CPM_Bipartite is the extension of CPM to bipartite graphs


    **Supported Graph Types**

    ========== ======== ======== =========
    Undirected Directed Weighted Bipartite
    ========== ======== ======== =========
    Yes        No       No       Yes
    ========== ======== ======== =========

    :param g_original: a networkx/igraph object
    :param resolution_parameter_01: Resolution parameter for in between two classes.
    :param resolution_parameter_0: Resolution parameter for class 0.
    :param resolution_parameter_1: Resolution parameter for class 1.
    :param degree_as_node_size: If ``True`` use degree as node size instead of 1, to mimic modularity
    :param seed: the random seed to be used in CPM method to keep results/partitions replicable
    :return: BiNodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.algorithms.bipartite.generators.random_graph(100, 20, 0.5)
    >>> coms = algorithms.CPM_Bipartite(G, 1)

    :References:

    Barber, M. J. (2007). Modularity and community detection in bipartite networks. Physical Review E, 76(6), 066102. 10.1103/PhysRevE.76.066102

    .. note:: Reference implementation: https://leidenalg.readthedocs.io/en/stable/multiplex.html?highlight=bipartite#bipartite
    """
    if ig is None or leidenalg is None:
        raise ModuleNotFoundError(
            "Optional dependency not satisfied: install igraph and leidenalg to use the "
            "selected feature."
        )

    g = convert_graph_formats(g_original, ig.Graph)

    try:
        g.vs["name"]
    except:
        g.vs["name"] = [v.index for v in g.vs]

    optimiser = leidenalg.Optimiser()
    leidenalg.Optimiser.set_rng_seed(self=optimiser, value=seed)

    p_01, p_0, p_1 = leidenalg.CPMVertexPartition.Bipartite(
        g,
        resolution_parameter_01=resolution_parameter_01,
        resolution_parameter_0=resolution_parameter_0,
        resolution_parameter_1=resolution_parameter_1,
        degree_as_node_size=degree_as_node_size,
    )
    optimiser.optimise_partition_multiplex([p_01, p_0, p_1], layer_weights=[1, -1, -1])

    coms = defaultdict(list)
    for n in g.vs:
        coms[p_01.membership[n.index]].append(n.index)

    return BiNodeClustering(
        list(coms.values()),
        [],
        g_original,
        "CPM_Bipartite",
        method_parameters={
            "resolution_parameter_01": resolution_parameter_01,
            "resolution_parameter_0": resolution_parameter_0,
            "resolution_parameter_1": resolution_parameter_1,
            "degree_as_node_size": degree_as_node_size,
            "seed": seed,
        },
    )


def infomap_bipartite(g_original: object, flags: str = "") -> BiNodeClustering:
    """
    Infomap is based on ideas of information theory.
    The algorithm uses the probability flow of random walks on a bipartite network as a proxy for information flows in the real system and it decomposes the network into modules by compressing a description of the probability flow.


    **Supported Graph Types**

    ========== ======== ======== =========
    Undirected Directed Weighted Bipartite
    ========== ======== ======== =========
    Yes        Yes      Yes      Yes
    ========== ======== ======== =========

    :param g_original: a networkx/igraph object
    :param flags: str flags for Infomap
    :return: BiNodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.infomap_bipartite(G)

    :References:

    Rosvall M, Bergstrom CT (2008) `Maps of random walks on complex networks reveal community structure. <https://www.pnas.org/content/105/4/1118/>`_ Proc Natl Acad SciUSA 105(4):1118–1123

    .. note:: Reference implementation: https://pypi.org/project/infomap/

    .. note:: Infomap Python API documentation: https://mapequation.github.io/infomap/python/
    """

    if imp is None:
        raise ModuleNotFoundError(
            "Optional dependency not satisfied: install infomap to use the selected feature."
        )
    if pipes is None:
        raise ModuleNotFoundError(
            "Optional dependency not satisfied: install package wurlitzer to use infomap."
        )

    g = convert_graph_formats(g_original, nx.Graph)

    g1 = nx.convert_node_labels_to_integers(g, label_attribute="name")
    name_map = nx.get_node_attributes(g1, "name")
    if not nx.algorithms.bipartite.is_bipartite(g1):
        raise ValueError("The graph is not bipartite")

    X, Y = nx.algorithms.bipartite.sets(g1)
    X = {x: n for n, x in enumerate(X)}
    Y = {y: n + max(X.values()) + 1 for n, y in enumerate(Y)}
    Z = {**X, **Y}

    g1 = nx.relabel_nodes(g1, Z)
    inv_Z = {v: k for k, v in Z.items()}

    coms_to_node = defaultdict(list)

    with pipes():
        im = imp.Infomap(flags)
        im.bipartite_start_id = min(Y.keys())

        if int(imp.__version__.replace(".", "")) >= 171:
            n_dict = {i: str(n) for i, n in enumerate(g1.nodes)}
            im.add_nodes(n_dict)
        else:
            im.add_nodes(g1.nodes)

        for source, target, data in g1.edges(data=True):
            if "weight" in data:
                im.add_link(source, target, data["weight"])
            else:
                im.add_link(source, target)
        im.run()

        for node_id, module_id in im.modules:
            node_name = name_map[inv_Z[node_id]]
            coms_to_node[module_id].append(node_name)

    coms_infomap = [list(c) for c in coms_to_node.values()]
    return BiNodeClustering(
        coms_infomap,
        [],
        g_original,
        "Infomap Bipartite",
        method_parameters={"flags": flags},
    )


def condor(g_original: object) -> BiNodeClustering:
    """
    BRIM algorithm for bipartite community structure detection.
    Works on weighted and unweighted graphs.


    **Supported Graph Types**

    ========== ======== ======== =========
    Undirected Directed Weighted Bipartite
    ========== ======== ======== =========
    Yes        No       No       Yes
    ========== ======== ======== =========

    :param g_original: a networkx/igraph object
    :return: BiNodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.condor(G)

    :References:

    Platig, J., Castaldi, P. J., DeMeo, D., & Quackenbush, J. (2016). Bipartite community structure of eQTLs. PLoS computational biology, 12(9), e1005033.

    .. note:: Reference implementation: https://github.com/genisott/pycondor
    """

    g = convert_graph_formats(g_original, nx.Graph)
    net = nx.to_pandas_edgelist(g)
    co = condor_object(net)
    co = initial_community(co)
    co = brim(co)

    left = co["tar_memb"]
    right = co["reg_memb"]

    lefts = defaultdict(list)
    for index, row in left.iterrows():
        if isinstance(row["tar"], str):
            lefts[row["com"]].append(row["tar"])
        else:
            lefts[row["com"]].append(int(row["tar"]))

    rights = defaultdict(list)
    for index, row in right.iterrows():
        if isinstance(row["reg"], str):
            rights[row["com"]].append(row["reg"])
        else:
            rights[row["com"]].append(int(row["reg"]))

    return BiNodeClustering(
        list(lefts.values()),
        list(rights.values()),
        g_original,
        "Condor",
        method_parameters={},
    )
