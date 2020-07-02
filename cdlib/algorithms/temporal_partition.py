from cdlib import TemporalClustering, NodeClustering
from cdlib.algorithms.internal_dcd.eTILES import eTILES

__all__ = ['tiles']


def tiles(dg, obs=1):
    """
    TILES is designed to incrementally identify and update communities in stream graphs.
    This implementation assume an explicit edge removal when pairwise interactions cease to exist.

    :param dg: dynetx graph object
    :param obs: community observation interval (default=1)
    :return: TemporalClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import dynetx as dn
    >>> dg = dn.DynGraph()
    >>> for x in range(10):
    >>>     g = nx.erdos_renyi_graph(200, 0.05)
    >>>     dg.add_interactions_from(list(g.edges()), t=x)
    >>> coms = algorithms.tiles(dg, 2)

    :References:

    Rossetti, Giulio; Pappalardo, Luca; Pedreschi, Dino, and Giannotti, Fosca. `Tiles: an online algorithm for community discovery in dynamic social networks.<https://link.springer.com/article/10.1007/s10994-016-5582-8>`_ Machine Learning (2016), 106(8), 1213-1241.
    """
    alg = eTILES(dg=dg, obs=obs)
    tc = TemporalClustering()
    t = obs
    for c in alg.execute():
        communities = []
        for k, v in c.items():
            communities.append(list(v.keys()))
        sg = dg.time_slice(t-obs, t)

        nc = NodeClustering(communities, sg, 'TILES', {'obs': obs}, overlap=True)

        if t <= max(dg.temporal_snapshots_ids()):
            tc.add_clustering(nc, time=t)
            t += obs
        else:
            break
    return tc
