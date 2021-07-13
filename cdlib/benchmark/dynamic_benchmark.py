from cdlib.benchmark.internal import rdyn
from cdlib import TemporalClustering, NamedClustering
from collections import defaultdict
import dynetx as dn

__all__ = ["RDyn"]


def RDyn(
    size: int = 300,
    iterations: int = 5,
    avg_deg: int = 15,
    sigma: float = 0.6,
    lambdad: float = 1,
    alpha: float = 2.5,
    paction: float = 1,
    prenewal: float = 0.8,
    quality_threshold: float = 0.5,
    new_node: float = 0.0,
    del_node: float = 0.0,
    max_evts: int = 1,
    simplified: bool = True,
) -> [dn.DynGraph, object]:
    """
    RDyn is a syntetic dynamic network generator with time-dependent ground-truth partitions having tunable quality (in terms of conductance).
    Communities' ids are aligned across time and a predefined number of merge/plit events are planted in between consecutive stable iterations.

    :param size: Number of nodes
    :param iterations: Number of stable iterations
    :param avg_deg: Average node degree. Int, default 15
    :param sigma: Percentage of node's edges within a community. Float, default .6
    :param lambdad: Community size distribution exponent. Float, default 1
    :param alpha:  Degree distribution exponent. Float, default 2.5
    :param paction: Probability of node action. Float, default 1
    :param prenewal: Probability of edge renewal. Float, default, .8
    :param quality_threshold: Conductance quality threshold for stable iteration. Float, default .5
    :param new_node: Probability of node appearance. Float, default 0
    :param del_node: Probability of node vanishing. Float, default 0
    :param max_evts: Max number of community events for stable iteration. Int, default 1
    :param simplified: Simplified execution. Boolean, default True. (NB: when True an approximation of the original process is executed - some network characteristics can deviate from the expected ones)

    :return: A dynetx DynGraph, the TemporalClustering object

    :Example:

    >>> from cdlib.benchmark import RDyn
    >>> G, coms = RDyn(n=300)

    :References:

    Rossetti, Giulio. "RDyn: graph benchmark handling community dynamics." Journal of Complex Networks 5.6 (2017): 893-912.

    .. note:: Reference implementation: https://github.com/GiulioRossetti/RDyn
    """
    tc = TemporalClustering()
    dg = dn.DynGraph()
    rdb = rdyn.RDynV2(
        size=size,
        iterations=iterations,
        avg_deg=avg_deg,
        quality_threshold=quality_threshold,
        sigma=sigma,
        lambdad=lambdad,
        alpha=alpha,
        paction=paction,
        prenewal=prenewal,
        new_node=new_node,
        del_node=del_node,
        max_evts=max_evts,
    )

    t = 0
    for g, communities in rdb.execute(simplified=simplified):
        dg.add_interactions_from(g.edges(), t=t, e=t)
        sg = dg.time_slice(t)
        nc = NamedClustering(
            communities,
            sg,
            "RDyn",
            {
                "size": size,
                "iterations": iterations,
                "avg_deg": avg_deg,
                "quality_threshold": quality_threshold,
                "sigma": sigma,
                "lambdad": lambdad,
                "alpha": alpha,
                "paction": paction,
                "prenewal": prenewal,
                "new_node": new_node,
                "del_node": del_node,
                "max_evts": max_evts,
                "simplified": simplified,
            },
            overlap=False,
        )
        tc.add_clustering(nc, time=t)
        t += 1

    return dg, tc
