from cdlib.classes.clustering import Clustering
from cdlib import evaluation
import networkx as nx

try:
    import igraph as ig
except ModuleNotFoundError:
    ig = None

from collections import defaultdict


class NodeClustering(Clustering):
    """Node Communities representation.

    :param communities: list of communities
    :param graph: a networkx/igraph object
    :param method_name: community discovery algorithm name
    :param method_parameters: configuration for the community discovery algorithm used
    :param overlap: boolean, whether the partition is overlapping or not
    """

    def __init__(
        self,
        communities: list,
        graph: object,
        method_name: str = "",
        method_parameters: dict = None,
        overlap: bool = False,
    ):
        super().__init__(communities, graph, method_name, method_parameters, overlap)

        if graph is not None:
            node_count = len(
                {nid: None for community in communities for nid in community}
            )
            if isinstance(graph, nx.Graph):
                self.node_coverage = node_count / graph.number_of_nodes()
            elif ig is not None and isinstance(graph, ig.Graph):
                self.node_coverage = node_count / graph.vcount()
            else:
                raise ValueError("Unsupported Graph type.")

    def __check_graph(self) -> bool:
        return self.graph is not None

    def to_node_community_map(self) -> dict:
        """
        Generate a <node, list(communities)> representation of the current clustering

        :return: dict of the form <node, list(communities)>
        """

        node_to_communities = defaultdict(list)
        for cid, community in enumerate(self.communities):
            for node in community:
                node_to_communities[node].append(cid)

        return node_to_communities

    def link_modularity(self) -> evaluation.FitnessResult:
        """
        Quality function designed for directed graphs with overlapping communities.

        :return: the link modularity score

        :Example:

        >>> from cdlib import evaluation
        >>> from cdlib.algorithms import louvain
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> mod = communities.link_modularity()

        """
        if self.__check_graph():
            return evaluation.link_modularity(self.graph, self)
        else:
            raise ValueError("Graph instance not specified")

    def normalized_cut(self, **kwargs: dict) -> evaluation.FitnessResult:
        """
        Normalized variant of the Cut-Ratio

        .. math:: : f(S) = \\frac{c_S}{2m_S+c_S} + \\frac{c_S}{2(m−m_S )+c_S}

        where :math:`m` is the number of graph edges, :math:`m_S` is the number of algorithms internal edges and :math:`c_S` is the number of algorithms nodes.

        :param summary: (optional, default True) if **True**, an overall summary is returned for the partition (min, max, avg, std); if **False** a list of community-wise score
        :return: a FitnessResult object/a list of community-wise score

        :Example:

        >>> from cdlib.algorithms import louvain
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> mod = communities.normalized_cut()

        """
        if self.__check_graph():
            return evaluation.normalized_cut(self.graph, self, **kwargs)
        else:
            raise ValueError("Graph instance not specified")

    def internal_edge_density(self, **kwargs: dict) -> evaluation.FitnessResult:
        """
        The internal density of the algorithms set.

        .. math:: f(S) = \\frac{m_S}{n_S(n_S−1)/2}

        where :math:`m_S` is the number of algorithms internal edges and :math:`n_S` is the number of algorithms nodes.

        :param summary: (optional, default True) if **True**, an overall summary is returned for the partition (min, max, avg, std); if **False** a list of community-wise score
        :return: a FitnessResult object/a list of community-wise score

        :Example:

        >>> from cdlib.algorithms import louvain
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> mod = communities.internal_edge_density()

        """
        if self.__check_graph():
            return evaluation.internal_edge_density(self.graph, self, **kwargs)
        else:
            raise ValueError("Graph instance not specified")

    def average_internal_degree(self, **kwargs: dict) -> evaluation.FitnessResult:
        """
        The average internal degree of the algorithms set.

        .. math:: f(S) = \\frac{2m_S}{n_S}

        where :math:`m_S` is the number of algorithms internal edges and :math:`n_S` is the number of algorithms nodes.

        :param summary: (optional, default True) if **True**, an overall summary is returned for the partition (min, max, avg, std); if **False** a list of community-wise score
        :return: a FitnessResult object/a list of community-wise score

        :Example:

        >>> from cdlib.algorithms import louvain
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> mod = communities.average_internal_degree()


        """
        if self.__check_graph():
            return evaluation.average_internal_degree(self.graph, self, **kwargs)
        else:
            raise ValueError("Graph instance not specified")

    def fraction_over_median_degree(self, **kwargs: dict) -> evaluation.FitnessResult:
        """
        Fraction of algorithms nodes of having internal degree higher than the median degree value.

        .. math:: f(S) = \\frac{|\{u: u \\in S,| \{(u,v): v \\in S\}| > d_m\}| }{n_S}


        where :math:`d_m` is the internal degree median value

        :param summary: (optional, default True) if **True**, an overall summary is returned for the partition (min, max, avg, std); if **False** a list of community-wise score
        :return: a FitnessResult object/a list of community-wise score

        :Example:

        >>> from cdlib.algorithms import louvain
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> mod = communities.fraction_over_median_degree()

        """
        if self.__check_graph():
            return evaluation.fraction_over_median_degree(self.graph, self, **kwargs)
        else:
            raise ValueError("Graph instance not specified")

    def expansion(self, **kwargs: dict) -> evaluation.FitnessResult:
        """
        Number of edges per algorithms node that point outside the cluster.

        .. math:: f(S) = \\frac{c_S}{n_S}

        where :math:`n_S` is the number of edges on the algorithms boundary, :math:`c_S` is the number of algorithms nodes.

        :param summary: (optional, default True) if **True**, an overall summary is returned for the partition (min, max, avg, std); if **False** a list of community-wise score
        :return: a FitnessResult object/a list of community-wise score

        :Example:

        >>> from cdlib.algorithms import louvain
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> mod = communities.expansion()

        """
        if self.__check_graph():
            return evaluation.expansion(self.graph, self, **kwargs)
        else:
            raise ValueError("Graph instance not specified")

    def cut_ratio(self, **kwargs: dict) -> evaluation.FitnessResult:
        """
        Fraction of existing edges (out of all possible edges) leaving the algorithms.

        ..math:: f(S) = \\frac{c_S}{n_S (n − n_S)}

        where :math:`c_S` is the number of algorithms nodes and, :math:`n_S` is the number of edges on the algorithms boundary

        :param summary: (optional, default True) if **True**, an overall summary is returned for the partition (min, max, avg, std); if **False** a list of community-wise score
        :return: a FitnessResult object/a list of community-wise score

        :Example:

        >>> from cdlib.algorithms import louvain
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> mod = communities.cut_ratio()

        """
        if self.__check_graph():
            return evaluation.cut_ratio(self.graph, self, **kwargs)
        else:
            raise ValueError("Graph instance not specified")

    def edges_inside(self, **kwargs: dict) -> evaluation.FitnessResult:
        """
        Number of edges internal to the algorithms.

        :param summary: (optional, default True) if **True**, an overall summary is returned for the partition (min, max, avg, std); if **False** a list of community-wise score
        :return: a FitnessResult object/a list of community-wise score

        :Example:

        >>> from cdlib.algorithms import louvain
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> mod = communities.edges_inside()

        """
        if self.__check_graph():
            return evaluation.edges_inside(self.graph, self, **kwargs)
        else:
            raise ValueError("Graph instance not specified")

    def conductance(self, **kwargs: dict) -> evaluation.FitnessResult:
        """
        Fraction of total edge volume that points outside the algorithms.

        .. math:: f(S) = \\frac{c_S}{2 m_S+c_S}

        where :math:`c_S` is the number of algorithms nodes and, :math:`m_S` is the number of algorithms edges

        :param summary: (optional, default True) if **True**, an overall summary is returned for the partition (min, max, avg, std); if **False** a list of community-wise score
        :return: a FitnessResult object/a list of community-wise score

        :Example:

        >>> from cdlib.algorithms import louvain
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> mod = communities.conductance()

        """
        if self.__check_graph():
            return evaluation.conductance(self.graph, self, **kwargs)
        else:
            raise ValueError("Graph instance not specified")

    def max_odf(self, **kwargs: dict) -> evaluation.FitnessResult:
        """
        Maximum fraction of edges of a node of a algorithms that point outside the algorithms itself.

        .. math:: max_{u \\in S} \\frac{|\{(u,v)\\in E: v \\not\\in S\}|}{d(u)}

        where :math:`E` is the graph edge set, :math:`v` is a node in :math:`S` and :math:`d(u)` is the degree of :math:`u`

        :param summary: (optional, default True) if **True**, an overall summary is returned for the partition (min, max, avg, std); if **False** a list of community-wise score
        :return: a FitnessResult object/a list of community-wise score

        :Example:

        >>> from cdlib.algorithms import louvain
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> mod = communities.max_odf()

        """
        if self.__check_graph():
            return evaluation.max_odf(self.graph, self, **kwargs)
        else:
            raise ValueError("Graph instance not specified")

    def avg_odf(self, **kwargs: dict) -> evaluation.FitnessResult:
        """
        Average fraction of edges of a node of a algorithms that point outside the algorithms itself.

        .. math:: \\frac{1}{n_S} \\sum_{u \\in S} \\frac{|\{(u,v)\\in E: v \\not\\in S\}|}{d(u)}

        where :math:`E` is the graph edge set, :math:`v` is a node in :math:`S`, :math:`d(u)` is the degree of :math:`u` and :math:`n_S` is the set of algorithms nodes.

        :param summary: (optional, default True) if **True**, an overall summary is returned for the partition (min, max, avg, std); if **False** a list of community-wise score
        :return: a FitnessResult object/a list of community-wise score
        :Example:

        >>> from cdlib.algorithms import louvain
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> mod = communities.avg_odf()

        """
        if self.__check_graph():
            return evaluation.avg_odf(self.graph, self, **kwargs)
        else:
            raise ValueError("Graph instance not specified")

    def flake_odf(self, **kwargs: dict) -> evaluation.FitnessResult:
        """
        Fraction of nodes in S that have fewer edges pointing inside than to the outside of the algorithms.

        .. math:: f(S) = \\frac{| \{ u:u \in S,| \{(u,v) \in E: v \in S \}| < d(u)/2 \}|}{n_S}

        where :math:`E` is the graph edge set, :math:`v` is a node in :math:`S`, :math:`d(u)` is the degree of :math:`u` and :math:`n_S` is the set of algorithms nodes.

        :param summary: (optional, default True) if **True**, an overall summary is returned for the partition (min, max, avg, std); if **False** a list of community-wise score
        :return: a FitnessResult object/a list of community-wise score

        :Example:

        >>> from cdlib.algorithms import louvain
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> mod = communities.flake_odf()

        """
        if self.__check_graph():
            return evaluation.flake_odf(self.graph, self, **kwargs)
        else:
            raise ValueError("Graph instance not specified")

    def triangle_participation_ratio(self, **kwargs: dict) -> evaluation.FitnessResult:
        """
        Fraction of algorithms nodes that belong to a triad.

        .. math:: f(S) = \\frac{ | \{ u: u \in S,\{(v,w):v, w \in S,(u,v) \in E,(u,w) \in E,(v,w) \in E \} \\not = \\emptyset \} |}{n_S}

        where :math:`n_S` is the set of algorithms nodes.

        :param summary: (optional, default True) if **True**, an overall summary is returned for the partition (min, max, avg, std); if **False** a list of community-wise score
        :return: a FitnessResult object/a list of community-wise score

        :Example:

        >>> from cdlib.algorithms import louvain
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> mod = communities.triangle_participation_ratio()

        """
        if self.__check_graph():
            return evaluation.triangle_participation_ratio(self.graph, self, **kwargs)
        else:
            raise ValueError("Graph instance not specified")

    def size(self, **kwargs: dict) -> evaluation.FitnessResult:
        """Size is the number of nodes in the community

        :param summary: (optional, default True) if **True**, an overall summary is returned for the partition (min, max, avg, std); if **False** a list of community-wise score
        :return: a FitnessResult object/a list of community-wise score

        Example:

        >>> from cdlib.algorithms import louvain
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> mod = communities.size()
        """

        return evaluation.size(self.graph, self, **kwargs)

    def newman_girvan_modularity(self) -> evaluation.FitnessResult:
        """
        Difference the fraction of intra algorithms edges of a partition with the expected number of such edges if distributed according to a null model.

        In the standard version of modularity, the null model preserves the expected degree sequence of the graph under consideration. In other words, the modularity compares the real network structure with a corresponding one where nodes are connected without any preference about their neighbors.

        .. math:: Q(S) = \\frac{1}{m}\\sum_{c \\in S}(m_S - \\frac{(2 m_S + l_S)^2}{4m})

        where :math:`m` is the number of graph edges, :math:`m_S` is the number of algorithms edges, :math:`l_S` is the number of edges from nodes in S to nodes outside S.


        :return: the Newman-Girvan modularity score

        :Example:

        >>> from cdlib.algorithms import louvain
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> mod = communities.newman_girvan_modularity()

        :References:

        Newman, M.E.J. & Girvan, M. **Finding and evaluating algorithms structure in networks.** Physical Review E 69, 26113(2004).
        """
        if self.__check_graph():
            return evaluation.newman_girvan_modularity(self.graph, self)
        else:
            raise ValueError("Graph instance not specified")

    def erdos_renyi_modularity(self) -> evaluation.FitnessResult:
        """

        Erdos-Renyi modularity is a variation of the Newman-Girvan one.
        It assumes that vertices in a network are connected randomly with a constant probability :math:`p`.

        .. math:: Q(S) = \\frac{1}{m}\\sum_{c \\in S} (m_S − \\frac{mn_S(n_S −1)}{n(n−1)})

        where :math:`m` is the number of graph edges, :math:`m_S` is the number of algorithms edges, :math:`l_S` is the number of edges from nodes in S to nodes outside S.


        :return: the Erdos-Renyi modularity score

        :Example:

        >>> from cdlib.algorithms import louvain
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> mod = communities.erdos_renyi_modularity()

        :References:

        Erdos, P., & Renyi, A. (1959). **On random graphs I.** Publ. Math. Debrecen, 6, 290-297.
        """
        if self.__check_graph():
            return evaluation.erdos_renyi_modularity(self.graph, self)
        else:
            raise ValueError("Graph instance not specified")

    def modularity_density(self) -> evaluation.FitnessResult:
        """
        The modularity density is one of several propositions that envisioned to palliate the resolution limit issue of modularity based measures.
        The idea of this metric is to include the information about algorithms size into the expected density of algorithms to avoid the negligence of small and dense communities.
        For each algorithms :math:`C` in partition :math:`S`, it uses the average modularity degree calculated by :math:`d(C) = d^{int(C)} − d^{ext(C)}` where :math:`d^{int(C)}` and :math:`d^{ext(C)}` are the average internal and external degrees of :math:`C` respectively to evaluate the fitness of :math:`C` in its network.
        Finally, the modularity density can be calculated as follows:

        .. math:: Q(S) = \\sum_{C \\in S} \\frac{1}{n_C} ( \\sum_{i \\in C} k^{int}_{iC} - \\sum_{i \\in C} k^{out}_{iC})

        where :math:`n_C` is the number of nodes in C, :math:`k^{int}_{iC}` is the degree of node i within :math:`C` and :math:`k^{out}_{iC}` is the deree of node i outside :math:`C`.


        :return: the modularity density score

        :Example:

        >>> from cdlib.algorithms import louvain
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> mod = communities.modularity_density()

        :References:

        Li, Z., Zhang, S., Wang, R. S., Zhang, X. S., & Chen, L. (2008). **Quantitative function for algorithms detection.** Physical review E, 77(3), 036109.

        """

        if self.__check_graph():
            return evaluation.modularity_density(self.graph, self)
        else:
            raise ValueError("Graph instance not specified")

    def z_modularity(self) -> evaluation.FitnessResult:
        """
        Z-modularity is another variant of the standard modularity proposed to avoid the resolution limit.
        The concept of this version is based on an observation that the difference between the fraction of edges inside communities and the expected number of such edges in a null model should not be considered as the only contribution to the final quality of algorithms structure.

        :return: the z-modularity score

        :Example:

        >>> from cdlib.algorithms import louvain
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> mod = communities.z_modularity()


        :References:

        Miyauchi, Atsushi, and Yasushi Kawase. **Z-score-based modularity for algorithms detection in networks.** PloS one 11.1 (2016): e0147805.
        """

        if self.__check_graph():
            return evaluation.z_modularity(self.graph, self)
        else:
            raise ValueError("Graph instance not specified")

    def modularity_overlap(self, weight: str = None) -> evaluation.FitnessResult:
        """Determines the Overlapping Modularity of a partition C on a graph G.

        Overlapping Modularity is defined as

        .. math:: M_{c_{r}}^{ov} = \\sum_{i \\in c_{r}} \\frac{\\sum_{j \\in c_{r}, i \\neq j}a_{ij} - \\sum_{j \\not \\in c_{r}}a_{ij}}{d_{i} \\cdot s_{i}} \\cdot \\frac{n_{c_{r}}^{e}}{n_{c_{r}} \\cdot \\binom{n_{c_{r}}}{2}}

        :param weight: label identifying the edge weight parameter name (if present), default None
        :return: FitnessResult object

        Example:

        >>> from cdlib.algorithms import louvain
        >>> from cdlib import evaluation
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> mod = communities.modularity_overlap()

        :References:

        1. A. Lazar, D. Abel and T. Vicsek, "Modularity measure of networks with overlapping communities"  EPL, 90 (2010) 18001 doi: 10.1209/0295-5075/90/18001
        """

        if self.__check_graph():
            return evaluation.modularity_overlap(self.graph, self, weight)
        else:
            raise ValueError("Graph instance not specified")

    def surprise(self) -> evaluation.FitnessResult:
        """
        Surprise is statistical approach proposes a quality metric assuming that edges between vertices emerge randomly according to a hyper-geometric distribution.

        According to the Surprise metric, the higher the score of a partition, the less likely it is resulted from a random realization, the better the quality of the algorithms structure.

        :return: the surprise score

        :Example:

        >>> from cdlib.algorithms import louvain
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> mod = communities.surprise()

        :References:

        Traag, V. A., Aldecoa, R., & Delvenne, J. C. (2015). **Detecting communities using asymptotical surprise.** Physical Review E, 92(2), 022816.
        """

        if self.__check_graph():
            return evaluation.surprise(self.graph, self)
        else:
            raise ValueError("Graph instance not specified")

    def significance(self) -> evaluation.FitnessResult:
        """
        Significance estimates how likely a partition of dense communities appear in a random graph.

        :return: the significance score

        :Example:

        >>> from cdlib.algorithms import louvain
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> mod = communities.significance()


        :References:

        Traag, V. A., Aldecoa, R., & Delvenne, J. C. (2015). **Detecting communities using asymptotical surprise.** Physical Review E, 92(2), 022816.
        """

        if self.__check_graph():
            return evaluation.significance(self.graph, self)
        else:
            raise ValueError("Graph instance not specified")

    def scaled_density(self, **kwargs: dict) -> evaluation.FitnessResult:
        """Scaled density.

        The scaled density of a community is defined as the ratio of the community density w.r.t. the complete graph density.

        :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
        :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

        Example:

        >>> from cdlib.algorithms import louvain
        >>> from cdlib import evaluation
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> scd = communities.scaled_density()
        """

        if self.__check_graph():
            return evaluation.scaled_density(self.graph, self, **kwargs)
        else:
            raise ValueError("Graph instance not specified")

    def avg_distance(self, **kwargs: dict) -> evaluation.FitnessResult:
        """Average distance.

        The average distance of a community is defined average path length across all possible pair of nodes composing it.

        :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
        :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

        Example:

        >>> from cdlib.algorithms import louvain
        >>> from cdlib import evaluation
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> scd = communities.avg_distance()
        """

        if self.__check_graph():
            return evaluation.avg_distance(self.graph, self, **kwargs)
        else:
            raise ValueError("Graph instance not specified")

    def hub_dominance(self, **kwargs: dict) -> evaluation.FitnessResult:
        """Hub dominance.

        The hub dominance of a community is defined as the ratio of the degree of its most connected node w.r.t. the theoretically maximal degree within the community.

        :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
        :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

        Example:

        >>> from cdlib.algorithms import louvain
        >>> from cdlib import evaluation
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> scd = communities.hub_dominance()
        """

        if self.__check_graph():
            return evaluation.hub_dominance(self.graph, self, **kwargs)
        else:
            raise ValueError("Graph instance not specified")

    def avg_transitivity(self, **kwargs: dict) -> evaluation.FitnessResult:
        """Average transitivity.

        The average transitivity of a community is defined the as the average clustering coefficient of its nodes w.r.t. their connection within the community itself.

        :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
        :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

        Example:

        >>> from cdlib.algorithms import louvain
        >>> from cdlib import evaluation
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> scd = communities.avg_transitivity()
        """

        if self.__check_graph():
            return evaluation.avg_transitivity(self.graph, self, **kwargs)
        else:
            raise ValueError("Graph instance not specified")

    def avg_embeddedness(self, **kwargs: dict) -> evaluation.FitnessResult:
        """Average embeddedness of nodes within the community.

        The embeddedness of a node n w.r.t. a community C is the ratio of its degree within the community and its overall degree.

        .. math:: emb(n,C) = \\frac{k_n^C}{k_n}

        The average embeddedness of a community C is:

        .. math:: avg_embd(c) = \\frac{1}{|C|} \sum_{i \in C} \\frac{k_n^C}{k_n}

        :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
        :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

        Example:

        >>> from cdlib.algorithms import louvain
        >>> from cdlib import evaluation
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> ave = communities.avg_embeddedness()

        """

        if self.__check_graph():
            return evaluation.avg_embeddedness(self.graph, self, **kwargs)
        else:
            raise ValueError("Graph instance not specified")

    def normalized_mutual_information(
        self, clustering: Clustering
    ) -> evaluation.MatchingResult:
        """
        Normalized Mutual Information between two clusterings.

        Normalized Mutual Information (NMI) is an normalization of the Mutual
        Information (MI) score to scale the results between 0 (no mutual
        information) and 1 (perfect correlation). In this function, mutual
        information is normalized by ``sqrt(H(labels_true) * H(labels_pred))``

        :param clustering: NodeClustering object
        :return: normalized mutual information score

        :Example:

        >>> from cdlib.algorithms import louvain
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> leiden_communities = algorithms.leiden(g)
        >>> mod = communities.normalized_mutual_information(leiden_communities)

        """

        return evaluation.normalized_mutual_information(self, clustering)

    def overlapping_normalized_mutual_information_LFK(
        self, clustering: Clustering
    ) -> evaluation.MatchingResult:
        """
        Overlapping Normalized Mutual Information between two clusterings.

        Extension of the Normalized Mutual Information (NMI) score to cope with overlapping partitions.
        This is the version proposed by Lancichinetti et al.

        :param clustering: NodeClustering object
        :return: onmi score

        :Example:

        >>> from cdlib.algorithms import louvain
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> leiden_communities = algorithms.leiden(g)
        >>> mod = communities.overlapping_normalized_mutual_information_LFK(leiden_communities)

        :Reference:

        1. Lancichinetti, A., Fortunato, S., & Kertesz, J. (2009). Detecting the overlapping and hierarchical community structure in complex networks. New Journal of Physics, 11(3), 033015.

        """
        return evaluation.overlapping_normalized_mutual_information_LFK(
            self, clustering
        )

    def overlapping_normalized_mutual_information_MGH(
        self, clustering: Clustering, normalization: str = "max"
    ) -> evaluation.MatchingResult:
        """
        Overlapping Normalized Mutual Information between two clusterings.

        Extension of the Normalized Mutual Information (NMI) score to cope with overlapping partitions.
        This is the version proposed by McDaid et al. using a different normalization than the original LFR one. See ref.
        for more details.

        :param clustering: NodeClustering object
        :param normalization: one of "max" or "LFK". Default "max" (corresponds to the main method described in the article)
        :return: onmi score

        :Example:

        >>> from cdlib import evaluation, algorithms
        >>> g = nx.karate_club_graph()
        >>> louvain_communities = algorithms.louvain(g)
        >>> leiden_communities = algorithms.leiden(g)
        >>> evaluation.overlapping_normalized_mutual_information_MGH(louvain_communities,leiden_communities)
        :Reference:

        1. McDaid, A. F., Greene, D., & Hurley, N. (2011). Normalized mutual information to evaluate overlapping community finding algorithms. arXiv preprint arXiv:1110.2515. Chicago
        """
        return evaluation.overlapping_normalized_mutual_information_MGH(
            self, clustering, normalization=normalization
        )

    def omega(self, clustering: Clustering) -> evaluation.MatchingResult:
        """

        Index of resemblance for overlapping, complete coverage, network clusterings.

        :param clustering: NodeClustering object
        :return: omega index

        :Example:

        >>> from cdlib.algorithms import louvain
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> leiden_communities = algorithms.leiden(g)
        >>> mod = communities.omega(leiden_communities)


        :Reference:

        1. Gabriel Murray, Giuseppe Carenini, and Raymond Ng. 2012. **Using the omega index for evaluating abstractive algorithms detection.** In Proceedings of Workshop on Evaluation Metrics and System Comparison for Automatic Summarization. Association for Computational Linguistics, Stroudsburg, PA, USA, 10-18.
        """
        return evaluation.omega(self, clustering)

    def f1(self, clustering: Clustering) -> evaluation.MatchingResult:
        """
        Compute the average F1 score of the optimal algorithms matches among the partitions in input.
        Works on overlapping/non-overlapping complete/partial coverage partitions.

        :param clustering: NodeClustering object
        :return: F1 score (harmonic mean of precision and recall)

        :Example:

        >>> from cdlib.algorithms import louvain
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> leiden_communities = algorithms.leiden(g)
        >>> mod = communities.f1(leiden_communities)


        :Reference:

        1. Rossetti, G., Pappalardo, L., & Rinzivillo, S. (2016). **A novel approach to evaluate algorithms detection internal on ground truth.** In Complex Networks VII (pp. 133-144). Springer, Cham.
        """
        return evaluation.f1(self, clustering)

    def nf1(self, clustering: Clustering) -> evaluation.MatchingResult:
        """
        Compute the Normalized F1 score of the optimal algorithms matches among the partitions in input.
        Works on overlapping/non-overlapping complete/partial coverage partitions.

        :param clustering: NodeClustering object
        :return: MatchingResult instance

        :Example:

        >>> from cdlib.algorithms import louvain
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> leiden_communities = algorithms.leiden(g)
        >>> mod = communities.nf1(leiden_communities)


        :Reference:

        1. Rossetti, G., Pappalardo, L., & Rinzivillo, S. (2016). **A novel approach to evaluate algorithms detection internal on ground truth.**

        2. Rossetti, G. (2017). : **RDyn: graph benchmark handling algorithms dynamics. Journal of Complex Networks.** 5(6), 893-912.
        """
        return evaluation.nf1(self, clustering)

    def adjusted_rand_index(self, clustering: Clustering) -> evaluation.MatchingResult:
        """
        Rand index adjusted for chance.

        The Rand Index computes a similarity measure between two clusterings
        by considering all pairs of samples and counting pairs that are
        assigned in the same or different clusters in the predicted and
        true clusterings.

        The raw RI score is then "adjusted for chance" into the ARI score
        using the following scheme::

            ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)

        The adjusted Rand index is thus ensured to have a value close to
        0.0 for random labeling independently of the number of clusters and
        samples and exactly 1.0 when the clusterings are identical (up to
        a permutation).

        ARI is a symmetric measure::

            adjusted_rand_index(a, b) == adjusted_rand_index(b, a)

        :param clustering: NodeClustering object
        :return: ARI score

        :Example:

        >>> from cdlib.algorithms import louvain
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> leiden_communities = algorithms.leiden(g)
        >>> mod = communities.adjusted_rand_index(leiden_communities)


        :Reference:

        1. Hubert, L., & Arabie, P. (1985). **Comparing partitions**. Journal of classification, 2(1), 193-218.
        """
        return evaluation.adjusted_rand_index(self, clustering)

    def adjusted_mutual_information(
        self, clustering: Clustering
    ) -> evaluation.MatchingResult:
        """
        Adjusted Mutual Information between two clusterings.

        Adjusted Mutual Information (AMI) is an adjustment of the Mutual
        Information (MI) score to account for chance. It accounts for the fact that
        the MI is generally higher for two clusterings with a larger number of
        clusters, regardless of whether there is actually more information shared.
        For two clusterings :math:`U` and :math:`V`, the AMI is given as::

            AMI(U, V) = [MI(U, V) - E(MI(U, V))] / [max(H(U), H(V)) - E(MI(U, V))]

        This metric is independent of the absolute values of the labels:
        a permutation of the class or cluster label values won't change the
        score value in any way.

        This metric is furthermore symmetric: switching ``label_true`` with
        ``label_pred`` will return the same score value. This can be useful to
        measure the agreement of two independent label assignments strategies
        on the same dataset when the real ground truth is not known.

        Be mindful that this function is an order of magnitude slower than other
        metrics, such as the Adjusted Rand Index.

        :param clustering: NodeClustering object
        :return: AMI score

        :Example:

        >>> from cdlib.algorithms import louvain
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> leiden_communities = algorithms.leiden(g)
        >>> mod = communities.adjusted_mutual_information(leiden_communities)

        :Reference:

        1. Vinh, N. X., Epps, J., & Bailey, J. (2010). **Information theoretic measures for clusterings comparison: Variants, properties, normalization and correction for chance.** Journal of Machine Learning Research, 11(Oct), 2837-2854.
        """
        return evaluation.adjusted_mutual_information(self, clustering)

    def variation_of_information(
        self, clustering: Clustering
    ) -> evaluation.MatchingResult:
        """
        Variation of Information among two nodes partitions.

        $$ H(p)+H(q)-2MI(p, q) $$

        where MI is the mutual information, H the partition entropy and p,q are the algorithms sets

        :param clustering: NodeClustering object
        :return: VI score

        :Example:

        >>> from cdlib.algorithms import louvain
        >>> g = nx.karate_club_graph()
        >>> communities = louvain(g)
        >>> leiden_communities = algorithms.leiden(g)
        >>> mod = communities.variation_of_information(leiden_communities)


        :Reference:

        1. Meila, M. (2007). **Comparing clusterings - an information based distance.** Journal of Multivariate Analysis, 98, 873-895. doi:10.1016/j.jmva.2006.11.013
        """
        return evaluation.variation_of_information(self, clustering)
