import numpy as np
from cdlib.evaluation.internal import onmi
from omega_index_py3 import Omega
from nf1 import NF1
from collections import namedtuple

__all__ = ["MatchingResult", "normalized_mutual_information", "overlapping_normalized_mutual_information", "omega",
           "f1", "nf1", "adjusted_rand_index", "adjusted_mutual_information", "variation_of_information"]

MatchingResult = namedtuple("MatchingResult", ['mean', 'std'])


def __check_partition_coverage(first_partition, second_partition):
    nodes_first = {node: None for community in first_partition.communities for node in community}
    nodes_second = {node: None for community in second_partition.communities for node in community}

    if len(set(nodes_first.keys()) ^ set(nodes_second.keys())) != 0:
        raise ValueError("Both partitions should cover the same node set")
    
    
def __check_partition_overlap(first_partition, second_partition):
    if first_partition.overlap or second_partition.overlap:
        raise ValueError("Not defined for overlapping partitions")


def normalized_mutual_information(first_partition, second_partition):
    """
    Normalized Mutual Information between two clusterings.

    Normalized Mutual Information (NMI) is an normalization of the Mutual
    Information (MI) score to scale the results between 0 (no mutual
    information) and 1 (perfect correlation). In this function, mutual
    information is normalized by ``sqrt(H(labels_true) * H(labels_pred))``

    :param first_partition: NodeClustering object
    :param second_partition: NodeClustering object
    :return: normalized mutual information score

    :Example:

      >>> from cdlib import evaluation, algorithms
      >>> g = nx.karate_club_graph()
      >>> louvain_communities = algorithms.louvain(g)
      >>> leiden_communities = algorithms.leiden(g)
      >>> evaluation.normalized_mutual_information(louvain_communities,leiden_communities)

    """

    __check_partition_coverage(first_partition, second_partition)
    __check_partition_overlap(first_partition, second_partition)

    first_partition_c = [x[1]
                       for x in sorted([(node, nid)
                                        for nid, cluster in enumerate(first_partition.communities)
                                        for node in cluster], key=lambda x: x[0])]

    second_partition_c = [x[1]
                       for x in sorted([(node, nid)
                                        for nid, cluster in enumerate(second_partition.communities)
                                        for node in cluster], key=lambda x: x[0])]

    from sklearn.metrics import normalized_mutual_info_score
    return normalized_mutual_info_score(first_partition_c, second_partition_c)


def overlapping_normalized_mutual_information(first_partition, second_partition):
    """
    Overlapping Normalized Mutual Information between two clusterings.

    Extension of the Normalized Mutual Information (NMI) score to cope with overlapping partitions.

    :param first_partition: NodeClustering object
    :param second_partition: NodeClustering object
    :return: onmi score

    :Example:

    >>> from cdlib import evaluation, algorithms
    >>> g = nx.karate_club_graph()
    >>> louvain_communities = algorithms.louvain(g)
    >>> leiden_communities = algorithms.leiden(g)
    >>> evaluation.overlapping_normalized_mutual_information(louvain_communities,leiden_communities)
    :Reference:

    Original internal: https://github.com/RapidsAtHKUST/CommunityDetectionCodes
    """

    __check_partition_coverage(first_partition, second_partition)

    vertex_number_first = len({node: None for community in first_partition.communities for node in community})

    return onmi.calc_overlap_nmi(vertex_number_first, first_partition.communities, second_partition.communities)


def omega(first_partition, second_partition):
    """
    Index of resemblance for overlapping, complete coverage, network clusterings.

    :param first_partition: NodeClustering object
    :param second_partition: NodeClustering object
    :return: omega index

    :Example:

    >>> from cdlib import evaluation, algorithms
    >>> g = nx.karate_club_graph()
    >>> louvain_communities = algorithms.louvain(g)
    >>> leiden_communities = algorithms.leiden(g)
    >>> evaluation.omega(louvain_communities,leiden_communities)
    :Reference:

    1. Gabriel Murray, Giuseppe Carenini, and Raymond Ng. 2012. `Using the omega index for evaluating abstractive algorithms detection. <https://pdfs.semanticscholar.org/59d6/5d5aa09d789408fd9fd3c009a1b070ff5859.pdf/>`_ In Proceedings of Workshop on Evaluation Metrics and System Comparison for Automatic Summarization. Association for Computational Linguistics, Stroudsburg, PA, USA, 10-18.
    """

    __check_partition_coverage(first_partition, second_partition)

    first_partition = {k: v for k, v in enumerate(first_partition.communities)}
    second_partition = {k: v for k, v in enumerate(second_partition.communities)}

    om_idx = Omega(first_partition, second_partition)
    return om_idx.omega_score


def f1(first_partition, second_partition):
    """
    Compute the average F1 score of the optimal algorithms matches among the partitions in input.
    Works on overlapping/non-overlapping complete/partial coverage partitions.

    :param first_partition: NodeClustering object
    :param second_partition: NodeClustering object
    :return: F1 score (harmonic mean of precision and recall)

    :Example:

    >>> from cdlib import evaluation, algorithms
    >>> g = nx.karate_club_graph()
    >>> louvain_communities = algorithms.louvain(g)
    >>> leiden_communities = algorithms.leiden(g)
    >>> evaluation.f1(louvain_communities,leiden_communities)

    :Reference:

    1. Rossetti, G., Pappalardo, L., & Rinzivillo, S. (2016). `A novel approach to evaluate algorithms detection internal on ground truth. <https://www.researchgate.net/publication/287204505_A_novel_approach_to_evaluate_community_detection_algorithms_on_ground_truth/>`_ In Complex Networks VII (pp. 133-144). Springer, Cham.
    """

    nf = NF1(first_partition.communities, second_partition.communities)
    results = nf.summary()
    return MatchingResult(results['details']['F1 mean'][0], results['details']['F1 std'][0])


def nf1(first_partition, second_partition):
    """
    Compute the Normalized F1 score of the optimal algorithms matches among the partitions in input.
    Works on overlapping/non-overlapping complete/partial coverage partitions.

    :param first_partition: NodeClustering object
    :param second_partition: NodeClustering object
    :return: MatchingResult instance

    :Example:

    >>> from cdlib import evaluation, algorithms
    >>> g = nx.karate_club_graph()
    >>> louvain_communities = algorithms.louvain(g)
    >>> leiden_communities = algorithms.leiden(g)
    >>> evaluation.nf1(louvain_communities,leiden_communities)

    :Reference:

    1. Rossetti, G., Pappalardo, L., & Rinzivillo, S. (2016). `A novel approach to evaluate algorithms detection internal on ground truth. <https://www.researchgate.net/publication/287204505_A_novel_approach_to_evaluate_community_detection_algorithms_on_ground_truth/>`_

    2. Rossetti, G. (2017). : `RDyn: graph benchmark handling algorithms dynamics. Journal of Complex Networks. <https://academic.oup.com/comnet/article-abstract/5/6/893/3925036?redirectedFrom=PDF/>`_ 5(6), 893-912.
    """

    nf = NF1(first_partition.communities, second_partition.communities)
    results = nf.summary()
    return results['scores'].loc["NF1"][0]


def adjusted_rand_index(first_partition, second_partition):
    """Rand index adjusted for chance.

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

    :param first_partition: NodeClustering object
    :param second_partition: NodeClustering object
    :return: ARI score

    :Example:

    >>> from cdlib import evaluation, algorithms
    >>> g = nx.karate_club_graph()
    >>> louvain_communities = algorithms.louvain(g)
    >>> leiden_communities = algorithms.leiden(g)
    >>> evaluation.adjusted_rand_index(louvain_communities,leiden_communities)

    :Reference:

    1. Hubert, L., & Arabie, P. (1985). `Comparing partitions. <https://link.springer.com/article/10.1007/BF01908075/>`_ Journal of classification, 2(1), 193-218.
    """

    __check_partition_coverage(first_partition, second_partition)
    __check_partition_overlap(first_partition, second_partition)

    first_partition_c = [x[1]
                       for x in sorted([(node, nid)
                                        for nid, cluster in enumerate(first_partition.communities)
                                        for node in cluster], key=lambda x: x[0])]

    second_partition_c = [x[1]
                        for x in sorted([(node, nid)
                                         for nid, cluster in enumerate(second_partition.communities)
                                         for node in cluster], key=lambda x: x[0])]

    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(first_partition_c, second_partition_c)


def adjusted_mutual_information(first_partition, second_partition):
    """Adjusted Mutual Information between two clusterings.

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

    :param first_partition: NodeClustering object
    :param second_partition: NodeClustering object
    :return: AMI score

    :Example:

    >>> from cdlib import evaluation, algorithms
    >>> g = nx.karate_club_graph()
    >>> louvain_communities = algorithms.louvain(g)
    >>> leiden_communities = algorithms.leiden(g)
    >>> evaluation.adjusted_mutual_information(louvain_communities,leiden_communities)

    :Reference:

    1. Vinh, N. X., Epps, J., & Bailey, J. (2010). `Information theoretic measures for clusterings comparison: Variants, properties, normalization and correction for chance. <http://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf/>`_ Journal of Machine Learning Research, 11(Oct), 2837-2854.
    """

    __check_partition_coverage(first_partition, second_partition)
    __check_partition_overlap(first_partition, second_partition)

    first_partition_c = [x[1]
                       for x in sorted([(node, nid)
                                        for nid, cluster in enumerate(first_partition.communities)
                                        for node in cluster], key=lambda x: x[0])]

    second_partition_c = [x[1]
                        for x in sorted([(node, nid)
                                         for nid, cluster in enumerate(second_partition.communities)
                                         for node in cluster], key=lambda x: x[0])]

    from sklearn.metrics import adjusted_mutual_info_score
    return adjusted_mutual_info_score(first_partition_c, second_partition_c)


def variation_of_information(first_partition, second_partition):
    """ Variation of Information among two nodes partitions.

    $$ H(p)+H(q)-2MI(p, q) $$

    where MI is the mutual information, H the partition entropy and p,q are the algorithms sets

    :param first_partition: NodeClustering object
    :param second_partition: NodeClustering object
    :return: VI score

    :Example:

    >>> from cdlib import evaluation, algorithms
    >>> g = nx.karate_club_graph()
    >>> louvain_communities = algorithms.louvain(g)
    >>> leiden_communities = algorithms.leiden(g)
    >>> evaluation.variation_of_information(louvain_communities,leiden_communities)

    :Reference:

    1. Meila, M. (2007). `Comparing clusterings - an information based distance. <https://www.sciencedirect.com/science/article/pii/S0047259X06002016/>`_ Journal of Multivariate Analysis, 98, 873-895. doi:10.1016/j.jmva.2006.11.013
    """

    __check_partition_coverage(first_partition, second_partition)
    __check_partition_overlap(first_partition, second_partition)

    n = float(sum([len(c1) for c1 in first_partition.communities]))
    sigma = 0.0
    for c1 in first_partition.communities:
        p = len(c1) / n
        for c2 in second_partition.communities:
            q = len(c2) / n
            r = len(set(c1) & set(c2)) / n
            if r > 0.0:
                sigma += r * (np.log2(r / p) + np.log2(r / q))

    return abs(sigma)

