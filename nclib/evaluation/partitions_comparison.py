import numpy as np
from nclib.evaluation.scoring_functions import onmi
from omega_index import Omega
from nf1 import NF1
import sklearn
from collections import namedtuple

MatchingResult = namedtuple("Result", ['mean', 'std'])


def __check_partition_coverage(first_partition, second_partition):
    nodes_first = {node: None for community in first_partition for node in community}
    nodes_second = {node: None for community in second_partition for node in community}

    if len(set(nodes_first.keys()) ^ set(nodes_second.keys())) != 0:
        raise ValueError("Both partitions should cover the same node set")


def normalized_mutual_information(first_partition, second_partition):
    """

    :param first_partition:
    :param second_partition:
    :return:
    """

    __check_partition_coverage(first_partition, second_partition)

    first_partition = [x[1]
                       for x in sorted([(node, nid)
                                        for nid, cluster in enumerate(first_partition)
                                        for node in cluster], key=lambda x: x[0])]

    second_partition = [x[1]
                       for x in sorted([(node, nid)
                                        for nid, cluster in enumerate(second_partition)
                                        for node in cluster], key=lambda x: x[0])]

    return sklearn.metrics.normalized_mutual_info_score(first_partition, second_partition)


def overlapping_normalized_mutual_information(first_partition, second_partition):
    """

    :param first_partition:
    :param second_partition:
    :return:
    """

    __check_partition_coverage(first_partition, second_partition)

    vertex_number_first = len({node: None for community in first_partition for node in community})

    return onmi.calc_overlap_nmi(vertex_number_first, first_partition, second_partition)


def omega(first_partition, second_partition):
    """

    :param first_partition:
    :param second_partition:
    :return:
    """

    __check_partition_coverage(first_partition, second_partition)

    first_partition = {k: v for k, v in enumerate(first_partition)}
    second_partition = {k: v for k, v in enumerate(second_partition)}

    om_idx = Omega(first_partition, second_partition)
    return om_idx.omega_score


def f1(first_partition, second_partition):
    """

    :param first_partition:
    :param second_partition:
    :return:
    """

    nf = NF1(first_partition, second_partition)
    results = nf.summary()
    return MatchingResult(results['details']['F1 mean'][0], results['details']['F1 std'][0])


def nf1(first_partition, second_partition):
    """

    :param first_partition:
    :param second_partition:
    :return:
    """

    nf = NF1(first_partition, second_partition)
    results = nf.summary()
    return results['scores'].loc["NF1"][0]


def adjusted_rand_index(first_partition, second_partition):
    """

    :param first_partition:
    :param second_partition:
    :return:
    """

    __check_partition_coverage(first_partition, second_partition)

    first_partition = [x[1]
                       for x in sorted([(node, nid)
                                        for nid, cluster in enumerate(first_partition)
                                        for node in cluster], key=lambda x: x[0])]

    second_partition = [x[1]
                        for x in sorted([(node, nid)
                                         for nid, cluster in enumerate(second_partition)
                                         for node in cluster], key=lambda x: x[0])]

    return sklearn.metrics.adjusted_rand_score(first_partition, second_partition)


def adjusted_mutual_information(first_partition, second_partition):
    """

    :param first_partition:
    :param second_partition:
    :return:
    """

    __check_partition_coverage(first_partition, second_partition)

    first_partition = [x[1]
                       for x in sorted([(node, nid)
                                        for nid, cluster in enumerate(first_partition)
                                        for node in cluster], key=lambda x: x[0])]

    second_partition = [x[1]
                        for x in sorted([(node, nid)
                                         for nid, cluster in enumerate(second_partition)
                                         for node in cluster], key=lambda x: x[0])]

    return sklearn.metrics.adjusted_mutual_info_score(first_partition, second_partition)


def variation_of_information(first_partition, second_partition):
    """

    Meila, M. (2007). Comparing clusterings - an information based distance.
    Journal of Multivariate Analysis, 98, 873-895. doi:10.1016/j.jmva.2006.11.013

    https://en.wikipedia.org/wiki/Variation_of_information

    :param first_partition:
    :param second_partition:
    :return:
    """

    __check_partition_coverage(first_partition, second_partition)

    n = float(sum([len(c1) for c1 in first_partition]))
    sigma = 0.0
    for c1 in first_partition:
        p = len(c1) / n
        for c2 in second_partition:
            q = len(c2) / n
            r = len(set(c1) & set(c2)) / n
            if r > 0.0:
                sigma += r * (np.log2(r / p) + np.log2(r / q))

    return abs(sigma)

