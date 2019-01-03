import numpy as np
from nclib.evaluation.scoring_functions import onmi
from omega_index import Omega
from nf1 import NF1
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score


def nmi(first_partition, second_partition):
    """

    :param first_partition:
    :param second_partition:
    :return:
    """

    first_partition = [x[1]
                       for x in sorted([(node, nid)
                                        for nid, cluster in enumerate(first_partition)
                                        for node in cluster], key=lambda x: x[0])]

    second_partition = [x[1]
                       for x in sorted([(node, nid)
                                        for nid, cluster in enumerate(second_partition)
                                        for node in cluster], key=lambda x: x[0])]

    return normalized_mutual_info_score(first_partition, second_partition)


def overlapping_nmi(first_partition, second_partition):
    """

    :param first_partition:
    :param second_partition:
    :return:
    """

    vertex_number = len({node: None for community in first_partition for node in community})
    return onmi.calc_overlap_nmi(vertex_number, first_partition, second_partition)


def omega(first_partition, second_partition):
    """

    :param first_partition:
    :param second_partition:
    :return:
    """

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
    return results['details']['F1 mean'], results['details']['F1 std']


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

    first_partition = [x[1]
                       for x in sorted([(node, nid)
                                        for nid, cluster in enumerate(first_partition)
                                        for node in cluster], key=lambda x: x[0])]

    second_partition = [x[1]
                        for x in sorted([(node, nid)
                                         for nid, cluster in enumerate(second_partition)
                                         for node in cluster], key=lambda x: x[0])]

    return adjusted_rand_score(first_partition, second_partition)


def adjusted_mutual_information(first_partition, second_partition):
    """

    :param first_partition:
    :param second_partition:
    :return:
    """

    first_partition = [x[1]
                       for x in sorted([(node, nid)
                                        for nid, cluster in enumerate(first_partition)
                                        for node in cluster], key=lambda x: x[0])]

    second_partition = [x[1]
                        for x in sorted([(node, nid)
                                         for nid, cluster in enumerate(second_partition)
                                         for node in cluster], key=lambda x: x[0])]

    return adjusted_mutual_info_score(first_partition, second_partition)


def variation_of_information(first_partition, second_partition):
    """

    Meila, M. (2007). Comparing clusterings - an information based distance.
    Journal of Multivariate Analysis, 98, 873-895. doi:10.1016/j.jmva.2006.11.013

    https://en.wikipedia.org/wiki/Variation_of_information

    :param first_partition:
    :param second_partition:
    :return:
    """

    n = float(sum([len(c1) for c1 in first_partition]))
    sigma = 0.0
    for c1 in first_partition:
        p = len(c1) / n
        for c2 in second_partition:
            q = len(c2) / n
            r = len(set(c1) & set(c2)) / n
            if r > 0.0:
                sigma += r * (np.log(r / p, 2) + np.log(r / q, 2))

    return abs(sigma)


def normalized_variation_of_information(first_partition, second_partition):
    """

    :param first_partition:
    :param second_partition:
    :return:
    """

    return 1 - adjusted_mutual_info_score(first_partition, second_partition)