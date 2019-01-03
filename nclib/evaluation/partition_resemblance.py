from nclib.evaluation.scoring_functions import onmi
from omega_index import Omega
from nf1 import NF1
from sklearn.metrics import normalized_mutual_info_score


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
