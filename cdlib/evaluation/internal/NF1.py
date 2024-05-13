import numpy as np
from collections import defaultdict
import pandas as pd
import scipy


class NF1(object):
    def __init__(self, communities, ground_truth):
        self.matched_gt = {}
        self.gt_count = 0
        self.id_count = 0
        self.gt_nodes = {}
        self.id_nodes = {}
        self.communities = communities
        self.ground_truth = ground_truth
        self.prl = []
        self.__compute_precision_recall()

    def get_f1(self):
        """

        :return: a tuple composed by (average_f1, std_f1)
        """

        f1_list = np.array(self.__communities_f1())

        return (
            np.mean(f1_list),
            np.std(f1_list),
            np.max(f1_list),
            np.min(f1_list),
            scipy.stats.mode(f1_list, keepdims=True)[0][0],
        )

    def get_partition_stats(self):
        return (
            float(len(self.matched_gt)) / float(self.gt_count),  # coverage
            self.gt_count,
            self.id_count,
            float(self.id_count) / float(self.gt_count),
            float(len(self.id_nodes)) / float(len(self.gt_nodes)),
            float(self.id_count) / float(len(self.matched_gt)),  # redundancy
        )

    def __compute_precision_recall(self):
        """

        :return: a list of tuples (precision, recall)
        """

        # ground truth

        gt_coms = {cid: nodes for cid, nodes in enumerate(self.ground_truth)}
        node_to_com = defaultdict(list)
        for cid, nodes in gt_coms.items():
            for n in nodes:
                node_to_com[n].append(cid)
                if n not in self.gt_nodes:
                    self.gt_nodes[n] = True

        self.gt_count = len(gt_coms)

        # community
        ext_coms = {cid: nodes for cid, nodes in enumerate(self.communities)}
        prl = []

        for cid, nodes in ext_coms.items():
            ids = {}
            for n in nodes:
                if n not in self.id_nodes:
                    self.id_nodes[n] = True
                    try:
                        # community in ground truth
                        idd_list = node_to_com[n]
                        for idd in idd_list:
                            if idd not in ids:
                                ids[idd] = 1
                            else:
                                ids[idd] += 1
                    except KeyError:
                        pass
            try:
                # identify the maximal match ground truth communities (label) and their absolute frequency (p)
                maximal_match = {
                    label: p for label, p in ids.items() if p == max(ids.values())
                }

                for label, p in maximal_match.items():
                    if label not in self.matched_gt:
                        self.matched_gt[label] = True

                    precision = float(p) / len(nodes)
                    recall = float(p) / len(gt_coms[label])
                    prl.append((precision, recall))
            except (ZeroDivisionError, ValueError):
                pass

        self.id_count = len(ext_coms)
        self.prl = prl
        return prl

    @staticmethod
    def get_quality(f1, coverage, redundancy):
        """
        How much the communities are similar, how much the ground truth is covered
        and how many communities are needed

        :param f1:
        :param coverage:
        :param redundancy:
        :return:
        """
        return (f1 * coverage) / redundancy

    def __communities_f1(self):
        """

        :return: list of community f1 scores
        """

        f1s = []
        for l in self.prl:
            x, y = l[0], l[1]
            z = 2 * (x * y) / (x + y)
            z = float("%.2f" % z)
            f1s.append(z)
        return f1s

    def summary(self):

        mean, std, mx, mn, mode = self.get_f1()
        (
            coverage,
            gt_coms,
            id_coms,
            ratio_coms,
            node_coverage,
            redundancy,
        ) = self.get_partition_stats()
        quality = self.get_quality(mean, coverage, redundancy)

        m1 = [
            ["Ground Truth Communities", gt_coms],
            ["Identified Communities", id_coms],
            ["Community Ratio", ratio_coms],
            ["Ground Truth Matched", coverage],
            ["Node Coverage", node_coverage],
            ["NF1", quality],
        ]
        df = pd.DataFrame(m1, columns=["Index", "Value"])
        df.set_index("Index", inplace=True)

        m2 = [[mn, mx, mean, mode, std]]
        df2 = pd.DataFrame(
            m2,
            columns=[
                "F1 min",
                "F1 max",
                "F1 mean",
                "F1 mode",
                "F1 std",
            ],
        )

        result = {"scores": df, "details": df2}

        return result
