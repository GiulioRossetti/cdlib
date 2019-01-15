import numpy as np

# https://github.com/RapidsAtHKUST/CommunityDetectionCodes


def calc_overlap_nmi(num_vertices, result_comm_list, ground_truth_comm_list):
    return OverlapNMI(num_vertices, result_comm_list, ground_truth_comm_list).calculate_overlap_nmi()


class OverlapNMI(object):

    @staticmethod
    def entropy(num):
        if num == 0:
            return 0
        lg = np.log2(num)
        return -num * lg

    def __init__(self, num_vertices, result_comm_list, ground_truth_comm_list):
        self.x_comm_list = result_comm_list
        self.y_comm_list = ground_truth_comm_list
        self.num_vertices = num_vertices

    def calculate_overlap_nmi(self):

        def get_cap_x_given_cap_y(cap_x, cap_y):
            def get_joint_distribution(cx, cy):
                prob_matrix = np.ndarray(shape=(2, 2), dtype=float)
                intersect_size = float(len(set(cx) & set(cy)))
                cap_n = self.num_vertices + 4
                prob_matrix[1][1] = (intersect_size + 1) / cap_n
                prob_matrix[1][0] = (len(cx) - intersect_size + 1) / cap_n
                prob_matrix[0][1] = (len(cy) - intersect_size + 1) / cap_n
                prob_matrix[0][0] = (self.num_vertices - intersect_size + 1) / cap_n
                return prob_matrix

            def get_single_distribution(comm):
                prob_arr = [0] * 2
                prob_arr[1] = float(len(comm)) / self.num_vertices
                prob_arr[0] = 1 - prob_arr[1]
                return prob_arr

            def get_cond_entropy(cx, cy):
                prob_matrix = get_joint_distribution(cx, cy)
                entropy_list = list(map(OverlapNMI.entropy,
                                        (prob_matrix[0][0], prob_matrix[0][1], prob_matrix[1][0], prob_matrix[1][1])))
                if entropy_list[3] + entropy_list[0] <= entropy_list[1] + entropy_list[2]:
                    return np.inf
                else:
                    prob_arr_y = get_single_distribution(cy)
                    return sum(entropy_list) - sum(list(map(OverlapNMI.entropy, prob_arr_y)))

            partial_res_list = []
            for comm_x in cap_x:
                cond_entropy_list = list(map(lambda comm_y: get_cond_entropy(comm_x, comm_y), cap_y))
                min_cond_entropy = float(min(cond_entropy_list))
                partial_res_list.append(
                    min_cond_entropy / sum(list(map(OverlapNMI.entropy, get_single_distribution(comm_x)))))
            return np.mean(partial_res_list)

        return 1 - 0.5 * get_cap_x_given_cap_y(self.x_comm_list, self.y_comm_list) - 0.5 * get_cap_x_given_cap_y(
            self.y_comm_list, self.x_comm_list)
