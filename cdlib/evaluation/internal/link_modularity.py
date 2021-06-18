import numpy as np
import math
import functools

# https://github.com/RapidsAtHKUST/CommunityDetectionCodes


class FuncTag:
    def __init__(self):
        pass

    exp_inv_mul_tag = "exp_inv_mul"
    mul_tag = "mul"
    min_tag = "min"
    max_tag = "max"


def get_coefficient_func(tag: str) -> float:
    if tag == FuncTag.exp_inv_mul_tag:
        return lambda l, r: 1.0 / functools.reduce(
            lambda il, ir: il * ir, map(lambda ele: 1.0 + math.exp(2 - ele), [l, r]), 1
        )
    elif tag == FuncTag.mul_tag:
        return lambda l, r: l * r
    elif tag == FuncTag.min_tag:
        return min
    elif tag == FuncTag.max_tag:
        return max


def cal_modularity(input_graph, comm_result):
    return LinkBelongModularity(
        input_graph, comm_result, get_coefficient_func(FuncTag.exp_inv_mul_tag)
    ).calculate_modularity()


class LinkBelongModularity:
    PRECISION = 0.0001

    def __init__(self, input_graph, comm_result, coefficient_func):
        """
        :type input_graph: nx.Graph
        """
        self.comm_list = comm_result
        self.graph = input_graph
        self.coefficient_func = coefficient_func
        self.belong_weight_dict = {}
        self.in_degree_dict = {}
        self.out_degree_dict = {}

        self.init_belong_weight_dict()
        self.init_degree_dicts()

    def init_belong_weight_dict(self):
        belong_dict = {}
        for comm in self.comm_list:
            for mem in comm:
                if mem not in belong_dict:
                    belong_dict[mem] = 0
                belong_dict[mem] += 1
        for mem in belong_dict:
            self.belong_weight_dict[mem] = (
                1.0 / belong_dict[mem] if belong_dict[mem] != 0 else 0
            )

    def init_degree_dicts(self):
        for vertex in self.graph.nodes():
            # since graph here studied are used in undirected manner
            self.in_degree_dict[vertex] = self.graph.degree(vertex)
            self.out_degree_dict[vertex] = self.graph.degree(vertex)

    def calculate_modularity(self) -> float:
        modularity_val = 0
        vertex_num = self.graph.number_of_nodes()
        edge_num = self.graph.number_of_edges()
        for comm in self.comm_list:
            comm_size = len(comm)
            f_val_matrix = np.ndarray(shape=(comm_size, comm_size), dtype=float)
            f_val_matrix.fill(0)
            f_sum_in_vec = np.zeros(comm_size, dtype=float)
            f_sum_out_vec = np.zeros(comm_size, dtype=float)
            in_deg_vec = np.zeros(comm_size, dtype=float)
            out_deg_vec = np.zeros(comm_size, dtype=float)

            # calculate f_val_matrix, f_sum_in, f_sum_out
            for i in range(comm_size):
                src_mem = comm[i]
                in_deg_vec[i] = self.in_degree_dict[src_mem]
                out_deg_vec[i] = self.out_degree_dict[src_mem]
                for j in range(comm_size):
                    dst_mem = comm[j]
                    if i != j and self.graph.has_edge(src_mem, dst_mem):
                        f_val_matrix[i][j] = self.coefficient_func(
                            self.belong_weight_dict[src_mem],
                            self.belong_weight_dict[dst_mem],
                        )
                        f_sum_out_vec[i] += f_val_matrix[i][j]
                        f_sum_in_vec[j] += f_val_matrix[i][j]

            f_sum_in_vec /= vertex_num
            f_sum_out_vec /= vertex_num

            for i in range(comm_size):
                for j in range(comm_size):
                    if i != j and f_val_matrix[i][j] > LinkBelongModularity.PRECISION:
                        null_model_val = (
                            out_deg_vec[i]
                            * in_deg_vec[j]
                            * f_sum_out_vec[i]
                            * f_sum_in_vec[j]
                            / edge_num
                        )
                        modularity_val += f_val_matrix[i][j] - null_model_val
        modularity_val /= edge_num
        return modularity_val
