import json


class Clustering(object):
    @staticmethod
    def __convert_back_to_original_nodes_names_if_needed(communities: list) -> list:
        """
        If original nodes are int and we converted the graph to igraph, they were transformed to int. So we need to
        transform them back to int
        :return:
        """

        if (
            len(communities) > 0
            and not isinstance(communities[0], dict)
            and isinstance(list(communities[0])[0], str)
        ):
            if communities[0][0][:1] == "\\":
                to_return = []
                for com in communities:
                    to_return.append([int(x[1:]) for x in com])
                return to_return
        return communities

    def __init__(
        self,
        communities: list,
        graph: object,
        method_name: str = "",
        method_parameters: dict = None,
        overlap: bool = False,
    ):
        """
        Communities representation.

        :param communities: list of communities (community: list of nodes)
        :param method_name: algorithms discovery algorithm name
        :param overlap: boolean, whether the partition is overlapping or not
        """
        if isinstance(communities, set):
            communities = list(communities)
        communities = self.__convert_back_to_original_nodes_names_if_needed(communities)
        self.communities = sorted(communities, key=len, reverse=True)
        self.graph = graph
        self.method_name = method_name
        self.overlap = overlap
        self.method_parameters = method_parameters
        self.node_coverage = 1

    def to_json(self) -> str:
        """
        Generate a JSON representation of the algorithms object

        :return: a JSON formatted string representing the object
        """
        partition = {
            "communities": self.communities,
            "algorithm": self.method_name,
            "params": self.method_parameters,
            "overlap": self.overlap,
            "coverage": self.node_coverage,
        }

        return json.dumps(partition)

    def get_description(
        self, parameters_to_display: list = None, precision: int = 3
    ) -> str:
        """
        Return a description of the clustering, with the name of the method and its numeric parameters.

        :param parameters_to_display: parameters to display. By default, all float parameters.
        :param precision: precision used to plot parameters. default: 3
        :return: a string description of the method.
        """
        description = self.method_name

        # if no parameter name provided, display all parameters (or return directly description if none)

        if parameters_to_display is None:
            if self.method_parameters is not None:
                for p in self.method_parameters:
                    if self.method_parameters[p] is not None:
                        parameters_to_display = self.method_parameters.keys()
                if parameters_to_display is None:
                    return description
            else:
                return description

        description += "("
        # for each parameter, if it is a float, add it with the required precision.
        for p in parameters_to_display:
            if isinstance(self.method_parameters[p], float):
                description += p + ":"
                description += "{1:.{0}f}".format(precision, self.method_parameters[p])
                description += ", "
            elif isinstance(self.method_parameters[p], int):
                description += p + ":"
                description += "%s" % (self.method_parameters[p])
                description += ", "

        # handle formatting
        if description[-2:] == ", ":
            description = description[:-2]
        description += ")"
        return description
