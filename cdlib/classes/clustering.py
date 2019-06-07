import json


class Clustering(object):

    def __init__(self, communities, graph, method_name, method_parameters=None, overlap=False):
        """
        Communities representation.

        :param communities: list of communities
        :param method_name: algorithms discovery algorithm name
        :param overlap: boolean, whether the partition is overlapping or not
        """
        self.communities = sorted(communities, key=len, reverse=True)
        self.graph = graph
        self.method_name = method_name
        self.overlap = overlap
        self.method_parameters = method_parameters
        self.node_coverage = 1

    def to_json(self):
        """
        Generate a JSON representation of the algorithms object

        :return: a JSON formatted string representing the object
        """
        partition = {"communities": self.communities, "algorithm": self.method_name,
                     "params": self.method_parameters, "overlap": self.overlap, "coverage": self.node_coverage}

        return json.dumps(partition)

    def get_description(self, parameters_to_display=None, precision=3):
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
                    if self.method_parameters[p] != None:
                        parameters_to_display = self.method_parameters.keys()
                if parameters_to_display is None:
                    return  description
            else:
                return description


        description += "("
        # for each parameter, if it is a float, add it with the required precision.
        for p in parameters_to_display:
            if isinstance(self.method_parameters[p],float):
                description += p + ":"
                description+="{1:.{0}f}".format(precision,self.method_parameters[p])
                description+=", "
            elif isinstance(self.method_parameters[p],int):
                description += p + ":"
                description += "%s" %(self.method_parameters[p])
                description += ", "


        # handle formatting
        if description[-2:] == ", ":
            description = description[:-2]
        description += ")"
        return description
