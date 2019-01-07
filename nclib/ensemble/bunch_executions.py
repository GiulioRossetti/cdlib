from collections import namedtuple
import itertools

Parameter = namedtuple("Parameter", ["name", "start", "end", "step"])
BoolParameter = namedtuple("BoolParameter", ["name"])


def __generate_ranges(parameter):
    """

    :param parameter:
    :return:
    """
    values = []
    if isinstance(parameter, Parameter):
        actual = parameter.start
        while actual < parameter.end:
            values.append((parameter.name, actual))
            actual += parameter.step
    elif isinstance(parameter, BoolParameter):
        values = [(parameter.name, True), (parameter.name, False)]
    else:
        raise ValueError("parameter should be either an instance of Parameter or of BoolParameter")
    return values


def grid_execution(method, graph, parameters):
    """

    :param method:
    :param graph:
    :param parameters:
    :return:
    """
    configurations = []
    for parameter in parameters:
        configurations.append(__generate_ranges(parameter))

    for element in itertools.product(*configurations):
        values = {e[0]: e[1] for e in element}
        res = method(graph, **values)
        yield values, res


def grid_search(method, graph, parameters, quality_score, aggregate=max):
    """

    :param method:
    :param graph:
    :param parameters:
    :param quality_score:
    :param aggregate:
    :return:
    """
    results = {}
    for param, communities in grid_execution(method, graph, parameters):
        key = tuple(sorted(param.items()))
        results[key] = {"communities": communities, 'scoring': quality_score(graph, communities)}

    res = aggregate(results, key=lambda x: results.get(x)['scoring'])
    return res, results[res]
