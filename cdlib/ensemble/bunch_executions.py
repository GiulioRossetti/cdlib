from collections import namedtuple
import itertools
from random import sample

import networkx as nx
import numpy as np
import cdlib
from typing import Callable

__all__ = [
    "BoolParameter",
    "Parameter",
    "grid_execution",
    "grid_search",
    "pool",
    "pool_grid_filter",
    "random_search",
]

Parameter = namedtuple("Parameter", "name start end step")
Parameter.__new__.__defaults__ = (None,) * len(Parameter._fields)

BoolParameter = namedtuple("BoolParameter", "name value")
BoolParameter.__new__.__defaults__ = (None,) * len(BoolParameter._fields)


def __generate_ranges(parameter: tuple) -> list:
    """

    :param parameter:
    :return:
    """
    values = []
    if isinstance(parameter, cdlib.ensemble.Parameter):
        if parameter.step is None:
            values.append((parameter.name, parameter.start))
        else:
            for actual in np.arange(parameter.start, parameter.end, parameter.step):
                if isinstance(actual, np.int64):
                    actual = int(actual)
                values.append((parameter.name, actual))

    elif isinstance(parameter, BoolParameter):
        if parameter.value is None:
            values = [(parameter.name, True), (parameter.name, False)]
        else:
            values = [(parameter.name, parameter.value)]
    else:
        raise ValueError(
            "parameter should be either an instance of Parameter or of BoolParameter"
        )
    return values


def grid_execution(
    graph: nx.Graph, method: Callable[[nx.Graph, dict], object], parameters: list
) -> tuple:
    """
    Instantiate the specified community discovery method performing a grid search on the parameter set.

    :param method: community discovery method (from nclib.community)
    :param graph: networkx/igraph object
    :param parameters: list of Parameter and BoolParameter objects
    :return: at each call the generator yields a tuple composed by the current configuration and the obtained communities

    :Example:

    >>> import networkx as nx
    >>> from cdlib import algorithms, ensemble
    >>> g = nx.karate_club_graph()
    >>> resolution = ensemble.Parameter(name="resolution", start=0.1, end=1, step=0.1)
    >>> for communities in ensemble.grid_execution(graph=g, method=algorithms.louvain, parameters=[resolution]):
    >>>     print(communities)
    """
    configurations = []
    for parameter in parameters:
        configurations.append(__generate_ranges(parameter))

    for element in itertools.product(*configurations):
        values = {e[0]: e[1] for e in element}
        res = method(graph, **values)
        yield res


def grid_search(
    graph: nx.Graph,
    method: Callable[[nx.Graph, dict], object],
    parameters: list,
    quality_score: Callable[[nx.Graph, object], object],
    aggregate: Callable[[list], object] = max,
) -> tuple:
    """
    Returns the optimal partition of the specified graph w.r.t. the selected algorithm and quality score.

    :param method: community discovery method (from nclib.community)
    :param graph: networkx/igraph object
    :param parameters: list of Parameter and BoolParameter objects
    :param quality_score: a fitness function to evaluate the obtained partition (from nclib.evaluation)
    :param aggregate: function to select the best fitness value. Possible values: min/max
    :return: at each call the generator yields a tuple composed by: the optimal configuration for the given algorithm, input paramters and fitness function; the obtained communities; the fitness score

    :Example:

    >>> import networkx as nx
    >>> from cdlib import algorithms, ensemble
    >>> g = nx.karate_club_graph()
    >>> resolution = ensemble.Parameter(name="resolution", start=0.1, end=1, step=0.1)
    >>> randomize = ensemble.BoolParameter(name="randomize")
    >>> communities, scoring = ensemble.grid_search(graph=g, method=algorithms.louvain,
    >>>                                                     parameters=[resolution, randomize],
    >>>                                                     quality_score=evaluation.erdos_renyi_modularity,
    >>>                                                     aggregate=max)
    >>> print(communities, scoring)
    """
    results = {}
    for communities in grid_execution(graph, method, parameters):
        results[tuple(communities.method_parameters.items())] = {
            "communities": communities,
            "scoring": quality_score(graph, communities),
        }

    res = aggregate(results, key=lambda x: results.get(x)["scoring"])

    return results[res]["communities"], results[res]["scoring"]


def random_search(
    graph: nx.Graph,
    method: Callable[[nx.Graph, dict], object],
    parameters: list,
    quality_score: Callable[[nx.Graph, object], object],
    instances: int = 10,
    aggregate: Callable[[list], object] = max,
) -> tuple:
    """
    Returns the optimal partition of the specified graph w.r.t. the selected algorithm and quality score over a randomized sample of the input parameters.

    :param method: community discovery method (from nclib.community)
    :param graph: networkx/igraph object
    :param parameters: list of Parameter and BoolParameter objects
    :param quality_score: a fitness function to evaluate the obtained partition (from nclib.evaluation)
    :param instances: number of randomly selected parameters configurations
    :param aggregate: function to select the best fitness value. Possible values: min/max

    :return: at each call the generator yields a tuple composed by: the optimal configuration for the given algorithm, input paramters and fitness function; the obtained communities; the fitness score

    :Example:

    >>> import networkx as nx
    >>> from cdlib import algorithms, ensemble
    >>> g = nx.karate_club_graph()
    >>> resolution = ensemble.Parameter(name="resolution", start=0.1, end=1, step=0.1)
    >>> randomize = ensemble.BoolParameter(name="randomize")
    >>> communities, scoring = ensemble.random_search(graph=g, method=algorithms.louvain,
    >>>                                                       parameters=[resolution, randomize],
    >>>                                                       quality_score=evaluation.erdos_renyi_modularity,
    >>>                                                       instances=5, aggregate=max)
    >>> print(communities, scoring)
    """

    configurations = []
    for parameter in parameters:
        configurations.append(__generate_ranges(parameter))

    # configuration sampling
    selected = sample(list(itertools.product(*configurations)), instances)

    results = {}
    for element in selected:
        values = {e[0]: e[1] for e in element}
        communities = method(graph, **values)
        results[element] = {
            "communities": communities,
            "scoring": quality_score(graph, communities),
        }

    res = aggregate(results, key=lambda x: results.get(x)["scoring"])

    return results[res]["communities"], results[res]["scoring"]


def pool(
    graph: nx.Graph, methods: Callable[[nx.Graph, dict], object], configurations: list
) -> tuple:
    """
    Execute on a pool of community discovery internal on the input graph.

    :param methods: list community discovery methods (from nclib.community)
    :param graph: networkx/igraph object
    :param configurations: list of lists (one for each method) of Parameter and BoolParameter objects
    :return: at each call the generator yields a tuple composed by: the actual method, its current configuration and the obtained communities
    :raises ValueError: if the number of methods is different from the number of configurations specified

    :Example:

    >>> import networkx as nx
    >>> from cdlib import algorithms, ensemble
    >>> g = nx.karate_club_graph()
    >>> # Louvain
    >>> resolution = ensemble.Parameter(name="resolution", start=0.1, end=1, step=0.1)
    >>> randomize = ensemble.BoolParameter(name="randomize")
    >>> louvain_conf = [resolution, randomize]
    >>>
    >>> # Angel
    >>> threshold = ensemble.Parameter(name="threshold", start=0.1, end=1, step=0.1)
    >>> angel_conf = [threshold]
    >>>
    >>> methods = [algorithms.louvain, algorithms.angel]
    >>>
    >>> for communities in ensemble.pool(g, methods, [louvain_conf, angel_conf]):
    >>>     print(communities)
    """
    if len(methods) != len(configurations):
        raise ValueError("The number of methods and configurations must match")

    for i in range(len(methods)):
        for res in grid_execution(graph, methods[i], configurations[i]):
            yield res


def pool_grid_filter(
    graph: nx.Graph,
    methods: Callable[[nx.Graph, dict], object],
    configurations: list,
    quality_score: Callable[[nx.Graph, object], object],
    aggregate: Callable[[list], object] = max,
) -> tuple:
    """
    Execute a pool of community discovery internal on the input graph.
    Returns the optimal partition for each algorithm given the specified quality function.

    :param methods: list community discovery methods (from nclib.community)
    :param graph: networkx/igraph object
    :param configurations: list of lists (one for each method) of Parameter and BoolParameter objects
    :param quality_score: a fitness function to evaluate the obtained partition (from nclib.evaluation)
    :param aggregate: function to select the best fitness value. Possible values: min/max
    :return: at each call the generator yields a tuple composed by: the actual method, its optimal configuration; the obtained communities; the fitness score.
    :raises ValueError: if the number of methods is different from the number of configurations specified

    :Example:

    >>> import networkx as nx
    >>> from cdlib import algorithms, ensemble
    >>> g = nx.karate_club_graph()
    >>> # Louvain
    >>> resolution = ensemble.Parameter(name="resolution", start=0.1, end=1, step=0.1)
    >>> randomize = ensemble.BoolParameter(name="randomize")
    >>> louvain_conf = [resolution, randomize]
    >>>
    >>> # Angel
    >>> threshold = ensemble.Parameter(name="threshold", start=0.1, end=1, step=0.1)
    >>> angel_conf = [threshold]
    >>>
    >>> methods = [algorithms.louvain, algorithms.angel]
    >>>
    >>> for communities, scoring in ensemble.pool_grid_filter(g, methods, [louvain_conf, angel_conf], quality_score=evaluation.erdos_renyi_modularity, aggregate=max):
    >>>     print(communities, scoring)

    """
    if len(methods) != len(configurations):
        raise ValueError("The number of methods and configurations must match")

    for i in range(len(methods)):
        communities, scoring = grid_search(
            graph, methods[i], configurations[i], quality_score, aggregate
        )
        yield communities, scoring
