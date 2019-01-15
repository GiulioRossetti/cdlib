from collections import namedtuple
import itertools
from random import sample

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


def grid_execution(graph, method, parameters):
    """
    Instantiate the specified community discovery method performing a grid search on the parameter set.

    :param method: community discovery method (from nclib.community)
    :param graph: networkx/igraph object
    :param parameters: list of Parameter and BoolParameter objects
    :return: at each call the generator yields a tuple composed by the current configuration and the obtained communities

    :Example:

    >>> import networkx as nx
    >>> from nclib import community, ensemble
    >>> g = nx.karate_club_graph()
    >>> resolution = ensemble.Parameter(name="resolution", start=0.1, end=1, step=0.1)
    >>> for params, communities in ensemble.grid_execution(graph=g, method=community.louvain, parameters=[resolution]):
    >>>     print(params, communities)
    """
    configurations = []
    for parameter in parameters:
        configurations.append(__generate_ranges(parameter))

    for element in itertools.product(*configurations):
        values = {e[0]: e[1] for e in element}
        res = method(graph, **values)
        yield element, res


def grid_search(graph, method, parameters, quality_score, aggregate=max):
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
    >>> from nclib import community, ensemble
    >>> g = nx.karate_club_graph()
    >>> resolution = ensemble.Parameter(name="resolution", start=0.1, end=1, step=0.1)
    >>> randomize = ensemble.BoolParameter(name="randomize")
    >>> params, communities, scoring = ensemble.grid_search(graph=g, method=community.louvain,
    >>>                                                     parameters=[resolution, randomize],
    >>>                                                     quality_score=evaluation.erdos_renyi_modularity,
    >>>                                                     aggregate=max)
    >>>     print(params, communities, scoring)
    """
    results = {}
    for param, communities in grid_execution(graph, method, parameters):
        results[param] = {"communities": communities, 'scoring': quality_score(graph, communities)}

    res = aggregate(results, key=lambda x: results.get(x)['scoring'])

    return res, results[res]['communities'], results[res]['scoring']


def random_search(graph, method, parameters, quality_score, instances=10, aggregate=max):
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
    >>> from nclib import community, ensemble
    >>> g = nx.karate_club_graph()
    >>> resolution = ensemble.Parameter(name="resolution", start=0.1, end=1, step=0.1)
    >>> randomize = ensemble.BoolParameter(name="randomize")
    >>> params, communities, scoring = ensemble.random_search(graph=g, method=community.louvain,
    >>>                                                       parameters=[resolution, randomize],
    >>>                                                       quality_score=evaluation.erdos_renyi_modularity,
    >>>                                                       instances=5, aggregate=max)
    >>>     print(params, communities, scoring)
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
        results[element] = {"communities": communities, 'scoring': quality_score(graph, communities)}

    res = aggregate(results, key=lambda x: results.get(x)['scoring'])

    return res, results[res]['communities'], results[res]['scoring']


def pool(graph, methods, configurations):
    """
    Execute on a pool of community discovery algorithms on the input graph.
    
    :param methods: list community discovery methods (from nclib.community)
    :param graph: networkx/igraph object
    :param configurations: list of lists (one for each method) of Parameter and BoolParameter objects
    :return: at each call the generator yields a tuple composed by: the actual method, its current configuration and the obtained communities
    :raises ValueError: if the number of methods is different from the number of configurations specified

    :Example:

    >>> import networkx as nx
    >>> from nclib import community, ensemble
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
    >>> methods = [community.louvain, community.angel]
    >>>
    >>> for method, parameters, communities in ensemble.pool(g, methods, [louvain_conf, angel_conf]):
    >>>     print(method, parameters, communities)
    """
    if len(methods) != len(configurations):
        raise ValueError("The number of methods and configurations must match")

    for i in range(len(methods)):
        for values, res in grid_execution(graph, methods[i], configurations[i]):
            yield methods[i].__name__, values, res


def pool_grid_filter(graph, methods, configurations, quality_score, aggregate=max):
    """
    Execute a pool of community discovery algorithms on the input graph.
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
    >>> from nclib import community, ensemble
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
    >>> methods = [community.louvain, community.angel]
    >>>
    >>> for method, parameters, communities, scoring in ensemble.pool_grid_filter(g, methods, [louvain_conf, angel_conf], quality_score=evaluation.erdos_renyi_modularity, aggregate=max):
    >>>     print(method, parameters, communities, scoring)

    """
    if len(methods) != len(configurations):
        raise ValueError("The number of methods and configurations must match")

    for i in range(len(methods)):
        values, communities, scoring = grid_search(graph, methods[i], configurations[i], quality_score, aggregate)
        yield methods[i].__name__, values, communities, scoring
