import pkg_resources
import pooch
from cdlib.readwrite import *
import networkx as nx
try:
    import igraph as ig
except ModuleNotFoundError:
    ig = None

__all__ = ["available_networks", "available_ground_truths", "fetch_network_data", "fetch_ground_truth_data",
           "fetch_network_ground_truth"]

__networks = pooch.create(
    path=pooch.os_cache("cdlib"),
    base_url="https://github.com/GiulioRossetti/cdlib_datasets/raw/main/networks/",
    version_dev="master",
    env="cdlib_data_dir",
    registry={"karate_club.csv.gz": "0bac1a017e59f505f073aef6dc1783c03826a3fd4f38ffc6a50fde582948e2f2",
              "dblp.csv.gz": "6b64a23e60083fa52d64320a2cd4366ff1b33d8a89ac4fa1b30f159e88c1730c",
              "amazon.csv.gz": "aea015386f62b21ba31e097553a298fb3df610ec8dc722addb06a1a59e646cd3",
              }
)

__ground_truths = pooch.create(
    path=pooch.os_cache("cdlib"),
    base_url="https://github.com/GiulioRossetti/cdlib_datasets/raw/main/ground_truth/",
    version_dev="master",
    env="cdlib_data_dir",
    registry={"karate_club.json.gz": "198fd42c3df9ab49e3eea5932f0d6e4cceac25db147c5108e0f8e9a4c55e11b7",
              "dblp.json.gz": "ca7dba98bd3bdc76999fd2991d1667b7b531e8bac777986247a8dcac302c085d",
              "amazon.json.gz": "c6a03909f2b14082523108be13616e5b24cfe830b96e416d784e90ab23d12bd7",
              }
)


def available_networks():
    """
    List the remotely available network datasets.

    :return: list of network names

    :Example:

    >>> from cdlib import datasets
    >>> import networkx as nx
    >>> graph_name_list = datasets.available_networks()

    """
    return [x.split(".")[0] for x in __networks.registry.keys()]


def available_ground_truths():
    """
    List the remotely available network ground truth datasets.

    :return: list of network names

    :Example:

    >>> from cdlib import datasets
    >>> import networkx as nx
    >>> graph_name_list = datasets.available_ground_truths()

    """
    return [x.split(".")[0] for x in __ground_truths.registry.keys()]


def fetch_network_data(net_name="karate_club", net_type="igraph"):
    """
    Load the required network from the remote repository

    :param net_name: network name
    :param net_type: desired graph object among "networkx" and "igraph". Default, igraph.
    :return: a graph object

    :Example:

    >>> from cdlib import datasets
    >>> import networkx as nx
    >>> G = datasets.fetch_network_data(net_name="karate_club", net_type="igraph")

    """

    download = pooch.HTTPDownloader(progressbar=True)
    fname = __networks.fetch(f"{net_name}.csv.gz", processor=pooch.Decompress(), downloader=download)

    if net_type == "networkx":
        g = nx.Graph()
        with open(fname) as f:
            for l in f:
                l = l.replace(" ", "\t").split("\t")
                g.add_edge(int(l[0]), int(l[1]))
    else:
        if ig is None:
            raise ModuleNotFoundError("Optional dependency not satisfied: install python-igraph to use the selected "
                                      "feature.")

        edges = []
        with open(fname) as f:
            for l in f:
                l = l.replace(" ", "\t").split("\t")
                edges.append((int(l[0]), int(l[1])))
        g = ig.Graph.TupleList(edges)

    return g


def fetch_ground_truth_data(net_name="karate_club", graph=None):
    """
    Load the required ground truth clustering from the remote repository

    :param net_name: network name
    :param graph: the graph object associated to the ground truth (optional)
    :return: a NodeClustering object

    :Example:

    >>> from cdlib import datasets
    >>> import networkx as nx
    >>> gt_coms = datasets.fetch_network_data(fetch_ground_truth_data="karate_club")

    """

    download = pooch.HTTPDownloader(progressbar=True)
    fname = __ground_truths.fetch(f"{net_name}.json.gz", processor=pooch.Decompress(), downloader=download)
    gt = read_community_json(fname)
    if graph is not None:
        gt.graph = graph
    return gt


def fetch_network_ground_truth(net_name="karate_club", net_type="igraph"):
    """
    Load the required network, along with its ground truth partition, from the remote repository.

    :param net_name: network name
    :param net_type: desired graph object among "networkx" and "igraph". Default, igraph.
    :return: a tuple of (graph_object, NodeClustering)

    :Example:

    >>> from cdlib import datasets
    >>> import networkx as nx
    >>> G, gt_coms = datasets.fetch_network_ground_truth(fetch_ground_truth_data="karate_club", net_type="igraph")

    """

    if net_name not in available_networks() or net_name not in available_ground_truths():
        raise ValueError(f"{net_name} is not present in the remote repository")

    g = fetch_network_data(net_name, net_type)
    gt = fetch_ground_truth_data(net_name, g)
    return g, gt
