*****************
Community Objects
*****************

``cdlib`` aims to standardize the representation of network communities.
To fulfill such a goal, several Clustering classes are introduced, each capturing specific community characteristics.
All classes inherit from the same interface, thus sharing some common functionalities.

In particular, ``cdlib`` algorithms can output the following Clustering types:

- **NodeClustering**: Node communities (either crisp partitions or overlapping groups);
- **FuzzyNodeClustering**: Overlapping node communities with explicit node-to-community belonging score;
- **BiNodeClustering**: Clustering of Bipartite graphs (with the explicit representation of class homogeneous communities);
- **AttrNodeClustering**: Clustering of feature-rich (node-attributed) graphs;
- **EdgeClustering**: Edge communities;
- **TemporalClustering**: Clustering of Temporal Networks;

Refer to the following documentation for a complete overview of the methods exposed by ``cdlib`` clustering objects.

.. toctree::
    :maxdepth: 1

    classes/node_clustering.rst
    classes/fuzzy_node_clustering.rst
    classes/attr_node_clustering.rst
    classes/bi_node_clustering.rst
    classes/edge_clustering.rst
    classes/temporal_clustering.rst

.. note::

    I have a clustering obtained by an algorithm not included in ``CDlib``. Can I load it in a Clustering object to leverage your library's evaluation and visualization facilities?

Yes, you can.

Just transform your clustering into a list of lists (we represent each community as a list of node IDs) and then create a NodeClustering (or any other Clustering) object.

.. code-block:: python

    from cdlib import NodeClustering

    communities = [[1,2,3], [4,5,6], [7,8,9,10,11]]
    coms = NodeClustering(communities, graph=None, method_name="your_method")

Of course, to compute some evaluation scores/plot community networks, you will also have to pass the original graph (as igraph/networkx object) while building the NodeClustering instance.

