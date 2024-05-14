***********
Quick Start
***********

``CDlib`` is a Python library that allows network partition extraction, comparison, and evaluation.
We designed it to be agnostic w.r.t. the data structure used to represent the network to be clustered: all the algorithms it implements accept interchangeably igraph/networkx objects.

Of course, such a choice comes with advantages as well as drawbacks. Here are the main ones you have to be aware of:

**Advantages**
- Easy integration of existing/novel (python implementation of) CD algorithms;
- Standardization of input and output;
- Zero-configuration user interface (e.g., you do not have to reshape your data!)

**Drawbacks**
- Algorithm performances are not comparable (execution time, scalability... they all depend on how each algorithm was originally implemented);
- Memory (in)efficiency: Depending by the type of structure each algorithm requires, memory consumption could be high;
- Hidden transformation times: usually not a bottleneck, moving from a graph representation to another can take "some" time (usually linear in the graph size)

Most importantly, remember that i) each algorithm will be able to handle graphs up to a given size, and ii) that maximum size may vary greatly across the exposed algorithms.

--------
Tutorial
--------

Extracting communities using ``CDlib`` is easy as:

.. code-block:: python

    from cdlib import algorithms
    import networkx as nx
    G = nx.karate_club_graph()
    coms = algorithms.louvain(G, weight='weight', resolution=1., randomize=False)

Of course, you can choose among all the algorithms available (taking care of specifying the correct parameters). As a result, you will get a Clustering object (or a more specific subclass).

Clustering objects exposes a set of methods to perform evaluation and comparisons. For instance, to get the partition modularity, write:

.. code-block:: python

    mod = coms.newman_girvan_modularity(g)

or, equivalently

.. code-block:: python

    from cdlib import evaluation
    mod = evaluation.newman_girvan_modularity(g,communities)

Moreover, you can also visualize networks and communities, plot indicators, and similarity matrices... take a look at the module reference to get a few examples.

I know plain tutorials are overrated: if you want to explore ``CDlib`` functionalities, please start playing around with our interactive `Google Colab Notebook <https://colab.research.google.com/github/KDDComplexNetworkAnalysis/CNA_Tutorials/blob/master/CDlib_tutorial.ipynb>`_ !

---
FAQ
---

**Q1.** I developed a novel Community Discovery algorithm/evaluation/visual analytics method and would like to see it integrated into ``CDlib``. What should I do?

**A1.** That is great! Just open an issue on the project `GitHub <https://github.com/GiulioRossetti/cdlib>`_ briefly describing the method (provide a link to the paper where it was first introduced) and links to a Python implementation (if available). We will return to you soon to discuss the next steps.

**Q2.** Can you add method XXX to your library?

**A2.** It depends. Do you have a link to a Python implementation, or are you willing to help us implement it? If so, that is perfect. If not, everything is possible, but it will likely require some time.