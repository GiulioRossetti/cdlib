***********
Quick Start
***********

``CDlib`` is a python library that allows to extract, compare and evaluate network partitions.
We designed it to be agnostic w.r.t. the data structure used to represent the network to be clustered: all the algorithms it implements accept interchangeably igraph/networkx objects.

Of course, such a choice comes with advantages as well as drawbacks. Here's the main ones you have to be aware of:

**Advantages**
- Easy integration of existing/novel (python implementation of) CD algorithms;
- Standardization of input and output;
- Zero-configuration user interface (e.g., you don't have to reshape your data!)

**Drawbacks**
- Algorithms performances are not comparable (execution time, scalability... they all depends on how each algorithm was originally implemented);
- Memory (in)efficiency: depending by the type of structure each individual algorithm requires memory consumption could be high;
- Hidden transformation times: usually not a bottleneck, moving from a graph representation to another can take "some" time (usually linear in the graph size)

Most importantly: remember that i) each algorithm will be able to handle graphs up to a given size, and that ii) that maximum size that may vary greatly across the exposed algorithms.

--------
Tutorial
--------

Extracting communities using ``CDlib`` is easy as this:

.. code-block:: python

    from cdlib import algorithms
    import networkx as nx
    G = nx.karate_club_graph()
    coms = algorithms.louvain(G, weight='weight', resolution=1., randomize=False)

Of course, you can choose among all the algorithms available (taking care of specifying the correct parameters): in any case, you'll get as a result a Clustering object (or a more specific subclass).

Clustering objects expose a set of methods to perform evaluation and comparisons. For instance, to get the partition modularity just write

.. code-block:: python

    mod = coms.newman_girvan_modularity(g)

or, equivalently

.. code-block:: python

    from cdlib import evaluation
    mod = evaluation.newman_girvan_modularity(g,communities)

Moreover, you can also visualize networks and communities, plot indicators and similarity matrices... just take a look to the module reference to get a few examples.

I know, plain tutorials are overrated: if you want to explore ``CDlib`` functionalities, please start playing around with our interactive `Google Colab Notebook <https://colab.research.google.com/github/KDDComplexNetworkAnalysis/CNA_Tutorials/blob/master/CDlib_tutorial.ipynb>`_!

---
FAQ
---

**Q1.** I developed a novel Community Discovery algorithm/evaluation/visual analytics method and I would like to see it integrated in ``CDlib``. What should I do?

**A1.** That's great! Just open an issue on the project `GitHub <https://github.com/GiulioRossetti/cdlib>`_ briefly describing the method (provide a link to the paper where it has been firstly introduced) and links to a python implementation (if available). We'll came back to you as soon as possible to discuss the next steps.

**Q2.** Can you add method XXX to your library?

**A2.** It depends. Do you have a link to a python implementation/are you willing to help us in implementing it? If so, that's perfect. If not, well... everything is possible but it is likely that it will require some time.
