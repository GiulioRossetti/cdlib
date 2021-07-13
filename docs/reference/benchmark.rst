********************
Synthetic Benchmarks
********************

Evaluating Community Detection algorithms on ground truth communities can be tricky when the annotation is based on external semantic information, not on topological ones.

For this reason, ``cdlib`` integrates synthetic network generators with planted community structures.


.. note::
    The following lists are aligned to CD evaluation methods available in the *GitHub main branch* of `cdlib`_.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Static Networks with Community Ground Truth
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Benchmarks for plain static networks.
All generators return a tuple: (``networkx.Graph``, ``cdlib.NodeClustering``)


.. automodule:: cdlib.benchmark


.. autosummary::
    :toctree: bench/

    GRP
    LFR
    PP
    RPG
    SBM

Benchmarks for node-attributed static networks.

.. autosummary::
    :toctree: bench/

    XMark


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Dynamic Networks with Community Ground Truth
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Time evolving network topologies with planted community life-cycles.
All generators return a tuple: (``dynetx.DynGraph``, ``cdlib.TemporalClustering``)

.. autosummary::
    :toctree: bench/

    RDyn



.. _`cdlib`: https://github.com/GiulioRossetti/cdlib