**********
Benchmarks
**********

Evaluating Community Detection algorithms on ground truth communities can be tricky when the annotation is based on external semantic information, not on topological ones.

For this reason, ``CDlib`` integrates synthetic network generators with planted community structures.


.. note::
    The following lists are aligned to CD evaluation methods available in the *GitHub main branch* of `CDlib`_.
    In particular, the following methods ara not yet available in the packaged version of the library: LFR, RDyn, GRP, PP, RPG, SBM, XMark.

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



.. _`CDlib`: https://github.com/GiulioRossetti/CDlib