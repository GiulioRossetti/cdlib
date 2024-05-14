************
Input-Output
************

Functions to save/load ``cdlib`` communities to/from file.

----------
CSV format
----------

The easiest way to save the result of a community discovery algorithm is to organize it in a .csv file.
The following methods allow you to read/write communities to/from CSV.

.. automodule:: cdlib.readwrite


.. autosummary::
    :toctree: generated/

    read_community_csv
    write_community_csv

.. note:: CSV formatting allows only the saving/retrieving NodeClustering object to lose most of the metadata in the CD computation result - e.g., algorithm name, parameters, coverage...

-----------
JSON format
-----------

JSON format allows the storage/loading of community discovery algorithm results more comprehensively.

.. autosummary::
    :toctree: generated/

    read_community_json
    write_community_json

.. note:: JSON formatting allows only saving/retrieving all kinds of Clustering objects and maintaining all their metadata - except for the graph object instance.