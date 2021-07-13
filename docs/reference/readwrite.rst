************
Input-Output
************

Functions to save/load ``cdlib`` communities to/from file.

----------
CSV format
----------

The easiest way to save the result of a community discovery algorithm is to organize it in a .csv file.
The following methods allows to read/write communities to/from csv.

.. automodule:: cdlib.readwrite


.. autosummary::
    :toctree: generated/

    read_community_csv
    write_community_csv

.. note:: CSV formatting allows only to save/retrieve NodeClustering object loosing most of the metadata present in the CD computation result - e.g., algorithm name, parameters, coverage...

-----------
JSON format
-----------

JSON format allows to store/load community discovery algorithm results in a more comprehensive way.

.. autosummary::
    :toctree: generated/

    read_community_json
    write_community_json

.. note:: JSON formatting allows only to save/retrieve all kind of Clustering object maintaining all their metadata - except for the graph object instance.