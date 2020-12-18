***************
Remote Datasets
***************

``cdlib`` allows to retrieve existing datasets, along with their ground truth partitions (if available), from an ad-hoc remote `repository`_.

.. note::
    The following features are still under testing: therefore, they are accessible only on the *GitHub* version of the library.


.. automodule:: cdlib.datasets


.. autosummary::
    :toctree: generated/

    available_networks
    available_ground_truths
    fetch_network_data
    fetch_ground_truth_data
    fetch_network_ground_truth


.. _`repository`: https://github.com/GiulioRossetti/cdlib_datasets