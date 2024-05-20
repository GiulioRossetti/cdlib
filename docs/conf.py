# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import sys, os
import sphinx_rtd_theme

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from cdlib import __version__
except ImportError:
    __version__ = "0.4.0"

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

version = __version__
# The full version, including alpha/beta/rc tags.
release = version

html_theme_options = {
    "collapse_navigation": False,
    "display_version": False,
    "navigation_depth": 3,
}

# -- Project information -----------------------------------------------------

project = "CDlib"
copyright = "2024, Giulio Rossetti"
author = "Giulio Rossetti"

autodoc_mock_imports = [
    "graph_tool.all",
    "graph_tool",
    "thresholdclustering.thresholdclustering",
    "thresholdclustering",
    "igraph.drawing",
    "igraph",
    "gurobipy",
    "dynetx",
    "GraphRicciCurvature.OllivierRicci",
    "networkx.algorithms",
    "pycombo",
    "pybind11",
    "ASLPAw_package",
    "ipaddress",
    "ASLPAw",
    "graph-tool",
    "leidenalg",
    "numpy",
    "scipy",
    "networkx",
    "bimlpa",
    "sklearn",
    "pquality",
    "functools",
    "nf1",
    "ipython",
    "pygtk",
    "gtk",
    "gobject",
    "argparse",
    "matplotlib",
    "matplotlib.pyplot",
    "scikit-learn",
    "python-igraph",
    "wurlitzer",
    "pulp",
    "seaborn",
    "pandas",
    "infomap",
    "angel-cd",
    "omega_index_py3",
    "markov_clustering",
    "scipy.sparse",
    "pyclustering",
    "pyclustering.cluster.kmedoids",
    "sklearn.preprocessing",
    "sklearn.cluster",
    "python-Levenshtein",
    "shuffle_graph",
    "similarity-index-of-label-graph",
    "gmpy2",
    "gurobipy",
    "bayanpy",
    "clusim",
    "scipy.stats",
    "clusim.sim",
    "clusim.clustering",
    "plotly",
    "plotly.graph_objects",
]

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'

html_logo = "cdlib_new.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
