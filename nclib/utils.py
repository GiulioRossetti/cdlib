from contextlib import contextmanager
import igraph as ig
import sys
import os


@contextmanager
def suppress_stdout():
    """
    
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def from_nx_to_igraph(g, directed=False):
    """

    :param g:
    :param directed:
    :return:
    """
    gi = ig.Graph(directed=directed)
    gi.add_vertices(list(g.nodes()))
    gi.add_edges(list(g.edges()))
    return gi
