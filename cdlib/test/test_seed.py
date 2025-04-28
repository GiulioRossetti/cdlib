import unittest
import networkx as nx
from cdlib import algorithms, seed, reset_seed, get_seed, fixed_seed


class TestSeedSetting(unittest.TestCase):

    def setUp(self):
        self.graph = nx.karate_club_graph()

    def test_leiden_seed(self):
        seed(42)
        comms1 = algorithms.leiden(self.graph)
        seed(42)
        comms2 = algorithms.leiden(self.graph)
        self.assertEqual(comms1.communities, comms2.communities)

    def test_infomap_seed(self):
        seed(123)
        comms1 = algorithms.infomap(self.graph)
        seed(123)
        comms2 = algorithms.infomap(self.graph)
        self.assertEqual(comms1.communities, comms2.communities)

    def test_manual_override(self):
        seed(42)
        comms1 = algorithms.leiden(self.graph, seed=100)
        seed(42)
        comms2 = algorithms.leiden(self.graph, seed=100)
        self.assertEqual(comms1.communities, comms2.communities)

    def test_reset_seed(self):
        seed(42)
        reset_seed()
        self.assertIsNone(get_seed())

    def test_warning_on_multiple_seed_calls(self):
        seed(42)
        with self.assertWarns(UserWarning):
            seed(123)

    def test_fixed_seed_context_manager(self):
        seed(42)
        original_seed = get_seed()

        with fixed_seed(100):
            self.assertEqual(get_seed(), 100)

        # After context, seed should be restored
        self.assertEqual(get_seed(), original_seed)
