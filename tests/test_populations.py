"""
Unit tests for EvoKit Population Managers:
LineageGraphArchive, ReflectionBuffer, and MAPElitesGrid.
"""

import unittest
from evokit.core.entity import Individual
from evokit.plugins.populations.lineage_archive import LineageGraphArchive, LineageArchiveConfig
from evokit.plugins.populations.reflection_buffer import ReflectionBuffer, ReflectionBufferConfig
from evokit.plugins.populations.map_elites import MAPElitesGrid, MAPElitesConfig


class TestPopulations(unittest.TestCase):
    def test_lineage_graph_archive(self):
        archive = LineageGraphArchive(LineageArchiveConfig(capacity=3, direction="MAXIMIZE"))

        ind1 = Individual(id="p1", code="def f1(): pass", fitness=10.0, is_valid=True)
        ind2 = Individual(id="p2", code="def f2(): pass", fitness=20.0, is_valid=True)
        ind3 = Individual(id="p3", code="def f3(): pass", fitness=5.0, is_valid=True)

        self.assertTrue(archive.add(ind1))
        self.assertTrue(archive.add(ind2))
        self.assertTrue(archive.add(ind3))
        self.assertEqual(archive.size(), 3)

        # Elites sorted
        elites = archive.get_elites(top_k=2)
        self.assertEqual(elites[0].id, "p2")
        self.assertEqual(elites[1].id, "p1")

        # Better individual replaces worst (p3 with fitness 5.0 replaced by p4 with fitness 25.0)
        ind4 = Individual(id="p4", code="def f4(): pass", fitness=25.0, is_valid=True, parent_ids=["p1", "p2"])
        self.assertTrue(archive.add(ind4))
        self.assertEqual(archive.size(), 3)
        self.assertEqual(archive.get_elites(1)[0].id, "p4")

        # Check ancestry lineage
        lineage = archive.get_lineage("p4")
        self.assertEqual(len(lineage), 2)
        self.assertEqual(lineage[0].id, "p4")

        # Check DOT export
        dot_str = archive.to_dot_graph()
        self.assertIn("digraph EvoLineage", dot_str)
        self.assertIn('"p1" -> "p4";', dot_str)

    def test_lineage_graph_archive_minimize(self):
        archive = LineageGraphArchive(LineageArchiveConfig(capacity=3, direction="MINIMIZE"))

        ind1 = Individual(id="p1", code="def f1(): pass", fitness=10.0, is_valid=True)
        ind2 = Individual(id="p2", code="def f2(): pass", fitness=20.0, is_valid=True)
        ind3 = Individual(id="p3", code="def f3(): pass", fitness=5.0, is_valid=True)

        self.assertTrue(archive.add(ind1))
        self.assertTrue(archive.add(ind2))
        self.assertTrue(archive.add(ind3))

        # Elites sorted best-first (lowest fitness is elite #1)
        elites = archive.get_elites(top_k=2)
        self.assertEqual(elites[0].id, "p3")  # fitness 5.0
        self.assertEqual(elites[1].id, "p1")  # fitness 10.0

        # In MINIMIZE, fitness 2.0 replaces worst (p2 with fitness 20.0)
        ind4 = Individual(id="p4", code="def f4(): pass", fitness=2.0, is_valid=True)
        self.assertTrue(archive.add(ind4))
        self.assertEqual(archive.get_elites(1)[0].id, "p4")

    def test_reflection_buffer(self):
        buf = ReflectionBuffer(ReflectionBufferConfig(capacity=2, max_failure_history=5, direction="MAXIMIZE"))

        # Add successful individual
        good_ind = Individual(code="def f(): pass", fitness=100.0, is_valid=True)
        self.assertTrue(buf.add(good_ind))

        # Add failing individual
        bad_ind = Individual(code="bad", error_trace="ZeroDivisionError", is_valid=False)
        self.assertFalse(buf.add(bad_ind))

        # Add reflection
        buf.add_reflection("Ensure denominator is not zero", is_long_term=False)
        self.assertEqual(len(buf.get_recent_reflections(1)), 1)
        self.assertEqual(buf.get_recent_reflections(1)[0], "Ensure denominator is not zero")

        # Check failed cases buffer
        failed_cases = buf.get_failed_cases(1)
        self.assertEqual(len(failed_cases), 1)
        self.assertEqual(failed_cases[0].error_trace, "ZeroDivisionError")

    def test_map_elites_grid(self):
        grid = MAPElitesGrid(MAPElitesConfig(grid_dim_1=3, grid_dim_2=3, max_complexity=200, direction="MAXIMIZE"))

        ind1 = Individual(code="def short(): pass", fitness=10.0, is_valid=True)
        ind2 = Individual(code="def short(): pass", fitness=15.0, is_valid=True)  # Same cell, higher fitness

        self.assertTrue(grid.add(ind1))
        self.assertTrue(grid.add(ind2))  # Replaces ind1
        self.assertEqual(grid.size(), 1)
        self.assertEqual(grid.get_elites(1)[0].fitness, 15.0)

        # Different cell (much longer code)
        ind3 = Individual(code="def very_long_function_implementation(): " + "# comment\n" * 20 + "pass", fitness=20.0, is_valid=True)
        self.assertTrue(grid.add(ind3))
        self.assertEqual(grid.size(), 2)
        self.assertGreater(grid.coverage(), 0.0)


if __name__ == "__main__":
    unittest.main()
