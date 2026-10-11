"""
Unit tests for EvoKit Phase 1 Foundation:
Lifecycle components, Data entities, LLMGateway, and Dual-Tier ExperimentStorage.
"""

import os
import shutil
import sqlite3
import tempfile
import unittest

from evokit.core.base import BaseComponent
from evokit.core.entity import Individual, MetricRecord
from evokit.core.context import EvolutionContext
from evokit.core.gateway import LLMGateway, GatewayConfig
from evokit.core.storage import ExperimentStorage, StorageConfig


class DummyComponent(BaseComponent):
    def __init__(self, config=None):
        super().__init__(config)
        self.setup_called = False
        self.teardown_called = False

    def _setup_internal(self):
        self.setup_called = True

    def _teardown_internal(self):
        self.teardown_called = True


class TestCoreComponents(unittest.TestCase):
    def test_base_component_lifecycle(self):
        comp = DummyComponent()
        self.assertFalse(comp.is_setup)
        self.assertFalse(comp.setup_called)

        comp.setup()
        self.assertTrue(comp.is_setup)
        self.assertTrue(comp.setup_called)

        # Idempotent setup
        comp.setup()
        self.assertTrue(comp.is_setup)

        comp.teardown()
        self.assertFalse(comp.is_setup)
        self.assertTrue(comp.teardown_called)

        # Context manager
        with DummyComponent() as c:
            self.assertTrue(c.is_setup)
        self.assertFalse(c.is_setup)

    def test_individual_lifecycle_and_cloning(self):
        ind = Individual(code="def run(): return 42", thought="Initial idea")
        self.assertIsNotNone(ind.id)
        self.assertFalse(ind.is_valid)
        self.assertIsNone(ind.fitness)

        record = MetricRecord(
            fitness=105.5,
            is_valid=True,
            metrics={"runtime_ms": 12.3},
            stdout="Ran ok",
            runtime_seconds=0.012,
        )
        ind.apply_evaluation(record)
        self.assertEqual(ind.fitness, 105.5)
        self.assertTrue(ind.is_valid)
        self.assertEqual(ind.metrics["runtime_ms"], 12.3)
        self.assertEqual(ind.metadata["stdout"], "Ran ok")

        clone = ind.clone(new_id=True)
        self.assertNotEqual(clone.id, ind.id)
        self.assertEqual(clone.code, ind.code)
        self.assertEqual(clone.fitness, ind.fitness)

        # Serialization round-trip
        d = ind.to_dict()
        restored = Individual.from_dict(d)
        self.assertEqual(restored.id, ind.id)
        self.assertEqual(restored.fitness, ind.fitness)

    def test_evolution_context(self):
        ctx = EvolutionContext(
            generation=1,
            population_view=["p1", "p2"],
            task_spec={"problem": "obp"},
        )
        self.assertEqual(ctx.generation, 1)
        self.assertEqual(len(ctx.population_view), 2)

        # Immutability check
        with self.assertRaises(Exception):
            ctx.generation = 2  # dataclass(frozen=True)

        updated = ctx.with_updates(generation=2)
        self.assertEqual(updated.generation, 2)
        self.assertEqual(ctx.generation, 1)

    def test_llm_gateway_mock(self):
        cfg = GatewayConfig(provider="mock")
        gateway = LLMGateway(cfg)
        gateway.set_mock_responses(["def mock_algo():\n    return 1", "def mock_algo_2():\n    return 2"])

        res1 = gateway.generate("test prompt 1")
        self.assertIn("mock_algo", res1.content)
        self.assertGreater(res1.total_tokens, 0)

        res2 = gateway.generate("test prompt 2")
        self.assertIn("mock_algo_2", res2.content)

        usage = gateway.get_token_usage()
        self.assertGreater(usage["total_tokens"], 0)
        self.assertFalse(usage["budget_exhausted"])

    def test_experiment_storage_dual_tier(self):
        temp_dir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(temp_dir, "test.db")
            art_dir = os.path.join(temp_dir, "artifacts")
            cfg = StorageConfig(
                db_path=db_path,
                artifacts_dir=art_dir,
                study_name="test_study",
                direction="MAXIMIZE",
            )
            storage = ExperimentStorage(cfg)
            storage.setup()

            # Verify Optuna SQLite Schema
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = {row[0] for row in cursor.fetchall()}
            required_optuna_tables = {
                "version_info",
                "studies",
                "study_directions",
                "trials",
                "trial_values",
                "trial_params",
                "trial_user_attributes",
                "evokit_individuals",
            }
            for table in required_optuna_tables:
                self.assertIn(table, tables, f"Missing table {table}")

            # Record individual
            ind = Individual(
                id="test_ind_1",
                code="def solve(): pass",
                thought="Heuristic rationale",
                fitness=99.2,
                is_valid=True,
                metrics={"cost": 10},
                operator="eoh_e1",
            )
            art_file = storage.record_individual(ind, generation=1)
            self.assertTrue(art_file.exists())
            self.assertTrue(art_file.is_file())

            # Read back artifact
            with open(art_file, "r") as f:
                content = f.read()
            self.assertIn("def solve(): pass", content)
            self.assertIn("test_ind_1", content)

            # Check DB record
            cursor.execute("SELECT count(*) FROM trials;")
            self.assertEqual(cursor.fetchone()[0], 1)

            cursor.execute("SELECT value FROM trial_values;")
            self.assertAlmostEqual(cursor.fetchone()[0], 99.2)

            cursor.execute("SELECT count(*) FROM trial_user_attributes;")
            self.assertGreater(cursor.fetchone()[0], 0)

            best = storage.get_best_individuals(top_k=1)
            self.assertEqual(len(best), 1)
            self.assertEqual(best[0]["individual_id"], "test_ind_1")
            self.assertEqual(best[0]["fitness"], 99.2)

            conn.close()
            storage.teardown()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
