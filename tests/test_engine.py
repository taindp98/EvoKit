"""
End-to-End Integration Tests for EvoKit EvolutionEngine and Loader.
"""

import os
import shutil
import sqlite3
import tempfile
import unittest

from evokit.loader import load_yaml_config
from evokit.main import assemble_engine
from evokit.benchmarks.bpp import INITIAL_BPP_CODE


class TestEngineIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_exp.db")
        self.artifacts_dir = os.path.join(self.temp_dir, "code")

        self.yaml_content = f"""
experiment:
  study_name: "test_integration_study"
  max_generations: 2
  candidates_per_generation: 2
  evaluate_test_split: true
  test_top_k: 1
  seed_evaluation: true

storage:
  db_path: "{self.db_path}"
  artifacts_dir: "{self.artifacts_dir}"
  direction: "MINIMIZE"

gateway:
  provider: "mock"

benchmark:
  type: "bpp_online"
  capacity: 100.0
  num_train_instances: 2
  num_test_instances: 2
  train_items_per_instance: 20
  test_items_per_instance: 20
  seed: 1

population:
  type: "lineage_archive"
  capacity: 5
  direction: "MINIMIZE"

mutation:
  type: "eoh"

evaluator:
  type: "subprocess_sandbox"
  timeout_seconds: 15.0
"""
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        with open(self.config_path, "w", encoding="utf-8") as f:
            f.write(self.yaml_content)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_end_to_end_evolution_loop(self):
        config_data = load_yaml_config(self.config_path)
        engine = assemble_engine(config_data, dry_run=True)

        # Set mock response for LLM code generation
        engine.gateway.set_mock_responses([
            "{Idea: Best fit variant}\n```python\n" + INITIAL_BPP_CODE + "\n```"
        ])

        summary = engine.run()

        # 1. Verify summary telemetry
        self.assertEqual(summary["benchmark"], "bpp_online")
        self.assertEqual(summary["total_generations"], 2)
        self.assertGreater(summary["population_size"], 0)
        self.assertEqual(len(summary["test_generalization"]), 1)
        self.assertTrue(summary["test_generalization"][0]["test_valid"])

        # 2. Verify physical code artifacts exist on disk
        gen0_dir = os.path.join(self.artifacts_dir, "test_integration_study", "gen_000")
        self.assertTrue(os.path.isdir(gen0_dir))
        gen0_files = [f for f in os.listdir(gen0_dir) if f.endswith(".py")]
        self.assertGreater(len(gen0_files), 0)

        # 3. Verify SQLite Optuna-compatible relational database
        self.assertTrue(os.path.isfile(self.db_path))
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT count(*) FROM trials;")
        trial_count = cursor.fetchone()[0]
        self.assertGreater(trial_count, 1)  # Seed + Gen 1 + Gen 2

        cursor.execute("SELECT count(*) FROM trial_values;")
        values_count = cursor.fetchone()[0]
        self.assertGreater(values_count, 1)

        cursor.execute("SELECT count(*) FROM evokit_individuals;")
        ind_count = cursor.fetchone()[0]
        self.assertGreater(ind_count, 1)

        conn.close()
        engine.teardown()


if __name__ == "__main__":
    unittest.main()
