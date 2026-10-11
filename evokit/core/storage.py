"""
Dual-Tier Experiment Storage for EvoKit.
Combines an Optuna-compatible SQLite relational database (for live dashboard visualization)
with physical filesystem code artifact storage (.py files per individual).
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import Field

from evokit.core.base import BaseComponent, ComponentConfig
from evokit.core.entity import Individual


def _optuna_schema_identifiers() -> tuple[int, str, Optional[str]]:
    """Return (schema_version, library_version, alembic_head_revision) for the
    installed Optuna, so the hand-written SQLite DB matches the optuna-dashboard
    that will open it. Falls back to safe constants if Optuna is not installed."""
    schema_version = 12
    library_version = "5.0.0"
    alembic_head: Optional[str] = None
    try:
        import optuna  # noqa: WPS433
        from optuna.storages._rdb import models  # noqa: WPS433

        schema_version = int(models.SCHEMA_VERSION)
        library_version = str(optuna.version.__version__)
    except Exception:
        return schema_version, library_version, alembic_head

    # Resolve the Alembic migration head bundled with this Optuna install.
    try:
        from alembic.config import Config  # noqa: WPS433
        from alembic.script import ScriptDirectory  # noqa: WPS433

        versions_dir = Path(models.__file__).resolve().parent / "alembic"
        cfg = Config()
        cfg.set_main_option("script_location", str(versions_dir))
        alembic_head = ScriptDirectory.from_config(cfg).get_current_head()
    except Exception:
        alembic_head = None
    return schema_version, library_version, alembic_head


class StorageConfig(ComponentConfig):
    db_path: str = Field(default="./evokit_experiments.db", description="SQLite database path")
    artifacts_dir: str = Field(default="./artifacts", description="Directory for physical code artifacts")
    study_name: str = Field(default="evokit_run", description="Optuna study name")
    direction: str = Field(default="MAXIMIZE", description="Optimization direction (MAXIMIZE or MINIMIZE)")


class ExperimentStorage(BaseComponent):
    """
    Dual-Tier Persistence Engine:
    1. Relational Tier: SQLite database strictly aligned with Optuna schema (v3-v5)
       so `optuna-dashboard sqlite:///evokit_experiments.db` works out of the box.
    2. Artifact Tier: Standalone Python files (.py) saved with trial IDs and generation hierarchy.
    """

    def __init__(self, config: Optional[Dict[str, Any] | StorageConfig] = None) -> None:
        super().__init__(config)
        if isinstance(config, StorageConfig):
            self.cfg = config
        elif isinstance(config, dict):
            self.cfg = StorageConfig(**config)
        else:
            self.cfg = StorageConfig()

        # ":memory:" is SQLite's in-memory sentinel, not a filesystem path; keep
        # it verbatim so we don't create a stray on-disk file named ":memory:".
        self._in_memory = str(self.cfg.db_path) == ":memory:"
        self.db_path = Path(self.cfg.db_path) if self._in_memory else Path(self.cfg.db_path).resolve()
        self.artifacts_dir = Path(self.cfg.artifacts_dir).resolve()
        self._conn: Optional[sqlite3.Connection] = None
        self._study_id: Optional[int] = None
        self._trial_counter: int = 0

    def _setup_internal(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        if not self._in_memory:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        self._conn.execute("PRAGMA journal_mode = WAL;")
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._init_schema()
        self._ensure_study()

    def _teardown_internal(self) -> None:
        if self._conn is not None:
            self._conn.commit()
            self._conn.close()
            self._conn = None

    def _init_schema(self) -> None:
        assert self._conn is not None
        cursor = self._conn.cursor()

        # Optuna Compatible Tables
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS version_info (
            version_info_id INTEGER PRIMARY KEY,
            schema_version INTEGER,
            library_version VARCHAR(256),
            CHECK (version_info_id=1)
        );
        """)

        # Discover the installed Optuna's schema/library version and Alembic head
        # so the hand-written DB stays compatible with the optuna-dashboard the
        # user actually has, rather than hardcoding a single Optuna release.
        schema_version, library_version, alembic_head = _optuna_schema_identifiers()

        cursor.execute("SELECT COUNT(*) FROM version_info;")
        if cursor.fetchone()[0] == 0:
            # Match Optuna's RDB schema: single row (id=1), schema_version,
            # plus the library_version column the dashboard's ORM requires.
            cursor.execute(
                "INSERT INTO version_info (version_info_id, schema_version, library_version) "
                "VALUES (1, ?, ?);",
                (schema_version, library_version),
            )

        # Alembic migration-tracking table: optuna-dashboard and RDBStorage check
        # the current migration revision against their head; without this table the
        # revision reads as NULL and the schema-compatibility check fails.
        if alembic_head is not None:
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS alembic_version ("
                "version_num VARCHAR(32) NOT NULL, "
                "CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num));"
            )
            cursor.execute("SELECT COUNT(*) FROM alembic_version;")
            if cursor.fetchone()[0] == 0:
                cursor.execute(
                    "INSERT INTO alembic_version (version_num) VALUES (?);",
                    (alembic_head,),
                )

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS studies (
            study_id INTEGER PRIMARY KEY AUTOINCREMENT,
            study_name VARCHAR(500) NOT NULL UNIQUE
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS study_directions (
            study_direction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            study_id INTEGER NOT NULL,
            direction VARCHAR(512) NOT NULL,
            objective INTEGER NOT NULL,
            FOREIGN KEY(study_id) REFERENCES studies (study_id)
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS study_user_attributes (
            study_user_attribute_id INTEGER PRIMARY KEY AUTOINCREMENT,
            study_id INTEGER NOT NULL,
            key VARCHAR(500),
            value_json VARCHAR(2048),
            FOREIGN KEY(study_id) REFERENCES studies (study_id)
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS study_system_attributes (
            study_system_attribute_id INTEGER PRIMARY KEY AUTOINCREMENT,
            study_id INTEGER NOT NULL,
            key VARCHAR(500),
            value_json VARCHAR(2048),
            FOREIGN KEY(study_id) REFERENCES studies (study_id)
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS trials (
            trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
            number INTEGER NOT NULL,
            study_id INTEGER NOT NULL,
            state VARCHAR(512) NOT NULL,
            datetime_start DATETIME,
            datetime_complete DATETIME,
            FOREIGN KEY(study_id) REFERENCES studies (study_id)
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS trial_values (
            trial_value_id INTEGER PRIMARY KEY AUTOINCREMENT,
            trial_id INTEGER NOT NULL,
            objective INTEGER NOT NULL,
            value FLOAT,
            value_type VARCHAR(512) NOT NULL DEFAULT 'FINITE',
            FOREIGN KEY(trial_id) REFERENCES trials (trial_id)
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS trial_params (
            param_id INTEGER PRIMARY KEY AUTOINCREMENT,
            trial_id INTEGER NOT NULL,
            param_name VARCHAR(512) NOT NULL,
            param_value FLOAT NOT NULL,
            distribution_json VARCHAR(2048) NOT NULL,
            FOREIGN KEY(trial_id) REFERENCES trials (trial_id)
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS trial_user_attributes (
            trial_user_attribute_id INTEGER PRIMARY KEY AUTOINCREMENT,
            trial_id INTEGER NOT NULL,
            key VARCHAR(500),
            value_json TEXT,
            FOREIGN KEY(trial_id) REFERENCES trials (trial_id)
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS trial_system_attributes (
            trial_system_attribute_id INTEGER PRIMARY KEY AUTOINCREMENT,
            trial_id INTEGER NOT NULL,
            key VARCHAR(500),
            value_json TEXT,
            FOREIGN KEY(trial_id) REFERENCES trials (trial_id)
        );
        """)

        # Custom EvoKit Evolutionary Lineage Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS evokit_individuals (
            individual_id VARCHAR(128) PRIMARY KEY,
            trial_id INTEGER,
            study_name VARCHAR(500),
            generation INTEGER,
            operator VARCHAR(128),
            is_valid INTEGER,
            fitness FLOAT,
            artifact_path TEXT,
            parent_ids_json TEXT,
            metrics_json TEXT,
            created_at FLOAT
        );
        """)

        self._conn.commit()

    def _ensure_study(self) -> None:
        assert self._conn is not None
        cursor = self._conn.cursor()
        cursor.execute("SELECT study_id FROM studies WHERE study_name = ?;", (self.cfg.study_name,))
        row = cursor.fetchone()
        if row:
            self._study_id = row[0]
        else:
            cursor.execute("INSERT INTO studies (study_name) VALUES (?);", (self.cfg.study_name,))
            self._study_id = cursor.lastrowid
            dir_str = self.cfg.direction.upper()
            cursor.execute(
                "INSERT INTO study_directions (study_id, direction, objective) VALUES (?, ?, 0);",
                (self._study_id, dir_str),
            )
            self._conn.commit()

        # Query highest trial number
        cursor.execute("SELECT COALESCE(MAX(number), -1) FROM trials WHERE study_id = ?;", (self._study_id,))
        self._trial_counter = cursor.fetchone()[0] + 1

    def record_individual(self, individual: Individual, generation: int) -> Path:
        """
        Persist an evaluated individual:
        1. Writes standalone .py file into artifacts directory.
        2. Inserts an Optuna trial record with fitness value, user attributes, and parameters.
        3. Inserts into evokit_individuals lineage table.
        """
        if not self._is_setup:
            self.setup()
        assert self._conn is not None
        assert self._study_id is not None

        # Tier 2: Persist physical Python artifact
        gen_dir = self.artifacts_dir / self.cfg.study_name / f"gen_{generation:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        artifact_file = gen_dir / f"{individual.id}.py"

        header = f"""# EvoKit Generated Algorithm
# Individual ID: {individual.id}
# Generation: {generation}
# Operator: {individual.operator}
# Fitness: {individual.fitness}
# Valid: {individual.is_valid}
# Parents: {individual.parent_ids}
# Metrics: {json.dumps(individual.metrics)}
"""
        if individual.thought:
            header += f"# Thought: {json.dumps(individual.thought)}\n"

        with open(artifact_file, "w", encoding="utf-8") as f:
            f.write(header + "\n" + individual.code + "\n")

        # Tier 1: Insert into Relational Database (Optuna-Compatible)
        cursor = self._conn.cursor()
        now_str = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        trial_num = self._trial_counter
        self._trial_counter += 1

        state = "COMPLETE" if individual.is_valid and individual.fitness is not None else "FAIL"
        cursor.execute(
            """
            INSERT INTO trials (number, study_id, state, datetime_start, datetime_complete)
            VALUES (?, ?, ?, ?, ?);
            """,
            (trial_num, self._study_id, state, now_str, now_str),
        )
        trial_id = cursor.lastrowid

        # Insert fitness value
        if individual.fitness is not None:
            cursor.execute(
                """
                INSERT INTO trial_values (trial_id, objective, value, value_type)
                VALUES (?, 0, ?, 'FINITE');
                """,
                (trial_id, float(individual.fitness)),
            )

        # Insert parameters (e.g. generation)
        cursor.execute(
            """
            INSERT INTO trial_params (trial_id, param_name, param_value, distribution_json)
            VALUES (?, 'generation', ?, '{"name": "IntDistribution", "attributes": {"low": 0, "high": 1000, "step": 1}}');
            """,
            (trial_id, float(generation)),
        )

        # Insert user attributes for Optuna Dashboard
        user_attrs = {
            "individual_id": individual.id,
            "generation": generation,
            "operator": individual.operator or "unknown",
            "is_valid": individual.is_valid,
            "artifact_path": str(artifact_file),
            "parents": individual.parent_ids,
            "metrics": individual.metrics,
        }
        if individual.thought:
            user_attrs["thought_preview"] = (
                individual.thought[:200] + "..." if len(individual.thought) > 200 else individual.thought
            )

        for k, v in user_attrs.items():
            cursor.execute(
                "INSERT INTO trial_user_attributes (trial_id, key, value_json) VALUES (?, ?, ?);",
                (trial_id, k, json.dumps(v)),
            )

        # Insert into evokit_individuals lineage table
        cursor.execute(
            """
            INSERT OR REPLACE INTO evokit_individuals (
                individual_id, trial_id, study_name, generation, operator,
                is_valid, fitness, artifact_path, parent_ids_json, metrics_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                individual.id,
                trial_id,
                self.cfg.study_name,
                generation,
                individual.operator,
                1 if individual.is_valid else 0,
                individual.fitness,
                str(artifact_file),
                json.dumps(individual.parent_ids),
                json.dumps(individual.metrics),
                individual.created_at,
            ),
        )

        self._conn.commit()
        return artifact_file

    def get_best_individuals(self, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self._is_setup:
            self.setup()
        assert self._conn is not None

        cursor = self._conn.cursor()
        order = "DESC" if self.cfg.direction.upper() == "MAXIMIZE" else "ASC"
        cursor.execute(
            f"""
            SELECT individual_id, generation, operator, fitness, artifact_path, metrics_json
            FROM evokit_individuals
            WHERE study_name = ? AND is_valid = 1 AND fitness IS NOT NULL
            ORDER BY fitness {order}
            LIMIT ?;
            """,
            (self.cfg.study_name, top_k),
        )
        results = []
        for row in cursor.fetchall():
            results.append({
                "individual_id": row[0],
                "generation": row[1],
                "operator": row[2],
                "fitness": row[3],
                "artifact_path": row[4],
                "metrics": json.loads(row[5]) if row[5] else {},
            })
        return results
