"""
Declarative Configuration Loader and Component Registry for EvoKit.
Dynamically instantiates and wires together the 3 pillars, storage, and gateway from YAML specifications.
"""

from __future__ import annotations

import importlib
import os
import re
from pathlib import Path
from typing import Any, Dict, Type
import yaml

from evokit.core.base import BaseComponent
from evokit.core.gateway import LLMGateway, GatewayConfig
from evokit.core.storage import ExperimentStorage, StorageConfig
from evokit.benchmarks.base import BaseBenchmarkTask
from evokit.benchmarks.obp import OBPBenchmarkTask, OBPTaskConfig
from evokit.benchmarks.tsp import TSPBenchmarkTask, TSPTaskConfig
from evokit.plugins.populations.base import BasePopulationManager
from evokit.plugins.populations.lineage_archive import LineageGraphArchive, LineageArchiveConfig
from evokit.plugins.populations.reflection_buffer import ReflectionBuffer, ReflectionBufferConfig
from evokit.plugins.populations.map_elites import MAPElitesGrid, MAPElitesConfig
from evokit.plugins.populations.openevolve_archive import OpenEvolveArchive, OpenEvolveArchiveConfig
from evokit.plugins.mutations.base import BaseMutationStrategy
from evokit.plugins.mutations.eoh import EoHMutationStrategy, EoHConfig
from evokit.plugins.mutations.reevo import ReEvoMutationStrategy, ReEvoConfig
from evokit.plugins.mutations.diff import DiffMutationStrategy
from evokit.plugins.mutations.openevolve import OpenEvolveMutationStrategy, OpenEvolveConfig
from evokit.plugins.evaluators.base import BaseEvaluator
from evokit.plugins.evaluators.subprocess_sandbox import SubprocessSandboxEvaluator
from evokit.plugins.evaluators.mcp_evaluator import MCPEvaluator

POPULATION_REGISTRY: Dict[str, Type[BasePopulationManager]] = {
    "lineage_archive": LineageGraphArchive,
    "reflection_buffer": ReflectionBuffer,
    "map_elites": MAPElitesGrid,
    "openevolve_archive": OpenEvolveArchive,
}

MUTATION_REGISTRY: Dict[str, Type[BaseMutationStrategy]] = {
    "eoh": EoHMutationStrategy,
    "reevo": ReEvoMutationStrategy,
    "diff": DiffMutationStrategy,
    "openevolve": OpenEvolveMutationStrategy,
}

EVALUATOR_REGISTRY: Dict[str, Type[BaseEvaluator]] = {
    "subprocess_sandbox": SubprocessSandboxEvaluator,
    "mcp": MCPEvaluator,
}

BENCHMARK_REGISTRY: Dict[str, Type[BaseBenchmarkTask]] = {
    "obp_online": OBPBenchmarkTask,
    "tsp_construct": TSPBenchmarkTask,
}


def _expand_env_vars(val: Any) -> Any:
    """Recursively expand environment variables in format ${VAR:-default} or $VAR."""
    if isinstance(val, str):
        pattern = re.compile(r"\$\{([^:-]+)(?::-([^}]*))?\}")

        def repl(match):
            var_name = match.group(1)
            default = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(var_name, default)

        return pattern.sub(repl, val)
    elif isinstance(val, dict):
        return {k: _expand_env_vars(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [_expand_env_vars(v) for v in val]
    return val


def load_yaml_config(file_path: str | Path) -> Dict[str, Any]:
    """Parse YAML file with environment variable interpolation."""
    p = Path(file_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Configuration file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _expand_env_vars(data)


def resolve_class(class_path: str) -> Type[Any]:
    """Resolve Python class from 'module.path:ClassName' notation."""
    if ":" in class_path:
        mod_name, cls_name = class_path.split(":", 1)
    else:
        mod_name, cls_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def build_component(name_or_path: str, registry: Dict[str, Type[Any]], config_dict: Dict[str, Any]) -> Any:
    """Instantiate a component from either registered shorthand alias or import path."""
    if name_or_path in registry:
        cls = registry[name_or_path]
    else:
        cls = resolve_class(name_or_path)
    return cls(config_dict)
