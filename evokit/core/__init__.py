"""
Core contracts, entities, lifecycle components, and storage for EvoKit.
"""

from evokit.core.base import BaseComponent, ComponentConfig
from evokit.core.entity import Individual, MetricRecord
from evokit.core.context import EvolutionContext
from evokit.core.gateway import LLMGateway, GatewayConfig, LLMResponse
from evokit.core.storage import ExperimentStorage, StorageConfig

__all__ = [
    "BaseComponent",
    "ComponentConfig",
    "Individual",
    "MetricRecord",
    "EvolutionContext",
    "LLMGateway",
    "GatewayConfig",
    "LLMResponse",
    "ExperimentStorage",
    "StorageConfig",
]
