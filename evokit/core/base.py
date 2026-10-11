"""
Base lifecycle contract and component interfaces for EvoKit.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict


class ComponentConfig(BaseModel):
    """Base configuration model for EvoKit components."""
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class BaseComponent(ABC):
    """
    Abstract base lifecycle component in EvoKit.
    All pluggable modules (Populations, Mutations, Evaluators, Storages, Gateways)
    must conform to this contract.
    """

    def __init__(self, config: Optional[Dict[str, Any] | ComponentConfig] = None) -> None:
        self.raw_config = config or {}
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._is_setup: bool = False

    @property
    def is_setup(self) -> bool:
        return self._is_setup

    def setup(self) -> None:
        """
        Lifecycle hook: Validates configuration, checks credentials,
        initializes database connections, container pools, or local caches.
        Must be idempotent.
        """
        if not self._is_setup:
            self._setup_internal()
            self._is_setup = True
            self.logger.debug(f"{self.__class__.__name__} successfully initialized.")

    def teardown(self) -> None:
        """
        Lifecycle hook: Gracefully terminates subprocesses, closes network connections,
        and flushes disk/database buffers.
        Must be idempotent.
        """
        if self._is_setup:
            try:
                self._teardown_internal()
            except Exception as e:
                self.logger.error(f"Error during {self.__class__.__name__} teardown: {e}", exc_info=True)
            finally:
                self._is_setup = False
                self.logger.debug(f"{self.__class__.__name__} torn down.")

    def _setup_internal(self) -> None:
        """Override in subclass to provide custom setup logic."""
        pass

    def _teardown_internal(self) -> None:
        """Override in subclass to provide custom teardown logic."""
        pass

    def __enter__(self) -> BaseComponent:
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.teardown()
