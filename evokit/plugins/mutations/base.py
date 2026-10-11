"""
Base abstraction for Mutation & Search Strategies in EvoKit (Pillar 2).
"""

from __future__ import annotations

import re
from abc import abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import Field

from evokit.core.base import BaseComponent, ComponentConfig
from evokit.core.context import EvolutionContext
from evokit.core.entity import Individual


class MutationConfig(ComponentConfig):
    temperature: float = Field(default=0.8, description="Sampling temperature for LLM mutations")
    max_tokens: int = Field(default=2048, description="Maximum tokens per generation")


class BaseMutationStrategy(BaseComponent):
    """
    Contract for prompt engineering, semantic transformation, and code generation routines.
    """

    def __init__(self, config: Optional[Dict[str, Any] | MutationConfig] = None) -> None:
        super().__init__(config)
        self.cfg = config if isinstance(config, MutationConfig) else MutationConfig(**(config or {}))

    @abstractmethod
    def mutate(self, context: EvolutionContext, num_candidates: int = 1) -> List[Individual]:
        """
        Generate un-evaluated candidate program variants given the active EvolutionContext.
        """
        pass

    @staticmethod
    def extract_thought(text: str) -> Optional[str]:
        """Extract reasoning or description enclosed within curly braces or headers."""
        brace_match = re.search(r"\{([^}]+)\}", text)
        if brace_match:
            return brace_match.group(1).strip()
        # Fallback to leading lines before markdown block
        before_code = text.split("```")[0].strip()
        if before_code:
            lines = [l.strip() for l in before_code.splitlines() if l.strip()]
            if lines:
                return lines[0]
        return None

    @staticmethod
    def extract_code(text: str, default_fallback: str = "") -> str:
        """
        Robustly extract Python code block from LLM response.
        Handles ```python ... ```, ``` ... ```, or direct function definitions.
        """
        # Try ```python block
        pattern = r"```(?:python)?\s*\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            # Pick longest code block
            return BaseMutationStrategy._sanitize_code(max(matches, key=len).strip())

        # Try to find def ... or import ... up to the end
        match = re.search(r"(?:import |from |def |class ).*", text, re.DOTALL)
        if match:
            return BaseMutationStrategy._sanitize_code(match.group(0).strip())

        return default_fallback.strip()

    @staticmethod
    def _sanitize_code(code: str) -> str:
        """Strip artifacts an LLM sometimes leaks into a code block.

        Most common failure (seen with ReEvo/EoH prompts that ask for a
        ``{thought}`` rationale): the model drops the literal ``{thought}``
        placeholder — or a brace-wrapped prose sentence — on its own line
        *inside* the function body, producing ``name 'thought' is not defined``
        or a syntax error. We remove any standalone line that is just a
        ``{...}`` placeholder / brace-wrapped prose, preserving real dict or set
        literals (which contain ``:``/``,`` or are short inline expressions).
        """
        out = []
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith("{") and stripped.endswith("}") and len(stripped) > 2:
                inner = stripped[1:-1]
                # Heuristic: a leaked rationale placeholder is a bare word or a
                # sentence of words/spaces (no ':' ',' '=' that a real literal has).
                if not any(ch in inner for ch in ":,=\"'") and (
                    inner == "thought" or " " in inner or inner.isidentifier()
                ):
                    continue  # drop the leaked placeholder/prose line
            out.append(line)
        return "\n".join(out).strip()
