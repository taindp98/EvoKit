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

        LLMs asked for a ``{thought}`` rationale (ReEvo/EoH/OpenEvolve prompts)
        occasionally drop that reasoning *inside* the function body — a literal
        ``{thought}`` placeholder, a brace-wrapped sentence, or bare English prose
        after the ``def`` line / an unterminated docstring — which yields
        ``name 'thought' is not defined`` or a syntax error, needlessly marking an
        otherwise-good heuristic invalid.

        This is parse-guided and conservative: if the code already parses, it is
        returned untouched (zero risk to valid code). Only when it fails to parse
        do we drop obvious leaked lines (brace-wrapped placeholders/prose, and
        prose sentences sitting at statement position) and keep the edit *only* if
        it makes the code parse.
        """
        code = code.strip()
        if BaseMutationStrategy._parses(code):
            return code

        kept = []
        for line in code.splitlines():
            if BaseMutationStrategy._is_leaked_line(line):
                continue
            kept.append(line)
        repaired = "\n".join(kept).strip()
        # Keep the repair only if it actually fixed parseability; else leave the
        # original so a genuine bug surfaces as an honest evaluation failure.
        if repaired and BaseMutationStrategy._parses(repaired):
            return repaired
        return code

    @staticmethod
    def _parses(code: str) -> bool:
        import ast
        try:
            ast.parse(code)
            return True
        except Exception:
            return False

    @staticmethod
    def _is_leaked_line(line: str) -> bool:
        """True for a line that looks like leaked rationale/prose, not real code."""
        s = line.strip()
        if not s:
            return False
        # Brace-wrapped placeholder or prose: {thought}, {some idea ...}
        if s.startswith("{") and s.endswith("}") and len(s) > 2:
            inner = s[1:-1]
            if not any(ch in inner for ch in ":,=\"'"):
                return True
        # Bare English prose at statement position: starts with a capitalized word,
        # contains spaces, has no Python operators/brackets/keywords that real code
        # lines carry, and isn't a comment/string/decorator.
        if s[0].isalpha() and s[0].isupper() and " " in s:
            if not any(ch in s for ch in "=()[]{}:#\"'") and not s.endswith(":"):
                words = s.split()
                # e.g. "The core idea is to balance two goals" / "Calculate priority scores..."
                if len(words) >= 3 and all(w.replace(",", "").replace(".", "").isalpha() for w in words):
                    return True
        return False
