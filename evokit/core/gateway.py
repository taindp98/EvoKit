"""
Unified LLM Gateway for EvoKit.
Provides unified inference routing to local vLLM OpenAI-compatible endpoints,
cloud providers, and mock backends with retry policies, token tracking,
and concurrent multi-request parallel sampling.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel, Field

from evokit.core.base import BaseComponent, ComponentConfig


class GatewayConfig(ComponentConfig):
    provider: str = Field(default="vllm", description="vllm, openai, anthropic, or mock")
    base_url: str = Field(default="http://172.22.4.1:8000/v1")
    model: str = Field(default="mistralai/Devstral-Small-2-24B-Instruct-2512")
    api_key: str = Field(default="EMPTY")
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(default=4096)
    seed: Optional[int] = None
    timeout: float = Field(default=120.0)
    max_retries: int = Field(default=3)
    max_parallel_requests: int = Field(default=8, description="Maximum concurrent parallel requests to LLM backend")
    token_budget: Optional[int] = Field(default=None, description="Total token budget cap")


class LLMResponse(BaseModel):
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    model: str = ""
    raw_response: Dict[str, Any] = Field(default_factory=dict)


class LLMGateway(BaseComponent):
    """
    Unified Gateway for language model inference.
    Routes requests to local vLLM OpenAI-compatible endpoints, cloud providers, or mock backends.
    Tracks token budgets, enforces retry policies, and supports concurrent parallel sampling.
    """

    def __init__(self, config: Optional[Dict[str, Any] | GatewayConfig] = None) -> None:
        super().__init__(config)
        if isinstance(config, GatewayConfig):
            self.cfg = config
        elif isinstance(config, dict):
            self.cfg = GatewayConfig(**config)
        else:
            self.cfg = GatewayConfig()

        self._client: Optional[httpx.Client] = None
        self._lock = threading.Lock()
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._mock_responses: List[str] = []
        self._mock_call_count = 0

    def _setup_internal(self) -> None:
        if self.cfg.provider != "mock":
            limits = httpx.Limits(max_connections=50, max_keepalive_connections=20)
            self._client = httpx.Client(
                base_url=self.cfg.base_url.rstrip("/"),
                headers={"Authorization": f"Bearer {self.cfg.api_key}"},
                timeout=self.cfg.timeout,
                limits=limits,
            )

    def _teardown_internal(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def set_mock_responses(self, responses: List[str]) -> None:
        """Configure sequential mock responses for deterministic testing."""
        with self._lock:
            self._mock_responses = list(responses)
            self._mock_call_count = 0

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return self._total_prompt_tokens + self._total_completion_tokens

    def get_token_usage(self) -> Dict[str, Any]:
        with self._lock:
            tot = self._total_prompt_tokens + self._total_completion_tokens
            return {
                "prompt_tokens": self._total_prompt_tokens,
                "completion_tokens": self._total_completion_tokens,
                "total_tokens": tot,
                "token_budget": self.cfg.token_budget,
                "budget_exhausted": (
                    self.cfg.token_budget is not None and tot >= self.cfg.token_budget
                ),
            }

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate text completion for a single prompt string."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Send chat completion request to the configured backend."""
        if self.cfg.token_budget and self.total_tokens >= self.cfg.token_budget:
            raise RuntimeError(f"LLM token budget exhausted ({self.total_tokens} >= {self.cfg.token_budget})")

        if self.cfg.provider == "mock":
            return self._handle_mock(messages)

        return self._call_openai_compatible(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)

    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_workers: Optional[int] = None,
        **kwargs,
    ) -> List[LLMResponse]:
        """
        Generate text completions for multiple prompts concurrently.
        Enables high-throughput continuous batching on vLLM.
        """
        if not prompts:
            return []

        workers = max_workers or self.cfg.max_parallel_requests
        workers = max(1, min(workers, len(prompts)))

        if workers == 1 or len(prompts) == 1:
            return [
                self.generate(p, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens, **kwargs)
                for p in prompts
            ]

        results: List[Optional[LLMResponse]] = [None] * len(prompts)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(
                    self.generate, p, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens, **kwargs
                ): idx
                for idx, p in enumerate(prompts)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        return [r for r in results if r is not None]

    def chat_batch(
        self,
        messages_list: List[List[Dict[str, str]]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_workers: Optional[int] = None,
        **kwargs,
    ) -> List[LLMResponse]:
        """
        Send multiple chat completion requests concurrently.
        """
        if not messages_list:
            return []

        workers = max_workers or self.cfg.max_parallel_requests
        workers = max(1, min(workers, len(messages_list)))

        if workers == 1 or len(messages_list) == 1:
            return [
                self.chat(m, temperature=temperature, max_tokens=max_tokens, **kwargs)
                for m in messages_list
            ]

        results: List[Optional[LLMResponse]] = [None] * len(messages_list)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(
                    self.chat, m, temperature=temperature, max_tokens=max_tokens, **kwargs
                ): idx
                for idx, m in enumerate(messages_list)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        return [r for r in results if r is not None]

    def _handle_mock(self, messages: List[Dict[str, str]]) -> LLMResponse:
        full_prompt = " ".join(m.get("content", "") for m in messages)
        with self._lock:
            if self._mock_responses:
                idx = self._mock_call_count % len(self._mock_responses)
                content = self._mock_responses[idx]
            elif "select_next_node" in full_prompt:
                content = (
                    "{Idea: Nearest neighbor mock}\n"
                    "```python\n"
                    "import numpy as np\n"
                    "def select_next_node(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:\n"
                    "    return min(unvisited_nodes, key=lambda node: distance_matrix[current_node, node])\n"
                    "```"
                )
            elif "priority" in full_prompt:
                content = (
                    "{Idea: Best fit mock}\n"
                    "```python\n"
                    "import numpy as np\n"
                    "def priority(item: float, bins: np.ndarray) -> np.ndarray:\n"
                    "    return -(bins - item)\n"
                    "```"
                )
            else:
                content = "def solve():\n    pass"

            self._mock_call_count += 1
            prompt_len = sum(len(m.get("content", "")) for m in messages) // 4
            comp_len = len(content) // 4
            self._total_prompt_tokens += prompt_len
            self._total_completion_tokens += comp_len

        return LLMResponse(
            content=content,
            prompt_tokens=prompt_len,
            completion_tokens=comp_len,
            total_tokens=prompt_len + comp_len,
            latency_seconds=0.001,
            model="mock",
            raw_response={"status": "mock"},
        )

    def _call_openai_compatible(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        if self._client is None:
            self.setup()

        temp = temperature if temperature is not None else self.cfg.temperature
        tokens = max_tokens if max_tokens is not None else self.cfg.max_tokens

        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": temp,
            "top_p": self.cfg.top_p,
            "max_tokens": tokens,
        }
        if self.cfg.seed is not None:
            payload["seed"] = self.cfg.seed
        payload.update(kwargs)

        last_error = None
        start_time = time.time()
        for attempt in range(self.cfg.max_retries):
            try:
                response = self._client.post("/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()

                latency = time.time() - start_time
                choice = data["choices"][0]
                message_content = choice.get("message", {}).get("content", "")
                usage = data.get("usage", {})
                p_tokens = usage.get("prompt_tokens", 0)
                c_tokens = usage.get("completion_tokens", 0)
                t_tokens = usage.get("total_tokens", p_tokens + c_tokens)

                with self._lock:
                    self._total_prompt_tokens += p_tokens
                    self._total_completion_tokens += c_tokens

                return LLMResponse(
                    content=message_content,
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    total_tokens=t_tokens,
                    latency_seconds=latency,
                    model=data.get("model", self.cfg.model),
                    raw_response=data,
                )
            except Exception as e:
                last_error = e
                self.logger.warning(f"LLM request failed (attempt {attempt + 1}/{self.cfg.max_retries}): {e}")
                time.sleep(1.0 * (attempt + 1))

        raise RuntimeError(f"LLM generation failed after {self.cfg.max_retries} attempts: {last_error}") from last_error
