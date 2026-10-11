"""
Model Context Protocol (MCP) / JSON-RPC Evaluator for EvoKit.
Enables sandboxed evaluation via standardized MCP tool calls ('evaluate_algorithm').
Supports concurrent batch execution across worker connections.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import time
from typing import Any, Dict, List, Optional
from pydantic import Field

from evokit.benchmarks.base import BaseBenchmarkTask
from evokit.core.entity import Individual, MetricRecord
from evokit.plugins.evaluators.base import BaseEvaluator, EvaluatorConfig


class MCPEvaluatorConfig(EvaluatorConfig):
    server_url: Optional[str] = Field(default=None, description="External MCP HTTP server URL (optional)")
    protocol_version: str = Field(default="2024-11-05", description="MCP protocol version")


class MCPEvaluator(BaseEvaluator):
    """
    Evaluator adhering to the Model Context Protocol (MCP) / JSON-RPC 2.0 specification.
    Wraps task evaluations into standard `tools/call` JSON-RPC payloads.
    """

    def __init__(self, config: Optional[Dict[str, Any] | MCPEvaluatorConfig] = None) -> None:
        if isinstance(config, MCPEvaluatorConfig):
            cfg = config
        elif isinstance(config, dict):
            cfg = MCPEvaluatorConfig(**config)
        else:
            cfg = MCPEvaluatorConfig()
        super().__init__(cfg)
        self.mcp_cfg: MCPEvaluatorConfig = cfg

    def format_mcp_request(self, call_id: int, individual: Individual, task: BaseBenchmarkTask, split: str) -> Dict[str, Any]:
        """Construct a standardized MCP tools/call JSON-RPC 2.0 request."""
        return {
            "jsonrpc": "2.0",
            "id": call_id,
            "method": "tools/call",
            "params": {
                "name": "evaluate_algorithm",
                "arguments": {
                    "individual_id": individual.id,
                    "code": individual.code,
                    "benchmark_name": task.name,
                    "split": split,
                    "timeout": self.cfg.timeout_seconds,
                },
            },
        }

    def parse_mcp_response(self, response_data: Dict[str, Any], runtime_seconds: float) -> MetricRecord:
        """Parse an MCP tools/call response into a MetricRecord."""
        if "error" in response_data:
            err = response_data["error"]
            return MetricRecord(
                fitness=None,
                is_valid=False,
                error_trace=f"MCP Error ({err.get('code')}): {err.get('message')}",
                runtime_seconds=runtime_seconds,
            )

        result = response_data.get("result", {})
        is_error = result.get("isError", False)
        metrics = result.get("metrics", {})
        fitness = metrics.get("fitness", None)
        text_content = ""
        for item in result.get("content", []):
            if item.get("type") == "text":
                text_content += item.get("text", "") + "\n"

        return MetricRecord(
            fitness=fitness,
            is_valid=not is_error and fitness is not None,
            metrics=metrics,
            stdout=text_content,
            error_trace=text_content if is_error else None,
            runtime_seconds=runtime_seconds,
        )

    def evaluate(
        self,
        individual: Individual,
        task: BaseBenchmarkTask,
        split: str = "train",
    ) -> MetricRecord:
        start_t = time.time()
        # If external server_url is configured, dispatch via HTTP client
        if self.mcp_cfg.server_url:
            import httpx
            req_payload = self.format_mcp_request(1, individual, task, split)
            try:
                with httpx.Client(timeout=self.cfg.timeout_seconds + 5.0) as client:
                    resp = client.post(self.mcp_cfg.server_url, json=req_payload)
                    resp.raise_for_status()
                    data = resp.json()
                    return self.parse_mcp_response(data, time.time() - start_t)
            except Exception as e:
                return MetricRecord(
                    fitness=None,
                    is_valid=False,
                    error_trace=f"MCP Remote Error: {e}",
                    runtime_seconds=time.time() - start_t,
                )

        # Local deterministic MCP emulation fallback
        record = task.evaluate_code(individual.code, split=split, timeout=self.cfg.timeout_seconds)
        simulated_mcp_res = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [{"type": "text", "text": record.stdout or ("OK" if record.is_valid else record.error_trace)}],
                "isError": not record.is_valid,
                "metrics": {**record.metrics, "fitness": record.fitness},
            },
        }
        return self.parse_mcp_response(simulated_mcp_res, time.time() - start_t)

    def evaluate_batch(
        self,
        individuals: List[Individual],
        task: BaseBenchmarkTask,
        split: str = "train",
    ) -> List[MetricRecord]:
        """Concurrently evaluate candidate programs via MCP tools/call."""
        if not individuals:
            return []
        max_workers = max(1, int(self.cfg.max_workers))
        if max_workers == 1 or len(individuals) == 1:
            return [self.evaluate(ind, task, split=split) for ind in individuals]

        pool_size = min(max_workers, len(individuals))
        results: List[Optional[MetricRecord]] = [None] * len(individuals)
        with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
            future_to_idx = {
                executor.submit(self.evaluate, ind, task, split): idx
                for idx, ind in enumerate(individuals)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = MetricRecord(
                        fitness=None,
                        is_valid=False,
                        error_trace=f"MCP parallel worker execution error: {e}",
                    )
        return [r for r in results if r is not None]
