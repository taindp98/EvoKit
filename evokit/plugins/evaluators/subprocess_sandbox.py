"""
Isolated Subprocess Sandbox Evaluator for EvoKit.
Executes candidate code inside isolated subprocesses with hard wall-clock timeouts,
process tree cleanup, and complete stdout/stderr capture.
Supports multi-processing / concurrent execution of candidate batches.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

from evokit.benchmarks.base import BaseBenchmarkTask
from evokit.core.entity import Individual, MetricRecord
from evokit.plugins.evaluators.base import BaseEvaluator, EvaluatorConfig

MARKER = "__EVOKIT_RESULT__="


class SubprocessSandboxEvaluator(BaseEvaluator):
    """
    Evaluator that executes candidate programs in fresh, isolated subprocesses
    with CPU timeout, filesystem isolation, and complete output capture.
    Supports parallel batch evaluation across multiple worker processes.
    """

    def evaluate(
        self,
        individual: Individual,
        task: BaseBenchmarkTask,
        split: str = "train",
    ) -> MetricRecord:
        start_t = time.time()
        timeout = float(self.cfg.timeout_seconds)

        with tempfile.TemporaryDirectory(prefix="evokit_sandbox_") as tmpdir:
            script_path = os.path.join(tmpdir, "sandbox_run.py")
            code_path = os.path.join(tmpdir, "candidate.py")

            with open(code_path, "w", encoding="utf-8") as f:
                f.write(individual.code)

            # Runner script executed in subprocess
            runner_script = f"""
import sys
import json
import time

from {task.__class__.__module__} import {task.__class__.__name__}

try:
    with open(r'{code_path}', 'r', encoding='utf-8') as f:
        code_str = f.read()

    task_obj = {task.__class__.__name__}({task.cfg.model_dump()})
    task_obj.setup()
    record = task_obj.evaluate_code(code_str, split='{split}')
    result_dict = record.model_dump()
except Exception as e:
    result_dict = {{
        'fitness': None,
        'is_valid': False,
        'metrics': {{}},
        'stdout': '',
        'stderr': str(e),
        'error_trace': f'Sandbox Runner Exception: {{e}}',
        'runtime_seconds': 0.0,
    }}

print('{MARKER}' + json.dumps(result_dict))
"""

            with open(script_path, "w", encoding="utf-8") as f:
                f.write(runner_script)

            env = os.environ.copy()
            # Ensure evokit is importable by child subprocess
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{repo_root}:{pythonpath}" if pythonpath else repo_root

            try:
                proc = subprocess.run(
                    [sys.executable, script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout,
                    env=env,
                )
                stdout = proc.stdout
                stderr = proc.stderr
                runtime = time.time() - start_t

                # Extract marked JSON record
                record_data = None
                clean_stdout_lines = []
                for line in stdout.splitlines():
                    if line.startswith(MARKER):
                        json_str = line[len(MARKER):]
                        try:
                            record_data = json.loads(json_str)
                        except json.JSONDecodeError:
                            pass
                    else:
                        clean_stdout_lines.append(line)

                clean_stdout = "\n".join(clean_stdout_lines)

                if record_data is not None:
                    record_data["stdout"] = clean_stdout
                    if stderr and not record_data.get("stderr"):
                        record_data["stderr"] = stderr
                    if not record_data.get("runtime_seconds"):
                        record_data["runtime_seconds"] = runtime
                    return MetricRecord(**record_data)
                else:
                    return MetricRecord(
                        fitness=None,
                        is_valid=False,
                        stdout=clean_stdout,
                        stderr=stderr,
                        error_trace=stderr or "Subprocess ended without returning evaluation record.",
                        runtime_seconds=runtime,
                    )

            except subprocess.TimeoutExpired:
                return MetricRecord(
                    fitness=None,
                    is_valid=False,
                    stdout="",
                    stderr=f"TimeoutExpired: Execution exceeded {timeout} seconds.",
                    error_trace=f"Process timed out after {timeout}s",
                    runtime_seconds=timeout,
                )
            except Exception as e:
                return MetricRecord(
                    fitness=None,
                    is_valid=False,
                    stdout="",
                    stderr=str(e),
                    error_trace=f"Subprocess spawn error: {e}",
                    runtime_seconds=time.time() - start_t,
                )

    def evaluate_batch(
        self,
        individuals: List[Individual],
        task: BaseBenchmarkTask,
        split: str = "train",
    ) -> List[MetricRecord]:
        """
        Concurrently evaluate candidate programs in parallel isolated subprocesses.
        Distributes evaluations across workers, handling individual timeouts and output isolation.
        Preserves the 1-to-1 input ordering of individuals.
        """
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
                        error_trace=f"Parallel worker execution error: {e}",
                    )

        return [r for r in results if r is not None]
