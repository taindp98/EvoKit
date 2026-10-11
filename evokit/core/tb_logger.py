"""
TensorBoard telemetry logger for EvoKit.

Logs the best-so-far (incumbent) objective as a scalar under multiple
sub-runs, each using a different progress counter as the TensorBoard step
("global_step"). This yields a configurable x-axis in TensorBoard: pick the
run `by_generation`, `by_llm_calls`, `by_fevals`, or use TensorBoard's built-in
RELATIVE / WALL time toggle for the wall-clock axis.

The logger is a no-op if `log_dir` is falsy or TensorBoard is not installed, so
the engine never hard-depends on it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("evokit.tb_logger")

# Each key is a TensorBoard "run" (its own subdirectory); the value names the
# progress counter whose value becomes the step (x-axis) for that run.
_AXIS_RUNS = {
    "by_generation": "generation",
    "by_llm_calls": "llm_calls",
    "by_fevals": "function_evaluations",
}

# Scalar tag used across every run so the curves overlay cleanly.
_OBJECTIVE_TAG = "best_objective"


class TensorBoardLogger:
    """Writes best-so-far objective curves with a switchable x-axis."""

    def __init__(self, log_dir: Optional[str], experiment_id: Optional[str] = None) -> None:
        self.enabled = False
        self._writers: Dict[str, object] = {}
        self.exp_dir: Optional[str] = None
        if not log_dir:
            return

        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
        except Exception:
            try:
                from tensorboardX import SummaryWriter  # type: ignore
            except Exception:
                logger.warning(
                    "TensorBoard logging requested but no SummaryWriter backend "
                    "(torch.utils.tensorboard or tensorboardX) is installed; skipping."
                )
                return

        # Each experiment gets its own subdirectory so successive runs never
        # collide or overwrite each other in TensorBoard's run list.
        root = Path(log_dir)
        exp_root = root / experiment_id if experiment_id else root
        exp_root.mkdir(parents=True, exist_ok=True)
        for run_name in _AXIS_RUNS:
            self._writers[run_name] = SummaryWriter(log_dir=str(exp_root / run_name))
        # A flat run for extra scalars keyed by generation (tokens, FE/s, etc.).
        self._writers["scalars"] = SummaryWriter(log_dir=str(exp_root / "scalars"))
        self.enabled = True
        self.exp_dir = str(exp_root)
        self.log_dir = str(exp_root)
        logger.info("TensorBoard logging enabled at %s", exp_root)

    def log_metadata(self, metadata: Dict[str, object]) -> None:
        """Write run provenance (experiment id, seed, timestamp, settings, log
        paths) as a markdown table into TensorBoard's TEXT tab so the run is
        self-describing: when it ran, with what settings, and where raw logs are.
        """
        if not self.enabled or not metadata:
            return
        writer = self._writers.get("scalars")
        if writer is None:
            return

        def _fmt(value: object) -> str:
            text = str(value)
            # Escape pipe so it doesn't break the markdown table.
            return text.replace("|", "\\|")

        rows = ["| Field | Value |", "| --- | --- |"]
        for key, value in metadata.items():
            rows.append(f"| {_fmt(key)} | {_fmt(value)} |")
        table = "\n".join(rows)
        try:
            # global_step=0 so it pins to the start of the run.
            writer.add_text("run_metadata", table, global_step=0)
        except Exception:
            logger.warning("Failed to write run_metadata to TensorBoard", exc_info=True)

    def log_generation(
        self,
        *,
        generation: int,
        best_objective: float,
        llm_calls: int,
        function_evaluations: int,
        wall_time_seconds: float,
        extra_scalars: Optional[Dict[str, float]] = None,
    ) -> None:
        """Emit one telemetry point for a completed generation (or gen 0 seed).

        `best_objective` is the incumbent (best-so-far) fitness. The same value
        is written into each axis-run at that run's step, plus a wall-clock
        timestamp so TensorBoard's RELATIVE/WALL toggle works.
        """
        if not self.enabled:
            return

        steps = {
            "by_generation": int(generation),
            "by_llm_calls": int(llm_calls),
            "by_fevals": int(function_evaluations),
        }
        import time as _time

        walltime = _time.time()
        for run_name, step in steps.items():
            writer = self._writers.get(run_name)
            if writer is not None:
                writer.add_scalar(
                    _OBJECTIVE_TAG, float(best_objective), global_step=step, walltime=walltime
                )

        scalars_writer = self._writers.get("scalars")
        if scalars_writer is not None:
            scalars_writer.add_scalar(
                "wall_time_seconds", float(wall_time_seconds), global_step=int(generation)
            )
            scalars_writer.add_scalar(
                "cumulative_llm_calls", float(llm_calls), global_step=int(generation)
            )
            scalars_writer.add_scalar(
                "cumulative_function_evaluations",
                float(function_evaluations),
                global_step=int(generation),
            )
            if extra_scalars:
                for name, value in extra_scalars.items():
                    scalars_writer.add_scalar(name, float(value), global_step=int(generation))

    def close(self) -> None:
        for writer in self._writers.values():
            try:
                writer.flush()
                writer.close()
            except Exception:
                pass
        self._writers.clear()
        self.enabled = False
