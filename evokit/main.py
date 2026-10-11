"""
Dynamic CLI Entrypoint and Orchestrator Assembler for EvoKit.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from evokit.loader import (
    load_yaml_config,
    build_component,
    POPULATION_REGISTRY,
    MUTATION_REGISTRY,
    EVALUATOR_REGISTRY,
    BENCHMARK_REGISTRY,
)
from evokit.core.gateway import LLMGateway, GatewayConfig
from evokit.core.storage import ExperimentStorage, StorageConfig
from evokit.engine import EvolutionEngine, EngineConfig

console = Console()


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def assemble_engine(config_data: Dict[str, Any], dry_run: bool = False) -> EvolutionEngine:
    """Wire together components from declarative config dictionary."""
    exp_cfg = config_data.get("experiment", {})
    storage_cfg = config_data.get("storage", {})
    gateway_cfg = config_data.get("gateway", {})
    bench_cfg = config_data.get("benchmark", {})
    pop_cfg = config_data.get("population", {})
    mut_cfg = config_data.get("mutation", {})
    eval_cfg = config_data.get("evaluator", {})

    # Dry-run override uses mock LLM
    if dry_run:
        gateway_cfg["provider"] = "mock"

    storage_cfg["study_name"] = exp_cfg.get("study_name", storage_cfg.get("study_name", "evokit_experiment"))

    storage = ExperimentStorage(StorageConfig(**storage_cfg))
    gateway = LLMGateway(GatewayConfig(**gateway_cfg))

    benchmark = build_component(bench_cfg.get("type", "bpp_online"), BENCHMARK_REGISTRY, bench_cfg)
    population = build_component(pop_cfg.get("type", "lineage_archive"), POPULATION_REGISTRY, pop_cfg)
    mutation = build_component(mut_cfg.get("type", "eoh"), MUTATION_REGISTRY, mut_cfg)
    evaluator = build_component(eval_cfg.get("type", "subprocess_sandbox"), EVALUATOR_REGISTRY, eval_cfg)

    engine = EvolutionEngine(
        gateway=gateway,
        storage=storage,
        benchmark=benchmark,
        population=population,
        mutation=mutation,
        evaluator=evaluator,
        config=EngineConfig(**exp_cfg),
    )
    return engine


def cmd_run(args: argparse.Namespace) -> None:
    console.print(Panel(f"[bold cyan]EvoKit Optimization Run[/bold cyan]\nConfig: {args.config}", border_style="cyan"))
    config_data = load_yaml_config(args.config)

    # Overrides from CLI flags
    if args.generations:
        config_data.setdefault("experiment", {})["max_generations"] = args.generations
    if args.vllm_url:
        config_data.setdefault("gateway", {})["base_url"] = args.vllm_url
    if args.model:
        config_data.setdefault("gateway", {})["model"] = args.model
    if args.max_workers:
        config_data.setdefault("evaluator", {})["max_workers"] = args.max_workers
    if args.instance_workers:
        config_data.setdefault("benchmark", {})["parallel_workers"] = args.instance_workers

    engine = assemble_engine(config_data, dry_run=args.dry_run)
    summary = engine.run()

    # Display Rich summary
    table = Table(title="EvoKit Run Results", border_style="green")
    table.add_column("Property", style="bold")
    table.add_column("Value")
    table.add_row("Benchmark", str(summary["benchmark"]))
    table.add_row("Generations", str(summary["total_generations"]))
    table.add_row("Runtime", f"{summary['total_runtime_seconds']:.2f}s")
    table.add_row("Active Population Size", str(summary["population_size"]))
    table.add_row("Total LLM Tokens", str(summary["token_usage"]["total_tokens"]))

    elites = summary.get("best_elites", [])
    if elites:
        best = elites[0]
        table.add_row("Best Train Fitness", str(best.get("fitness")))
        table.add_row("Best Individual ID", str(best.get("id")))

    test_gens = summary.get("test_generalization", [])
    if test_gens:
        test_fit = test_gens[0].get("test_fitness")
        table.add_row("Hold-out Test Fitness", str(test_fit))

    console.print(table)


def cmd_evaluate(args: argparse.Namespace) -> None:
    code_path = Path(args.code)
    if not code_path.is_file():
        console.print(f"[red]Error: file not found {code_path}[/red]")
        sys.exit(1)

    with open(code_path, "r", encoding="utf-8") as f:
        code_str = f.read()

    bench_kwargs = {}
    if args.instance_workers:
        bench_kwargs["parallel_workers"] = args.instance_workers

    benchmark = build_component(args.benchmark, BENCHMARK_REGISTRY, bench_kwargs)
    benchmark.setup()

    evaluator = EVALUATOR_REGISTRY["subprocess_sandbox"]({"timeout_seconds": args.timeout})
    evaluator.setup()

    from evokit.core.entity import Individual
    ind = Individual(code=code_str)
    rec = evaluator.evaluate(ind, benchmark, split=args.split)

    console.print(Panel(f"Evaluation on [bold]{args.benchmark}[/bold] ({args.split} split)", border_style="magenta"))
    console.print(f"Valid: [bold]{rec.is_valid}[/bold]")
    console.print(f"Fitness: [bold]{rec.fitness}[/bold]")
    console.print(f"Runtime: {rec.runtime_seconds:.3f}s")
    if rec.metrics:
        console.print(f"Metrics: {json.dumps(rec.metrics, indent=2)}")
    if rec.error_trace:
        console.print(f"[red]Error Trace:\n{rec.error_trace}[/red]")


def cmd_dashboard(args: argparse.Namespace) -> None:
    db_path = Path(args.db).resolve()
    sqlite_url = f"sqlite:///{db_path}"
    console.print(Panel(
        f"[bold green]Optuna Dashboard Integration[/bold green]\n\n"
        f"Database Path: {db_path}\n"
        f"Optuna URL: {sqlite_url}\n\n"
        f"To launch the interactive dashboard, run:\n"
        f"[bold cyan]optuna-dashboard {sqlite_url}[/bold cyan]",
        border_style="green",
    ))


def cli() -> None:
    parser = argparse.ArgumentParser(description="EvoKit: Modular Evolutionary Algorithm Design Framework")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run command
    run_parser = subparsers.add_parser("run", help="Execute evolutionary optimization run")
    run_parser.add_argument("-c", "--config", required=True, help="Path to declarative YAML configuration file")
    run_parser.add_argument("-g", "--generations", type=int, help="Override number of generations")
    run_parser.add_argument("--vllm-url", type=str, help="Override vLLM inference server URL")
    run_parser.add_argument("--model", type=str, help="Override model identifier")
    run_parser.add_argument("--max-workers", type=int, help="Tier 1: Parallel sandbox workers for evaluating multiple candidates")
    run_parser.add_argument("--instance-workers", type=int, help="Tier 2: Parallel workers for evaluating instances within a benchmark task")
    run_parser.add_argument("--dry-run", action="store_true", help="Execute with mock LLM backend for zero-cost validation")

    # evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Directly evaluate a code artifact")
    eval_parser.add_argument("--code", required=True, help="Path to Python code file")
    eval_parser.add_argument("--benchmark", default="bpp_online", help="Benchmark name (bpp_online, tsp_construct)")
    eval_parser.add_argument("--split", default="train", choices=["train", "test"], help="Dataset split")
    eval_parser.add_argument("--timeout", type=float, default=30.0, help="Per-evaluation timeout in seconds")
    eval_parser.add_argument("--instance-workers", type=int, help="Tier 2: Parallel workers for evaluating instances")

    # dashboard command
    dash_parser = subparsers.add_parser("dashboard", help="Optuna Dashboard connection information")
    dash_parser.add_argument("--db", default="./evokit_experiments.db", help="Path to SQLite database")

    args = parser.parse_args()
    configure_logging(args.verbose)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "dashboard":
        cmd_dashboard(args)


if __name__ == "__main__":
    cli()
