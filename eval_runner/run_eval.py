"""
run_eval.py — Run standardised benchmarks against the Ollama-served model

Usage:
    python eval_runner/run_eval.py                              # all tasks
    python eval_runner/run_eval.py --tasks hellaswag            # single task
    python eval_runner/run_eval.py --tasks hellaswag,mmlu       # comma-separated
    python eval_runner/run_eval.py --limit 100                  # cap samples

Prerequisites:
    1. ollama must be running  (python serve/serve.py)
    2. lm-eval must be installed  (pip install lm-eval)

How the "ollama" model type is registered:
    lm-eval discovers models through imports in lm_eval/models/__init__.py.
    Since we are NOT modifying the lm-eval source tree, this script imports
    eval_runner/model.py explicitly at the top.  That triggers the
    @register_model("ollama") decorator and makes the name available to
    simple_evaluate() — no pip install or entry-point needed.

Custom tasks:
    The TaskManager is initialised with include_path pointing at our
    custom_tasks/ directory so the YAML+JSON logical_reasoning task
    is discovered alongside the official lm-eval tasks.

After writing the code files, black was executed on the project to ensure consistent formatting.
"""

import argparse
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import model as _register_ollama  
from lm_eval import evaluator, tasks

DEFAULT_MODEL = "mistral:7b"
DEFAULT_BASE_URL = "http://localhost:11434"
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CUSTOM_TASKS_DIR = os.path.join(SCRIPT_DIR, "custom_tasks")

ALL_TASKS = ["hellaswag", "mmlu", "logical_reasoning"]


def run_benchmark(
    task_list: list[str],
    model_name: str,
    base_url: str,
    limit: int | None,
    seed: int,
    output_dir: str,
) -> dict:
    """Execute lm-eval evaluator and persist results."""
    os.makedirs(output_dir, exist_ok=True)

    task_manager = tasks.TaskManager(include_path=CUSTOM_TASKS_DIR)

    model_args = (
        f"model={model_name},"
        f"base_url={base_url},"
        f"temperature=0,"
        f"top_p=1,"
        f"seed={seed},"
        f"use_cache=true"
    )

    print(f"[eval] Tasks       : {task_list}")
    print(f"[eval] Model args  : {model_args}")
    print(f"[eval] Sample limit: {limit or 'all'}")
    print(f"[eval] Seed        : {seed}")
    print()

    t0 = time.perf_counter()
    results = evaluator.simple_evaluate(
        model="ollama",
        model_args=model_args,
        tasks=task_list,
        limit=limit,
        task_manager=task_manager,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
    )
    elapsed = time.perf_counter() - t0

    # Persist raw results
    ts = time.strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(output_dir, f"raw_{ts}.json")
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[eval] Raw results → {raw_path}")

    # Build and save summary table
    summary = build_summary(results, elapsed)
    summary_path = os.path.join(output_dir, f"summary_{ts}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] Summary     → {summary_path}")

    print_summary_table(summary)
    return results


def build_summary(results: dict, elapsed: float) -> dict:
    """Extract a concise summary table from lm-eval results."""
    rows = []
    task_results = results.get("results", {})
    for task_name, metrics in task_results.items():
        row = {"task": task_name}
        for key, val in metrics.items():
            if key.startswith("acc"):
                row[key] = round(val, 4) if isinstance(val, float) else val
        row["elapsed_s"] = round(elapsed, 1)
        rows.append(row)
    return {"model": results.get("config", {}).get("model", "?"), "rows": rows}


def print_summary_table(summary: dict) -> None:
    """Pretty-print the summary to stdout."""
    print()
    print("=" * 64)
    print(f"  Model: {summary['model']}")
    print("=" * 64)
    header = f"{'Task':<25} {'acc':>8} {'acc_norm':>10} {'time(s)':>8}"
    print(header)
    print("-" * 64)
    for row in summary["rows"]:
        acc = row.get("acc", "—")
        acc_n = row.get("acc_norm", "—")
        t = row.get("elapsed_s", "—")
        if isinstance(acc, float):
            acc = f"{acc:.4f}"
        if isinstance(acc_n, float):
            acc_n = f"{acc_n:.4f}"
        print(f"{row['task']:<25} {acc:>8} {acc_n:>10} {t:>8}")
    print("=" * 64)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run lm-eval benchmarks via Ollama"
    )
    parser.add_argument(
        "--tasks",
        default=",".join(ALL_TASKS),
        help="Comma-separated task list (default: all)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--limit", type=int, default=None, help="Max samples per task"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    run_benchmark(
        task_list=task_list,
        model_name=args.model,
        base_url=args.base_url,
        limit=args.limit,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()