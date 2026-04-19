"""
load_test.py — Concurrent load generator for an Ollama inference endpoint

Usage:
    python perf/load_test.py                         # default sweep
    python perf/load_test.py --concurrency 1,4,8     # specific levels
    python perf/load_test.py --prompts short          # only short prompts
    python perf/load_test.py --runs 5                 # repeats per config

Metrics collected per request:
  • TTFT  — time to first token (seconds)
  • TPOT  — tokens per second (total tokens / generation time)
  • Total latency (seconds)

Aggregate statistics:  P50 / P95 / P99 latency, mean TTFT, mean TPOT,
GPU utilisation (if nvidia-smi is available).

Results are written to perf/metrics.csv for downstream analysis.

After writing the code files, black was executed on the project to ensure consistent formatting.
"""

import argparse
import csv
import json
import os
import statistics
import subprocess
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict

DEFAULT_MODEL = "mistral:7b"
DEFAULT_BASE_URL = "http://localhost:11434"
METRICS_DIR = os.path.dirname(os.path.abspath(__file__))

SHORT_PROMPTS = [
    "What is 2 + 2?",
    "Name three primary colours.",
    "Say hello in French.",
    "What year did WW2 end?",
    "Define 'entropy' in one sentence.",
]

LONG_PROMPTS = [
    (
        "Write a detailed explanation of how the TCP/IP networking model works, "
        "covering each of the four layers, their responsibilities, key protocols "
        "at each layer, and how data flows from an application on one host to an "
        "application on another host across the internet."
    ),
    (
        "Explain the complete lifecycle of a machine learning project, from "
        "problem definition and data collection through feature engineering, "
        "model selection, training, evaluation, deployment, and monitoring. "
        "Include best practices for each stage."
    ),
    (
        "Describe the history of space exploration from Sputnik to the "
        "International Space Station, covering the major milestones, missions, "
        "agencies involved, technological breakthroughs, and the geopolitical "
        "context of the Space Race."
    ),
]


@dataclass
class RequestMetrics:
    prompt_type: str
    concurrency: int
    ttft: float
    tpot: float
    total_latency: float
    tokens_generated: int
    prompt_length: int
    error: str = ""


def _stream_request(prompt: str, model: str, base_url: str, max_tokens: int = 128) -> dict:
    """Send a streaming request and measure TTFT + throughput."""
    url = f"{base_url}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": max_tokens, "temperature": 0.7},
    }).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST",
    )
    t_start = time.perf_counter()
    t_first_token = None
    token_count = 0

    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw_line in resp:
            line = raw_line.decode().strip()
            if not line:
                continue
            chunk = json.loads(line)
            tok = chunk.get("response", "")
            if tok and t_first_token is None:
                t_first_token = time.perf_counter()
            if tok:
                token_count += 1
            if chunk.get("done"):
                break

    t_end = time.perf_counter()
    total = t_end - t_start
    ttft = (t_first_token - t_start) if t_first_token else total
    tpot = token_count / total if total > 0 else 0
    return {"ttft": ttft, "tpot": tpot, "total_latency": total, "tokens_generated": token_count}


def run_single(prompt, prompt_type, concurrency, model, base_url, max_tokens) -> RequestMetrics:
    """Execute one request and return its metrics (or an error)."""
    try:
        m = _stream_request(prompt, model, base_url, max_tokens)
        return RequestMetrics(
            prompt_type=prompt_type, concurrency=concurrency,
            ttft=m["ttft"], tpot=m["tpot"], total_latency=m["total_latency"],
            tokens_generated=m["tokens_generated"], prompt_length=len(prompt),
        )
    except Exception as exc:
        return RequestMetrics(
            prompt_type=prompt_type, concurrency=concurrency,
            ttft=0, tpot=0, total_latency=0, tokens_generated=0,
            prompt_length=len(prompt), error=str(exc),
        )


def get_gpu_utilisation() -> str:
    """Return GPU utilisation string, or 'N/A'."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            timeout=5,
        )
        return out.decode().strip() + "%"
    except (FileNotFoundError, subprocess.SubprocessError):
        return "N/A"


def percentile(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * pct / 100.0
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    return s[f] + (k - f) * (s[c] - s[f])


def run_sweep(model, base_url, concurrency_levels, prompt_types, runs_per_config, max_tokens):
    """Run the full parameter sweep and collect all metrics."""
    all_metrics = []
    prompts_map = {}
    if "short" in prompt_types:
        prompts_map["short"] = SHORT_PROMPTS
    if "long" in prompt_types:
        prompts_map["long"] = LONG_PROMPTS

    total_configs = len(concurrency_levels) * len(prompts_map) * runs_per_config
    print(f"[perf] Sweep: {len(concurrency_levels)} concurrency x "
          f"{len(prompts_map)} prompt types x {runs_per_config} runs = "
          f"{total_configs} configurations")
    print(f"[perf] GPU utilisation: {get_gpu_utilisation()}\n")

    config_idx = 0
    for conc in concurrency_levels:
        for ptype, prompts in prompts_map.items():
            for run in range(runs_per_config):
                config_idx += 1
                print(f"[perf] Config {config_idx}/{total_configs}: "
                      f"concurrency={conc}  type={ptype}  run={run + 1}/{runs_per_config}")
                with ThreadPoolExecutor(max_workers=conc) as pool:
                    futures = []
                    for i in range(conc):
                        p = prompts[i % len(prompts)]
                        futures.append(pool.submit(
                            run_single, p, ptype, conc, model, base_url, max_tokens
                        ))
                    for fut in as_completed(futures):
                        m = fut.result()
                        all_metrics.append(m)
                        if m.error:
                            print(f"         ERROR: {m.error}")
    return all_metrics


def write_csv(metrics, path):
    fieldnames = [
        "prompt_type", "concurrency", "ttft", "tpot",
        "total_latency", "tokens_generated", "prompt_length", "error",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics:
            writer.writerow(asdict(m))


def print_aggregate(metrics):
    """Print aggregate stats grouped by (concurrency, prompt_type)."""
    groups = {}
    for m in metrics:
        if m.error:
            continue
        key = (m.concurrency, m.prompt_type)
        groups.setdefault(key, []).append(m)

    gpu = get_gpu_utilisation()
    print(f"\n{'=' * 80}")
    print(f"{'Conc':>5} {'Type':<6} {'N':>4} {'TTFT_mean':>10} "
          f"{'TPOT_mean':>10} {'P50_lat':>9} {'P95_lat':>9} {'P99_lat':>9} {'GPU':>6}")
    print("-" * 80)
    for (conc, ptype), group in sorted(groups.items()):
        lats = [m.total_latency for m in group]
        ttfts = [m.ttft for m in group]
        tpots = [m.tpot for m in group]
        print(f"{conc:>5} {ptype:<6} {len(group):>4} "
              f"{statistics.mean(ttfts):>10.4f} {statistics.mean(tpots):>10.2f} "
              f"{percentile(lats, 50):>9.4f} {percentile(lats, 95):>9.4f} "
              f"{percentile(lats, 99):>9.4f} {gpu:>6}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Ollama load generator")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--prompts", default="short,long")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", default=os.path.join(METRICS_DIR, "metrics.csv"))
    args = parser.parse_args()

    conc_levels = [int(c) for c in args.concurrency.split(",")]
    prompt_types = [p.strip() for p in args.prompts.split(",")]

    print(f"[perf] Model    : {args.model}")
    print(f"[perf] Endpoint : {args.base_url}")
    print(f"[perf] Output   : {args.output}\n")

    metrics = run_sweep(args.model, args.base_url, conc_levels, prompt_types, args.runs, args.max_tokens)
    write_csv(metrics, args.output)
    print(f"[perf] Wrote {len(metrics)} rows -> {args.output}")
    print_aggregate(metrics)


if __name__ == "__main__":
    main()