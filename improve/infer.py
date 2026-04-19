"""
infer.py — Run optimised inference against the Ollama endpoint

Usage:
    python improve/infer.py --task hellaswag --strategy template,few_shot,cot
    python improve/infer.py --task arc_challenge --strategy self_consistency --k 5
    python improve/infer.py --task mmlu --limit 200 --strategy all

After writing the code files, black was executed on the project to ensure consistent formatting.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimize_prompt import build_optimized_prompt, build_template_prompt, load_tfidf_index

DEFAULT_MODEL = "mistral:7b"
DEFAULT_BASE_URL = "http://localhost:11434"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PRED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions")

LETTER_RE = re.compile(r"\b([A-Da-d])\b")
NUMBER_RE = re.compile(r"\b([0-3])\b")
LETTER_MAP = {"a": 0, "b": 1, "c": 2, "d": 3}


def generate(prompt, model, base_url, max_tokens=128, temperature=0.0, seed=42, stop=None):
    url = f"{base_url}/api/generate"
    options = {"num_predict": max_tokens, "temperature": temperature, "top_p": 1.0, "seed": seed}
    if stop:
        options["stop"] = stop
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False, "options": options}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode()).get("response", "")


def extract_answer(text, n_choices=4):
    """Extract predicted answer index. Tries multiple strategies."""
    labels = "ABCD"[:n_choices]
    text_clean = text.strip()

    # Strategy 1: look for "ANSWER: X" pattern (from CoT prompts)
    m = re.search(r"ANSWER:\s*([A-Da-d])", text_clean, re.IGNORECASE)
    if m:
        idx = LETTER_MAP.get(m.group(1).lower())
        if idx is not None and idx < n_choices:
            return idx

    # Strategy 2: last line often has the answer after reasoning
    last_line = text_clean.strip().split("\n")[-1].strip()
    m = LETTER_RE.search(last_line)
    if m:
        idx = LETTER_MAP.get(m.group(1).lower())
        if idx is not None and idx < n_choices:
            return idx

    # Strategy 3: if response is very short (1-3 chars), take first letter
    if len(text_clean) <= 3:
        m = LETTER_RE.search(text_clean)
        if m:
            idx = LETTER_MAP.get(m.group(1).lower())
            if idx is not None and idx < n_choices:
                return idx

    # Strategy 4: first line (for non-CoT simple responses)
    first_line = text_clean.split("\n")[0].strip()
    if len(first_line) <= 5:
        m = LETTER_RE.search(first_line)
        if m:
            idx = LETTER_MAP.get(m.group(1).lower())
            if idx is not None and idx < n_choices:
                return idx

    # Strategy 5: digit match
    m = NUMBER_RE.search(last_line)
    if m and int(m.group(1)) < n_choices:
        return int(m.group(1))

    return None


def self_consistent_predict(prompt, n_choices, model, base_url, k=5, max_tokens=256):
    votes = []
    for i in range(k):
        text = generate(prompt, model, base_url, max_tokens=max_tokens, temperature=0.7, seed=42 + i)
        ans = extract_answer(text, n_choices)
        if ans is not None:
            votes.append(ans)
    if not votes:
        return None, {"votes": []}
    return Counter(votes).most_common(1)[0][0], {"votes": votes, "counts": dict(Counter(votes))}


def ensemble_predict(prompts, n_choices, model, base_url, max_tokens=128):
    votes = []
    for i, p in enumerate(prompts):
        text = generate(p, model, base_url, max_tokens=max_tokens, seed=42 + i)
        ans = extract_answer(text, n_choices)
        if ans is not None:
            votes.append(ans)
    if not votes:
        return None, {"votes": []}
    return Counter(votes).most_common(1)[0][0], {"votes": votes, "counts": dict(Counter(votes))}


def run_inference(task, strategies, model, base_url, limit, sc_k, few_shot_k, max_tokens):
    test_path = os.path.join(DATA_DIR, f"{task}_test.json")
    if not os.path.exists(test_path):
        print(f"[infer] ERROR: run prepare_data.py --task {task} first"); sys.exit(1)

    with open(test_path) as f:
        test_data = json.load(f)
    if limit:
        test_data = test_data[:limit]

    index = load_tfidf_index(task)
    use_sc = "self_consistency" in strategies
    use_ens = "ensemble" in strategies
    use_cot = "cot" in strategies or use_sc
    effective_max_tokens = max(max_tokens, 512) if use_cot else max_tokens

    print(f"[infer] Task={task}  strategies={strategies}  items={len(test_data)}")

    predictions, baseline_predictions = [], []
    correct = baseline_correct = total = 0
    t0 = time.perf_counter()

    for i, item in enumerate(test_data):
        n_ch = len(item.get("choices", []))
        gold = item.get("answer", 0)

        # Baseline
        bp = build_template_prompt(item, task, variant="baseline")
        bt = generate(bp, model, base_url, max_tokens=max_tokens, seed=42)
        ba = extract_answer(bt, n_ch)
        bok = ba == gold
        if bok: baseline_correct += 1
        baseline_predictions.append({"idx": i, "gold": gold, "pred": ba, "correct": bok, "raw": bt.strip()[:200]})

        # Optimised
        prompt = build_optimized_prompt(item, task, strategies, index=index, few_shot_k=few_shot_k)
        if use_ens and isinstance(prompt, list):
            pred, meta = ensemble_predict(prompt, n_ch, model, base_url, effective_max_tokens)
        elif use_sc:
            if isinstance(prompt, list): prompt = prompt[0]
            pred, meta = self_consistent_predict(prompt, n_ch, model, base_url, k=sc_k, max_tokens=effective_max_tokens)
        else:
            if isinstance(prompt, list): prompt = prompt[0]
            raw = generate(prompt, model, base_url, max_tokens=effective_max_tokens, seed=42)
            pred = extract_answer(raw, n_ch)
            meta = {"raw": raw.strip()[:200]}

        ok = pred == gold
        if ok: correct += 1
        total += 1
        predictions.append({"idx": i, "gold": gold, "pred": pred, "correct": ok, **meta})

        if (i + 1) % 20 == 0 or i == len(test_data) - 1:
            bl_acc = baseline_correct / total * 100
            opt_acc = correct / total * 100
            print(f"  [{i + 1}/{len(test_data)}]  baseline={bl_acc:.1f}%  optimised={opt_acc:.1f}%  delta={opt_acc - bl_acc:+.1f}%")

    elapsed = time.perf_counter() - t0
    bl_acc = baseline_correct / total * 100
    opt_acc = correct / total * 100

    results = {
        "task": task, "strategies": strategies, "model": model, "n_items": total,
        "baseline_accuracy": round(bl_acc, 2), "optimised_accuracy": round(opt_acc, 2),
        "delta": round(opt_acc - bl_acc, 2), "elapsed_s": round(elapsed, 1),
        "predictions": predictions, "baseline_predictions": baseline_predictions,
    }

    os.makedirs(PRED_DIR, exist_ok=True)
    out_path = os.path.join(PRED_DIR, f"{task}_{'_'.join(strategies)}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[infer] Results -> {out_path}")
    print(f"[infer] Baseline: {bl_acc:.2f}%  Optimised: {opt_acc:.2f}%  Delta: {opt_acc - bl_acc:+.2f}%")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run optimised inference")
    parser.add_argument("--task", required=True, choices=["hellaswag", "arc_challenge", "mmlu"])
    parser.add_argument("--strategy", default="template,few_shot,cot")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sc-k", type=int, default=5)
    parser.add_argument("--few-shot-k", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    strategies = (
        ["template", "few_shot", "cot"] if args.strategy == "all"
        else [s.strip() for s in args.strategy.split(",")]
    )
    run_inference(args.task, strategies, args.model, args.base_url, args.limit, args.sc_k, args.few_shot_k, args.max_tokens)


if __name__ == "__main__":
    main()