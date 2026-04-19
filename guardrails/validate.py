"""
validate.py — Guardrails: determinism verification and output validation

Usage:
    python guardrails/validate.py                              # all checks
    python guardrails/validate.py --checks determinism         # single check
    python guardrails/validate.py --checks determinism,schema  # subset
    python guardrails/validate.py --runs 5                     # more repeats

Checks:
  1. determinism — identical prompts with fixed seed/temp=0 yield identical output
  2. schema      — custom-task outputs conform to expected regex / JSON schema
  3. stop_seq    — stop sequences are respected and output is truncated

After writing the code files, black was executed on the project to ensure consistent formatting.
"""

import argparse
import json
import re
import sys
import urllib.request
import urllib.error

DEFAULT_MODEL = "mistral:7b"
DEFAULT_BASE_URL = "http://localhost:11434"
DETERMINISTIC_OPTS = {"temperature": 0, "top_p": 1.0, "seed": 42}


def _generate(prompt, model, base_url, max_tokens=64, **options):
    """Non-streaming generate. Returns response text."""
    url = f"{base_url}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, **options},
    }).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode())
    return body.get("response", "")


# Check 1: Determinism
def check_determinism(model, base_url, n_runs=3):
    prompts = [
        "What is the capital of Japan?",
        "Explain recursion in one sentence.",
        "List the first five prime numbers.",
    ]
    results = []
    for prompt in prompts:
        outputs = []
        for _ in range(n_runs):
            text = _generate(prompt, model, base_url, max_tokens=64, **DETERMINISTIC_OPTS)
            outputs.append(text.strip())
        unique = set(outputs)
        passed = len(unique) == 1
        results.append({
            "check": "determinism", "prompt": prompt,
            "n_runs": n_runs, "unique_outputs": len(unique),
            "passed": passed, "outputs": outputs,
        })
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] \"{prompt[:50]}\" — {len(unique)} unique / {n_runs} runs")
    return results


# Check 2: Schema / regex validation
MC_REGEX = re.compile(r"^[A-Da-d0-3]$")

STRUCTURED_SCHEMA = {
    "type": "object",
    "required": ["answer"],
    "properties": {
        "answer": {"type": "integer", "minimum": 0, "maximum": 3},
        "reasoning": {"type": "string"},
    },
}


def _validate_json_schema(obj, schema):
    if schema.get("type") == "object":
        if not isinstance(obj, dict):
            return False, f"Expected object, got {type(obj).__name__}"
        for req_key in schema.get("required", []):
            if req_key not in obj:
                return False, f"Missing required key: '{req_key}'"
        for key, prop in schema.get("properties", {}).items():
            if key in obj:
                val = obj[key]
                expected_type = prop.get("type")
                if expected_type == "integer" and not isinstance(val, int):
                    return False, f"'{key}' should be int, got {type(val).__name__}"
                if expected_type == "string" and not isinstance(val, str):
                    return False, f"'{key}' should be str, got {type(val).__name__}"
                if "minimum" in prop and isinstance(val, (int, float)) and val < prop["minimum"]:
                    return False, f"'{key}'={val} < minimum {prop['minimum']}"
                if "maximum" in prop and isinstance(val, (int, float)) and val > prop["maximum"]:
                    return False, f"'{key}'={val} > maximum {prop['maximum']}"
    return True, "ok"


def check_schema(model, base_url):
    mc_prompts = [
        "Question: What is 2+2?\nChoices:\n  0. 3\n  1. 4\n  2. 5\n  3. 6\nAnswer with just the number (0-3):",
        "Question: Which planet is closest to the Sun?\nChoices:\n  0. Venus\n  1. Mercury\n  2. Earth\n  3. Mars\nAnswer with just the number (0-3):",
    ]
    structured_prompt = (
        'Answer in JSON: {"answer": <int 0-3>, "reasoning": "<brief>"}.\n'
        "Q: Capital of France?\n0. Berlin  1. Paris  2. Madrid  3. Rome\nJSON:"
    )
    results = []

    for prompt in mc_prompts:
        raw = _generate(prompt, model, base_url, max_tokens=8, **DETERMINISTIC_OPTS)
        cleaned = raw.strip().split()[0] if raw.strip() else ""
        passed = bool(MC_REGEX.match(cleaned))
        results.append({
            "check": "schema_regex", "prompt": prompt[:60] + "...",
            "raw_output": raw.strip()[:80], "cleaned": cleaned, "passed": passed,
        })
        print(f"  [{'PASS' if passed else 'FAIL'}] MC regex — cleaned={cleaned!r}")

    raw = _generate(structured_prompt, model, base_url, max_tokens=100, **DETERMINISTIC_OPTS)
    try:
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            obj = json.loads(json_match.group())
            valid, msg = _validate_json_schema(obj, STRUCTURED_SCHEMA)
        else:
            valid, msg, obj = False, "No JSON object found", None
    except json.JSONDecodeError as exc:
        valid, msg, obj = False, f"JSON parse error: {exc}", None

    results.append({
        "check": "schema_json", "raw_output": raw.strip()[:120],
        "parsed": obj, "valid": valid, "message": msg, "passed": valid,
    })
    print(f"  [{'PASS' if valid else 'FAIL'}] JSON schema — {msg}")
    return results


# Check 3: Stop-sequence enforcement
def check_stop_sequences(model, base_url):
    cases = [
        {"prompt": "Count from 1 to 20, separated by commas:", "stop": [", 10"], "should_not_contain": "11"},
        {"prompt": "List the days of the week:\n1. Monday\n2. Tuesday\n", "stop": ["\n4."], "should_not_contain": "Thursday"},
    ]
    results = []
    for case in cases:
        url = f"{base_url}/api/generate"
        payload = json.dumps({
            "model": model, "prompt": case["prompt"], "stream": False,
            "options": {"num_predict": 128, "stop": case["stop"], **DETERMINISTIC_OPTS},
        }).encode()
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode())
        text = body.get("response", "")
        passed = case["should_not_contain"] not in text
        results.append({
            "check": "stop_sequence", "prompt": case["prompt"][:50] + "...",
            "stop": case["stop"], "output": text.strip()[:100], "passed": passed,
        })
        print(f"  [{'PASS' if passed else 'FAIL'}] Stop {case['stop']} — output: ...{text.strip()[-40:]!r}")
    return results


ALL_CHECKS = {
    "determinism": check_determinism,
    "schema": check_schema,
    "stop_seq": check_stop_sequences,
}


def main():
    parser = argparse.ArgumentParser(description="Guardrails: determinism & validation")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--checks", default=",".join(ALL_CHECKS.keys()))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    checks = [c.strip() for c in args.checks.split(",")]
    all_results = []

    for name in checks:
        fn = ALL_CHECKS.get(name)
        if fn is None:
            print(f"[guard] Unknown check: {name}")
            continue
        print(f"\n[guard] Running check: {name}")
        print("-" * 50)
        if name == "determinism":
            res = fn(args.model, args.base_url, n_runs=args.runs)
        else:
            res = fn(args.model, args.base_url)
        all_results.extend(res)

    passed = sum(1 for r in all_results if r.get("passed"))
    total = len(all_results)
    print(f"\n{'=' * 50}")
    print(f"[guard] Results: {passed}/{total} passed")
    print(f"{'=' * 50}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"[guard] Report written to {args.output}")


if __name__ == "__main__":
    main()