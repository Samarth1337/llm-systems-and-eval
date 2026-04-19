# Guardrails & Determinism

## What was tested

### 1. Determinism (`--checks determinism`)
Verifies identical prompts produce identical outputs with `temperature=0, top_p=1.0, seed=42`.
Three prompts are each run N times; the check passes if all N outputs are byte-identical.

### 2. Schema validation (`--checks schema`)
- **Regex** — multiple-choice outputs checked against `^[A-Da-d0-3]$`.
- **JSON schema** — structured-output prompt validated for required fields, types, bounds.

### 3. Stop-sequence enforcement (`--checks stop_seq`)
Prompts with explicit stop sequences verified to not contain text past the stop token.

## Where nondeterminism persists

Even with `temperature=0, top_p=1, seed=42`:

1. **GPU floating-point** — CUDA kernel scheduling is not bit-reproducible unless
   `CUBLAS_WORKSPACE_CONFIG` is set (Ollama does not expose this).
2. **KV-cache state** — warm cache from a prior request can change the numerical path.
   Restart Ollama between comparative runs to eliminate this.
3. **Batching** — concurrent requests change operation order. Use `concurrency=1`.
4. **Quantisation** — Q4/Q5 rounding can flip a token when logits are close.
   FP16 weights are most reproducible.

## Recommendations

- Always use `temperature=0, top_p=1, seed=42, concurrency=1` for evals.
- Restart Ollama before comparative runs to flush KV-cache.
- Use the SQLite prompt cache in `eval_runner/model.py` to lock in first-pass results.
- Log Ollama version and model digest for full reproducibility.

## Running

```bash
python guardrails/validate.py                          # all checks
python guardrails/validate.py --checks determinism --runs 10
python guardrails/validate.py --output guardrails/report.json
```