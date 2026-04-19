#!/usr/bin/env bash
# eval.sh — End-to-end benchmark improvement pipeline
# Usage:  bash improve/eval.sh hellaswag [--limit 200]
set -euo pipefail

TASK="${1:-hellaswag}"
LIMIT="${2:---limit 200}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo " Benchmark Improvement: ${TASK}"
echo "========================================"

echo "[1/4] Preparing data..."
python prepare_data.py --task "$TASK"

echo "[2/4] Baseline + optimised..."
python infer.py --task "$TASK" --strategy template,few_shot,cot $LIMIT

echo "[3/4] Ablation studies..."
for S in template few_shot cot; do
    echo "  -> $S"
    python infer.py --task "$TASK" --strategy "$S" $LIMIT
done

echo "[4/4] Self-consistency..."
python infer.py --task "$TASK" --strategy self_consistency --sc-k 5 $LIMIT

echo ""
echo "Done. Results in predictions/"
ls -la predictions/${TASK}_*.json 2>/dev/null || echo "(no files)"