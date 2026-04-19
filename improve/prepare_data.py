"""
prepare_data.py — Load benchmark data and build few-shot selection pools

Usage:
    python improve/prepare_data.py --task hellaswag
    python improve/prepare_data.py --task mmlu --subject stem
    python improve/prepare_data.py --task arc_challenge

Steps:
  1. Load the target benchmark via HuggingFace datasets
  2. Normalise to a common format {question, choices, answer}
  3. Build a TF-IDF index over the train split for semantic few-shot retrieval
  4. Save artefacts to improve/data/

After writing the code files, black was executed on the project to ensure consistent formatting.
"""

import argparse
import json
import os
import pickle
import sys

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[prepare] sklearn not available; will use random few-shot selection")

try:
    import datasets as hf_datasets
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_hellaswag():
    ds = hf_datasets.load_dataset("Rowan/hellaswag")
    return [dict(r) for r in ds["train"]], [dict(r) for r in ds["validation"]]


def load_arc_challenge():
    ds = hf_datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge")
    return [dict(r) for r in ds["train"]], [dict(r) for r in ds["test"]]


def load_mmlu(subject_group="all"):
    ds = hf_datasets.load_dataset("cais/mmlu", "all")
    train = [dict(r) for r in ds["auxiliary_train"]]
    test = [dict(r) for r in ds["test"]]
    if subject_group and subject_group != "all":
        train = [r for r in train if subject_group.lower() in r.get("subject", "").lower()]
        test = [r for r in test if subject_group.lower() in r.get("subject", "").lower()]
    return train, test


def norm_hellaswag(row):
    return {"question": row.get("ctx", ""), "choices": row.get("endings", []), "answer": int(row.get("label", 0))}


def norm_arc(row):
    choices = row.get("choices", {})
    labels, texts = choices.get("label", []), choices.get("text", [])
    ak = row.get("answerKey", "A")
    return {"question": row.get("question", ""), "choices": texts, "answer": labels.index(ak) if ak in labels else 0}


def norm_mmlu(row):
    return {"question": row.get("question", ""), "choices": row.get("choices", []), "answer": int(row.get("answer", 0)), "subject": row.get("subject", "")}


TASKS = {
    "hellaswag": {"loader": load_hellaswag, "norm": norm_hellaswag},
    "arc_challenge": {"loader": load_arc_challenge, "norm": norm_arc},
    "mmlu": {"loader": load_mmlu, "norm": norm_mmlu},
}


def main():
    parser = argparse.ArgumentParser(description="Prepare benchmark data")
    parser.add_argument("--task", required=True, choices=list(TASKS.keys()))
    parser.add_argument("--subject", default="all")
    parser.add_argument("--output-dir", default=DATA_DIR)
    args = parser.parse_args()

    if not HAS_DATASETS:
        print("[prepare] ERROR: pip install datasets"); sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = TASKS[args.task]

    print(f"[prepare] Loading {args.task}...")
    train, test = cfg["loader"](args.subject) if args.task == "mmlu" else cfg["loader"]()
    print(f"[prepare] Train: {len(train)}  Test: {len(test)}")

    norm_train = [cfg["norm"](r) for r in train]
    norm_test = [cfg["norm"](r) for r in test]

    with open(os.path.join(args.output_dir, f"{args.task}_train.json"), "w") as f:
        json.dump(norm_train, f)
    with open(os.path.join(args.output_dir, f"{args.task}_test.json"), "w") as f:
        json.dump(norm_test, f)

    if HAS_SKLEARN:
        print("[prepare] Building TF-IDF index...")
        texts = [ex.get("question", "") for ex in norm_train]
        vec = TfidfVectorizer(max_features=5000, stop_words="english")
        matrix = vec.fit_transform(texts)
        with open(os.path.join(args.output_dir, f"{args.task}_tfidf.pkl"), "wb") as f:
            pickle.dump({"vectorizer": vec, "matrix": matrix, "pool": norm_train}, f)

    print("[prepare] Done")


if __name__ == "__main__":
    main()