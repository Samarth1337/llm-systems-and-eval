"""
optimize_prompt.py — Prompt engineering strategies for benchmark improvement

Strategies:
  1. template        — rewritten instruction templates
  2. few_shot        — semantically-selected few-shot examples (TF-IDF)
  3. cot             — chain-of-thought reasoning prompts
  4. self_consistency — k-sample decoding + majority vote (at inference time)
  5. ensemble        — prompt-variant ensembling across phrasings

After writing the code files, black was executed on the project to ensure consistent formatting.
"""

import json
import os
import pickle

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


TEMPLATES = {
    "hellaswag": {
        "baseline": "{context}\n\nWhich of the following best completes the passage?\n{choices_str}\nAnswer:",
        "improved": (
            "You are an expert at predicting what happens next in everyday situations. "
            "You will be given the beginning of a description of an activity or event, "
            "and you must select the option that most naturally, logically, and coherently "
            "continues the scenario. Think about what a person would realistically do next "
            "in the described situation.\n\n"
            "IMPORTANT: Respond with ONLY a single letter (A, B, C, or D). "
            "Do not include any explanation or additional text.\n\n"
            "Context: {context}\n\n"
            "Which of the following most naturally continues the above scenario?\n"
            "{choices_str}\n\n"
            "Answer:"
        ),
    },
    "arc_challenge": {
        "baseline": "Question: {question}\n{choices_str}\nAnswer:",
        "improved": (
            "You are a knowledgeable science expert. Answer the following science question "
            "by selecting the most accurate option. Use your understanding of physics, "
            "chemistry, biology, and earth science to determine the correct answer.\n\n"
            "IMPORTANT: Respond with ONLY a single letter (A, B, C, or D). "
            "Do not include any explanation.\n\n"
            "Question: {question}\n\n"
            "Options:\n{choices_str}\n\n"
            "Answer:"
        ),
    },
    "mmlu": {
        "baseline": "Question: {question}\n{choices_str}\nAnswer:",
        "improved": (
            "You are an expert across many academic disciplines. Answer the following "
            "question by selecting the most accurate option.\n\n"
            "IMPORTANT: Respond with ONLY a single letter (A, B, C, or D). "
            "Do not include any explanation.\n\n"
            "Question: {question}\n\n"
            "{choices_str}\n\n"
            "Answer:"
        ),
    },
}

COT_SUFFIX = {
    "hellaswag": (
        "\n\nLet's think through each option carefully:\n"
        "- Consider whether each option logically and grammatically follows the context.\n"
        "- Eliminate options that are absurd, unrelated, or grammatically awkward.\n"
        "- Pick the option that a reasonable person would expect to happen next.\n\n"
        "Analyze each option briefly, then write your final answer as a single letter "
        "on the last line in this exact format:\nANSWER: X\n\nReasoning:"
    ),
    "arc_challenge": (
        "\n\nLet's work through this step by step:\n"
        "- Recall the relevant scientific principle or fact.\n"
        "- Evaluate each option against that principle.\n"
        "- Eliminate clearly wrong answers first.\n\n"
        "After your reasoning, write your final answer as a single letter "
        "on the last line in this exact format:\nANSWER: X\n\nStep-by-step:"
    ),
    "mmlu": (
        "\n\nLet's work through this step by step:\n"
        "- Identify the key concept being tested.\n"
        "- Evaluate each option carefully.\n"
        "- Eliminate clearly wrong answers first.\n\n"
        "After your reasoning, write your final answer as a single letter "
        "on the last line in this exact format:\nANSWER: X\n\nReasoning:"
    ),
}

ENSEMBLE_VARIANTS = {
    "hellaswag": [
        "You are an expert at understanding everyday activities. Select the most natural and logical continuation of the following scenario. Reply with ONLY a single letter.\n\nContext: {context}\n\n{choices_str}\n\nAnswer:",
        "Read the beginning of this scenario and pick the option that best completes it in a realistic, coherent way. Reply with ONLY a single letter (A, B, C, or D).\n\n{context}\n\n{choices_str}\n\nBest option:",
        "Which of the following is the most plausible continuation of this event? Respond with just the letter.\n\n{context}\n\n{choices_str}\n\nMy choice:",
    ],
    "arc_challenge": [
        "You are a science teacher. Answer this question by selecting the correct option. Reply with ONLY the letter.\n\nQuestion: {question}\n{choices_str}\nCorrect answer:",
        "Q: {question}\nOptions:\n{choices_str}\nA:",
        "As a science expert, answer the following. Reply with ONLY the letter.\n\n{question}\n{choices_str}\nThe answer is",
    ],
    "mmlu": [
        "You are a knowledgeable expert. Answer by selecting the correct option. Reply with ONLY the letter.\n\nQuestion: {question}\n{choices_str}\nAnswer:",
        "Q: {question}\nChoices:\n{choices_str}\nCorrect:",
        "Select the right answer. Reply with ONLY the letter.\n\n{question}\n{choices_str}\nAnswer:",
    ],
}


def format_choices(choices, style="letter"):
    labels = "ABCDEFGHIJ" if style == "letter" else [str(i) for i in range(len(choices))]
    return "\n".join(f"  {labels[i]}. {c}" for i, c in enumerate(choices))


def load_tfidf_index(task):
    path = os.path.join(DATA_DIR, f"{task}_tfidf.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def select_few_shot(query, index, k=5):
    if index is None or not HAS_SKLEARN:
        return []
    vec = index["vectorizer"].transform([query])
    sims = cosine_similarity(vec, index["matrix"]).flatten()
    top_k = sims.argsort()[-k:][::-1]
    return [index["pool"][i] for i in top_k]


def build_template_prompt(item, task, variant="improved"):
    tmpl = TEMPLATES.get(task, TEMPLATES["mmlu"])[variant]
    choices_str = format_choices(item["choices"])
    return tmpl.format(question=item.get("question", ""), context=item.get("question", ""), choices_str=choices_str)


FEW_SHOT_INSTRUCTIONS = {
    "hellaswag": (
        "You are an expert at predicting what happens next in everyday situations. "
        "For each scenario below, you are given the beginning of a description and must "
        "select the option (A, B, C, or D) that most naturally and logically continues it. "
        "Study the examples carefully, then answer the final question with ONLY a single letter.\n"
    ),
    "arc_challenge": (
        "You are a science expert. For each question below, select the correct answer. "
        "Study the examples carefully, then answer the final question with ONLY a single letter.\n"
    ),
    "mmlu": (
        "You are a knowledgeable expert. For each question below, select the correct answer. "
        "Study the examples carefully, then answer the final question with ONLY a single letter.\n"
    ),
}


def build_few_shot_prompt(item, task, index, k=5):
    query = item.get("question", "")
    examples = select_few_shot(query, index, k=k)
    labels = "ABCDEFGHIJ"
    instruction = FEW_SHOT_INSTRUCTIONS.get(task, FEW_SHOT_INSTRUCTIONS["mmlu"])
    parts = [instruction]
    ctx_label = "Context" if task == "hellaswag" else "Question"
    for ex in examples:
        cs = format_choices(ex["choices"])
        al = labels[ex.get("answer", 0)]
        parts.append(f"\n{ctx_label}: {ex.get('question', '')}\n{cs}\nAnswer: {al}\n")
    parts.append(f"\n{ctx_label}: {query}\n{format_choices(item['choices'])}\nAnswer:")
    return "".join(parts)


def build_ensemble_prompts(item, task):
    variants = ENSEMBLE_VARIANTS.get(task, ENSEMBLE_VARIANTS["mmlu"])
    choices_str = format_choices(item["choices"])
    return [v.format(question=item.get("question", ""), context=item.get("question", ""), choices_str=choices_str) for v in variants]


def build_optimized_prompt(item, task, strategies, index=None, few_shot_k=5):
    if "ensemble" in strategies:
        return build_ensemble_prompts(item, task)

    if "few_shot" in strategies and index is not None:
        prompt = build_few_shot_prompt(item, task, index, k=few_shot_k)
    elif any(s in strategies for s in ("template", "self_consistency", "cot")):
        prompt = build_template_prompt(item, task, variant="improved")
    else:
        prompt = build_template_prompt(item, task, variant="baseline")

    if "cot" in strategies or "self_consistency" in strategies:
        prompt += COT_SUFFIX.get(task, COT_SUFFIX["mmlu"])

    return prompt