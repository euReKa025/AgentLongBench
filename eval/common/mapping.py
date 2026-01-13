"""Standardized labels and routing helpers for AgentLong benchmark."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

# Knowledge types (from path abbreviations)
KNOWLEDGE_TYPE_LABELS = {
    "ki": "knowledge_intensive",
    "kf": "knowledge_free",
}

# History types (from path abbreviations)
HISTORY_TYPE_LABELS = {
    "c": "Concise-Response",
    "v": "Verbose-Response",
}

# Standard question types (paper labels)
COUNT_FREQUENCY_TOOL = "Count Frequency(Tool)"
FIND_DUPLICATES_TOOL = "Find Duplicates(Tool)"
FIND_TARGET_OFFSETS_TOOL = "Find Target Offsets(Tool)"
COUNT_CORRECTNESS_ENV = "Count Correctness(Env)"
COUNT_FREQUENCY_ENV = "Count Frequency(Env)"
FIND_ROUND_LARGEST_VALUE_ENV = "Find Round with Largest Value(Env)"
WEIGHTED_SUMMATION_ENV = "Weighted Summation(Env)"
INTERSECTION = "Intersection"

QUESTION_TYPES = {
    COUNT_FREQUENCY_TOOL,
    FIND_DUPLICATES_TOOL,
    FIND_TARGET_OFFSETS_TOOL,
    COUNT_CORRECTNESS_ENV,
    COUNT_FREQUENCY_ENV,
    FIND_ROUND_LARGEST_VALUE_ENV,
    WEIGHTED_SUMMATION_ENV,
    INTERSECTION,
}

TOOL_RESPONSE_TYPES = {
    COUNT_FREQUENCY_TOOL,
    FIND_DUPLICATES_TOOL,
    FIND_TARGET_OFFSETS_TOOL,
}

ENV_RESPONSE_TYPES = {
    COUNT_CORRECTNESS_ENV,
    COUNT_FREQUENCY_ENV,
    FIND_ROUND_LARGEST_VALUE_ENV,
    WEIGHTED_SUMMATION_ENV,
}

FINAL_GUESS_TYPES = {INTERSECTION}

CATEGORY_DIRS = {
    **{q: "tool_response" for q in TOOL_RESPONSE_TYPES},
    **{q: "env_response" for q in ENV_RESPONSE_TYPES},
    **{q: "final_guess" for q in FINAL_GUESS_TYPES},
}

QUESTION_TYPE_SLUGS = {
    COUNT_FREQUENCY_TOOL: "count_frequency_tool",
    FIND_DUPLICATES_TOOL: "find_duplicates_tool",
    FIND_TARGET_OFFSETS_TOOL: "find_target_offsets_tool",
    COUNT_CORRECTNESS_ENV: "count_correctness_env",
    COUNT_FREQUENCY_ENV: "count_frequency_env",
    FIND_ROUND_LARGEST_VALUE_ENV: "find_round_largest_value_env",
    WEIGHTED_SUMMATION_ENV: "weighted_summation_env",
    INTERSECTION: "intersection",
}


def infer_context_from_path(path: Path) -> Tuple[str, str, str, str]:
    """Infer knowledge/history types from a dataset path.

    Returns: (knowledge_key, knowledge_label, history_key, history_label)
    """
    parts = [p.lower() for p in path.parts]
    match = None
    for part in parts:
        if re.fullmatch(r"k[if]-[cv]", part):
            match = part
            break
    if not match:
        raise ValueError(f"Cannot infer knowledge/history type from path: {path}")

    knowledge_key, history_key = match.split("-")
    knowledge_label = KNOWLEDGE_TYPE_LABELS.get(knowledge_key)
    history_label = HISTORY_TYPE_LABELS.get(history_key)
    if not knowledge_label or not history_label:
        raise ValueError(f"Invalid knowledge/history abbreviation in path: {match}")
    return knowledge_key, knowledge_label, history_key, history_label


def require_single_question_type(rows: Iterable[Dict]) -> str:
    types = {row.get("question_type") for row in rows if row.get("question_type")}
    if not types:
        raise ValueError("Dataset has no question_type field.")
    if len(types) != 1:
        raise ValueError(f"Dataset contains multiple question_type values: {sorted(types)}")
    question_type = next(iter(types))
    if question_type not in QUESTION_TYPES:
        raise ValueError(f"Unknown question_type: {question_type}")
    return question_type
