"""Prompt and parsing logic for standardized question types."""

from __future__ import annotations

import ast
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

from .mapping import (
    COUNT_FREQUENCY_TOOL,
    FIND_DUPLICATES_TOOL,
    FIND_TARGET_OFFSETS_TOOL,
    COUNT_CORRECTNESS_ENV,
    COUNT_FREQUENCY_ENV,
    FIND_ROUND_LARGEST_VALUE_ENV,
    WEIGHTED_SUMMATION_ENV,
    INTERSECTION,
)


def _extract_answer_tag(text: str) -> Optional[str]:
    if not text:
        return None
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if not matches:
        return None
    return matches[-1].strip()


def parse_number(text: str) -> Optional[int]:
    if not text:
        return None
    inner = _extract_answer_tag(text)
    if inner is None:
        return None
    pattern = r"-?\d[\d,]*(?:\.\d+)?"
    matches = re.findall(pattern, inner)
    if not matches:
        return None
    for candidate in reversed(matches):
        try:
            clean = candidate.replace(",", "")
            return int(float(clean))
        except ValueError:
            continue
    return None


def parse_boolean(text: str) -> Optional[bool]:
    if not text:
        return None
    inner = _extract_answer_tag(text)
    if inner is None:
        return None
    text_lower = inner.lower()
    neg_pattern = r"\b(no|false|not|doesn't|does not|none|neither)\b"
    if re.search(neg_pattern, text_lower):
        return False
    pos_pattern = r"\b(yes|true|contain|contains|appear|appears|does|both)\b"
    if re.search(pos_pattern, text_lower):
        return True
    match = re.search(r"-?\d[\d,]*", inner)
    if match:
        try:
            return int(float(match.group(0).replace(",", ""))) > 0
        except ValueError:
            return None
    return None


def parse_pair_list(text: str) -> Optional[List[str]]:
    if not text:
        return None
    inner = _extract_answer_tag(text)
    if inner is None:
        return None
    text = inner.strip()

    if text.startswith("[") and text.endswith("]"):
        try:
            arr = ast.literal_eval(text)
            if isinstance(arr, list):
                cleaned = [str(x).strip() for x in arr if str(x).strip()]
                if len(cleaned) >= 2:
                    return cleaned
        except (ValueError, SyntaxError):
            pass

    normalized = re.sub(r"(?i)\band\b", ",", text)
    normalized = re.sub(r"[\n;|]", ",", normalized)

    tokens: List[str] = []
    for chunk in normalized.split(","):
        item = re.sub(r"^\d+\.?\s*", "", chunk).strip()
        if item:
            tokens.append(item)
    return tokens if tokens else None


def parse_intersection_list(text: str) -> List[str]:
    if not text:
        return []
    inner = _extract_answer_tag(text)
    if inner is None:
        return []
    text = inner.strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            arr = ast.literal_eval(text)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except (ValueError, SyntaxError):
            pass

    normalized = re.sub(r"(?i)\band\b", ",", text)
    normalized = re.sub(r"[\n;|]", ",", normalized)
    parts = [chunk.strip() for chunk in normalized.split(",") if chunk.strip()]
    return parts


def parse_final_guess(text: str) -> Optional[str]:
    if not text:
        return None
    inner = _extract_answer_tag(text)
    if inner is None:
        return None
    return inner.strip() or None


def build_prompt(
    question_type: str,
    sample: Dict[str, Any],
    history_label: str,
    knowledge_label: str,
) -> List[Dict[str, str]]:
    if knowledge_label == "knowledge_free":
        sys_prompt = _build_masked_prompt(question_type, history_label)
    else:
        sys_prompt = _build_pokemon_prompt(question_type, history_label)

    messages = [{"role": "system", "content": sys_prompt}]
    for msg in sample.get("messages") or []:
        if msg.get("role") == "system":
            continue
        messages.append(deepcopy(msg))
    messages.append({"role": "user", "content": sample.get("question")})
    return messages


def _build_pokemon_prompt(question_type: str, history_label: str) -> str:
    if question_type in {COUNT_FREQUENCY_TOOL, COUNT_CORRECTNESS_ENV}:
        return (
            "You are analyzing a guess-the-Pokemon dialogue. Full conversation history (including tool results and feedback) is provided. "
            "Answer the question based on the tool return values or environment feedback. "
            "Wrap your answer in <answer></answer>."
            "If the answer is a number, answer in arabic numerals (e.g., 3 not three)."
        )
    if question_type == COUNT_FREQUENCY_ENV:
        return (
            "You are analyzing a guess-the-Pokemon dialogue. Full conversation history with feedback is provided.\n"
            "Answer the question by counting occurrences of a property value across all rounds' feedback.\n"
            "Wrap your final answer (a number) in <answer></answer>."
        )
    if question_type == FIND_ROUND_LARGEST_VALUE_ENV:
        return (
            "You are analyzing a guess-the-Pokemon dialogue. Full conversation history with feedback is provided.\n"
            "Answer the question by identifying which round has the highest total base stats.\n"
            "Wrap your final answer (round number) in <answer></answer>."
        )
    if question_type == WEIGHTED_SUMMATION_ENV:
        return (
            "You are analyzing a guess-the-Pokemon dialogue. Full conversation history with feedback is provided.\n"
            "Calculate the weighted scores for two rounds using this weighted rule:\n"
            "- Type: 6 points per correct item\n"
            "- Ability: 5 points per correct item\n"
            "- Base Stats: 4 points per correct item\n"
            "- Evolution: 3 points per correct item\n"
            "- Generation: 2 points per correct item\n"
            "- Other sections: 1 point per correct item\n"
            "Example: If a round has Type correct (6) + 2 Abilities correct (5+5) + 1 Evolution correct (3), score = 6+5+5+3 = 19.\n"
            "Then compute the absolute difference between the two rounds' scores.\n"
            "Wrap your final answer (difference value) in <answer></answer>."
        )
    if question_type == FIND_DUPLICATES_TOOL:
        return (
            "You are analyzing a guess-the-Pokemon dialogue. Full conversation history is provided.\n"
            "Answer the question with yes/no or true/false based on whether a Pokemon appears in both tool results.\n"
            "Wrap your final answer in <answer></answer>."
        )
    if question_type == FIND_TARGET_OFFSETS_TOOL:
        return (
            "You are analyzing a guess-the-Pokemon dialogue. Full conversation history is provided.\n"
            "Answer the question by identifying the two Pokemon names in order.\n"
            "Format your answer as: <answer>Pokemon1 and Pokemon2</answer>"
        )
    if question_type == INTERSECTION and history_label == "Verbose-Response":
        return (
            "You are reviewing a guess-the-Pokemon dialogue. Full history messages (including tool results) are provided; "
            "infer the intersection list for the target round's tool call. "
            "Each round is defined as: user guess -> optional tool call -> feedback. "
            "The first round has no tool call; the first tool call appears after the user's second guess (called round 2), and so on. "
            "Return only the intersection as a comma-separated list or JSON array. Do not call any tools. "
            "Wrap the final list in <answer></answer>."
        )
    if question_type == INTERSECTION and history_label == "Concise-Response":
        return (
            "You are an expert analyst for a deductive reasoning game. "
            "The full conversation history with system feedback is provided.\n"
            "Your task is to analyze the logical progression and constraints revealed throughout the dialogue "
            "to deduce the hidden target Pokemon.\n"
            "The correct answer must be logically consistent with the entire history of feedback.\n"
            "Return only the Pokemon name. Do not call any tools. Wrap your final answer in <answer></answer>."
        )
    return (
        "You are analyzing a guess-the-Pokemon dialogue. Full conversation history is provided. "
        "Answer the question and wrap your final answer in <answer></answer>."
    )


def _build_masked_prompt(question_type: str, history_label: str) -> str:
    if question_type in {COUNT_FREQUENCY_TOOL, COUNT_CORRECTNESS_ENV}:
        return (
            "You are analyzing a masked guess-the-entity dialogue. Full conversation history (including tool results and feedback) is provided. "
            "Answer the question based on the tool return values or environment feedback. "
            "Wrap your final answer in <answer></answer>."
        )
    if question_type == COUNT_FREQUENCY_ENV:
        return (
            "You are analyzing a masked guess-the-entity dialogue. Full conversation history with feedback is provided.\n"
            "Answer the question by counting occurrences of a property value across all rounds' feedback.\n"
            "Wrap your final answer (a number) in <answer></answer>."
        )
    if question_type == FIND_ROUND_LARGEST_VALUE_ENV:
        return (
            "You are analyzing a masked guess-the-entity dialogue. Full conversation history with feedback is provided.\n"
            "Answer the question by identifying which round has the highest attr_2 total (numeric field).\n"
            "Wrap your final answer (round number) in <answer></answer>."
        )
    if question_type == WEIGHTED_SUMMATION_ENV:
        return (
            "You are analyzing a masked guess-the-entity dialogue. Full conversation history with feedback is provided.\n"
            "Calculate the weighted scores for two rounds using this weighted rule:\n"
            "- attr_1: 6 points per correct item\n"
            "- attr_4: 5 points per correct item\n"
            "- attr_2: 4 points per correct item\n"
            "- attr_5: 3 points per correct item\n"
            "- attr_3: 2 points per correct item\n"
            "- attr_6: 1 point per correct item\n"
            "Example: If a round has attr_1 correct (6) + 2 attr_4 items correct (5+5) + 1 attr_5 correct (3), score = 6+5+5+3 = 19.\n"
            "Then compute the absolute difference between the two rounds' scores.\n"
            "Wrap your final answer (difference value) in <answer></answer>."
        )
    if question_type == FIND_DUPLICATES_TOOL:
        return (
            "You are analyzing a masked guess-the-entity dialogue. Full conversation history is provided.\n"
            "Answer the question with yes/no or true/false based on whether an entity id appears in both tool results.\n"
            "Wrap your final answer in <answer></answer>."
        )
    if question_type == FIND_TARGET_OFFSETS_TOOL:
        return (
            "You are analyzing a masked guess-the-entity dialogue. Full conversation history is provided.\n"
            "Answer the question by identifying the two entity ids in order.\n"
            "Format your answer as: <answer>id1 and id2</answer>"
        )
    if question_type == INTERSECTION and history_label == "Verbose-Response":
        return (
            "You are reviewing a masked guess-the-entity dialogue. Full history messages (including tool results) are provided; "
            "infer the intersection list for the target round's tool call. "
            "Each round is defined as: user guess -> optional tool call -> feedback. "
            "The first round has no tool call; the first tool call appears after the user's second guess (called round 2), and so on. "
            "Return only the intersection as a comma-separated list or JSON array. Do not call any tools. "
            "Wrap the final list in <answer></answer>."
        )
    if question_type == INTERSECTION and history_label == "Concise-Response":
        return (
            "You are an expert analyst for a deductive reasoning game with masked ids. "
            "The full conversation history with system feedback is provided.\n"
            "Analyze the constraints to deduce the hidden target id (the intersection-consistent answer).\n"
            "Return only the masked id. Do not call any tools. Wrap your final answer in <answer></answer>."
        )
    return (
        "You are analyzing a masked guess-the-entity dialogue. Full conversation history is provided. "
        "Answer the question and wrap your final answer in <answer></answer>."
    )


def parse_response(
    question_type: str,
    history_label: str,
    raw_text: str,
) -> Tuple[Optional[Union[int, bool, List[str], str]], str]:
    if question_type in {COUNT_FREQUENCY_TOOL, COUNT_CORRECTNESS_ENV, COUNT_FREQUENCY_ENV,
                         FIND_ROUND_LARGEST_VALUE_ENV, WEIGHTED_SUMMATION_ENV}:
        return parse_number(raw_text), "number"
    if question_type == FIND_DUPLICATES_TOOL:
        return parse_boolean(raw_text), "boolean"
    if question_type == FIND_TARGET_OFFSETS_TOOL:
        return parse_pair_list(raw_text), "list"
    if question_type == INTERSECTION and history_label == "Verbose-Response":
        return parse_intersection_list(raw_text), "intersection_list"
    if question_type == INTERSECTION and history_label == "Concise-Response":
        return parse_final_guess(raw_text), "final_answer"
    return None, "unknown"
